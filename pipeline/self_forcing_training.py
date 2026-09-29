from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import re
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 first_denoising_step_only: bool = False,
                 exit_last_n_steps: Optional[int] = None,
                 random_first_block_exit: bool = False,
                 first_block_first_denoising_step: bool = False,
                 fixed_first_block_exit_steps_path: Optional[str] = None,
                 fixed_first_block_exit_steps_limit: Optional[int] = None,
                 configured_denoising_step_labels: Optional[List[int]] = None,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 skip_final_cache_update: bool = False,
                 rollout_hide_first_frame_after: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference
        self.configured_denoising_step_labels = (
            [int(step) for step in configured_denoising_step_labels]
            if configured_denoising_step_labels is not None
            else [int(step) for step in self.denoising_step_list]
        )
        if self.configured_denoising_step_labels[-1] == 0:
            self.configured_denoising_step_labels = (
                self.configured_denoising_step_labels[:-1]
            )

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.skip_final_cache_update = skip_final_cache_update
        self.rollout_hide_first_frame_after = rollout_hide_first_frame_after
        if (
            self.rollout_hide_first_frame_after is not None
            and self.rollout_hide_first_frame_after < 0
        ):
            raise ValueError("rollout_hide_first_frame_after must be non-negative")
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.first_denoising_step_only = bool(first_denoising_step_only)
        self.random_first_block_exit = bool(random_first_block_exit)
        self.first_block_first_denoising_step = bool(
            first_block_first_denoising_step
        )
        self.fixed_first_block_exit_steps = (
            self._load_fixed_first_block_exit_steps(
                fixed_first_block_exit_steps_path,
                fixed_first_block_exit_steps_limit,
            )
            if fixed_first_block_exit_steps_path is not None
            else None
        )
        self.fixed_first_block_exit_cursor = 0
        self.exit_last_n_steps = (
            int(exit_last_n_steps) if exit_last_n_steps is not None else None
        )
        if self.exit_last_n_steps is not None and self.exit_last_n_steps <= 0:
            raise ValueError("exit_last_n_steps must be positive when set")
        if self.first_denoising_step_only and self.last_step_only:
            raise ValueError(
                "first_denoising_step_only and last_step_only are mutually "
                "exclusive"
            )
        if self.random_first_block_exit and not self.last_step_only:
            raise ValueError(
                "random_first_block_exit requires last_step_only=true so "
                "all later blocks remain fixed at the final denoising step"
            )
        if self.first_block_first_denoising_step and not self.last_step_only:
            raise ValueError(
                "first_block_first_denoising_step requires last_step_only=true "
                "so all later blocks remain fixed at the final denoising step"
            )
        if self.random_first_block_exit and self.first_block_first_denoising_step:
            raise ValueError(
                "random_first_block_exit and first_block_first_denoising_step "
                "are mutually exclusive"
            )
        if self.fixed_first_block_exit_steps is not None and not (
            self.random_first_block_exit and self.last_step_only
        ):
            raise ValueError(
                "fixed_first_block_exit_steps_path requires "
                "random_first_block_exit=true and last_step_only=true"
            )
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    @staticmethod
    def _load_fixed_first_block_exit_steps(path, limit):
        pattern = re.compile(
            r"\[train\] finished step=(\d+).*?"
            r"first_block_exit_step=(\d+)"
        )
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = pattern.search(line)
                if match is not None:
                    rows.append((int(match.group(1)), int(match.group(2))))

        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise ValueError(
                    "fixed_first_block_exit_steps_limit must be positive"
                )
            rows = [row for row in rows if row[0] <= limit]
        if not rows:
            raise ValueError(f"No fixed exit steps found in {path}")

        expected_steps = list(range(1, len(rows) + 1))
        observed_steps = [step for step, _ in rows]
        if observed_steps != expected_steps:
            raise ValueError(
                f"Fixed exit-step trajectory in {path} must contain exactly "
                f"one ordered row for every step from 1; got "
                f"{observed_steps[:3]}...{observed_steps[-3:]}"
            )
        if limit is not None and len(rows) != limit:
            raise ValueError(
                f"Fixed exit-step trajectory in {path} contains {len(rows)} "
                f"steps through requested limit {limit}"
            )
        return [exit_step for _, exit_step in rows]

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0 and self.fixed_first_block_exit_steps is not None:
            if self.fixed_first_block_exit_cursor >= len(
                self.fixed_first_block_exit_steps
            ):
                raise RuntimeError(
                    "Fixed first-block exit-step trajectory exhausted after "
                    f"{len(self.fixed_first_block_exit_steps)} calls; refusing "
                    "to fall back to random sampling"
                )
            fixed_step = self.fixed_first_block_exit_steps[
                self.fixed_first_block_exit_cursor
            ]
            self.fixed_first_block_exit_cursor += 1
            configured_labels = self.configured_denoising_step_labels
            if len(configured_labels) != num_denoising_steps:
                raise ValueError(
                    "configured_denoising_step_labels and the effective "
                    "denoising_step_list must have the same length"
                )
            if fixed_step not in configured_labels:
                raise ValueError(
                    f"Fixed exit step {fixed_step} is not present in "
                    f"configured_denoising_step_labels={configured_labels}"
                )
            # The training log stores the original YAML labels. Select by
            # position so timestep warping does not change the replayed exit.
            fixed_index = configured_labels.index(fixed_step)
            indices = torch.full(
                (num_blocks,),
                fill_value=num_denoising_steps - 1,
                dtype=torch.long,
                device=device,
            )
            indices[0] = fixed_index
        elif rank == 0:
            # Generate random indices.
            low = 0
            if self.exit_last_n_steps is not None:
                if self.exit_last_n_steps > num_denoising_steps:
                    raise ValueError(
                        "exit_last_n_steps cannot exceed num_denoising_steps"
                    )
                low = num_denoising_steps - self.exit_last_n_steps
            indices = torch.randint(
                low=low,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            first_block_exit = indices[0].clone()
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
            if self.first_denoising_step_only:
                indices = torch.zeros_like(indices)
            if self.random_first_block_exit:
                indices[0] = first_block_exit
            if self.first_block_first_denoising_step:
                indices[0] = 0
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast indices to all ranks.
        exit_indices = indices.tolist()
        # Retain the synchronized choice so the trainer can associate it with
        # the current optimization step without printing once per rank.
        self.last_exit_indices = exit_indices
        return exit_indices

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            return_replay_trajectory: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            if num_input_frames % self.num_frame_per_block != 0:
                raise ValueError(
                    f"Initial latent has {num_input_frames} frames, which is not "
                    f"divisible by block size {self.num_frame_per_block}"
                )
            timestep = torch.zeros(
                [batch_size, num_input_frames],
                device=noise.device,
                dtype=torch.int64,
            )
            output[:, :num_input_frames] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += num_input_frames

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21
        replay_noisy_latents = []
        replay_timesteps = []

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            kv_cache_attention_start = (
                self.frame_seq_length
                if self.rollout_hide_first_frame_after is not None
                and current_start_frame > self.rollout_hide_first_frame_after
                else None
            )
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if (
                    self.random_first_block_exit
                    or self.first_block_first_denoising_step
                ):
                    exit_flag = (index == exit_flags[block_index])
                elif self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            kv_cache_attention_start=kv_cache_attention_start,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    if return_replay_trajectory:
                        replay_noisy_latents.append(noisy_input.detach())
                        replay_timesteps.append(timestep.detach())
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                kv_cache_attention_start=kv_cache_attention_start,
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            kv_cache_attention_start=kv_cache_attention_start,
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache. The cache
            # is not consumed after the final block, so specialized short
            # training paths can skip that otherwise redundant forward.
            is_final_block = block_index == len(all_num_frames) - 1
            if not (self.skip_final_cache_update and is_final_block):
                context_timestep = torch.ones_like(timestep) * self.context_noise
                # add context noise
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    context_timestep * torch.ones(
                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        kv_cache_attention_start=kv_cache_attention_start,
                    )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if (
            not self.same_step_across_blocks
            or self.random_first_block_exit
            or self.first_block_first_denoising_step
        ):
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            scheduler_timesteps = self.scheduler.timesteps.to(device=noise.device)
            denoising_step = self.denoising_step_list[exit_flags[0]].to(device=noise.device)
            denoised_timestep_from = 1000 - torch.argmin(
                (scheduler_timesteps - denoising_step).abs(), dim=0).item()
        else:
            scheduler_timesteps = self.scheduler.timesteps.to(device=noise.device)
            denoising_step_to = self.denoising_step_list[exit_flags[0] + 1].to(device=noise.device)
            denoising_step_from = self.denoising_step_list[exit_flags[0]].to(device=noise.device)
            denoised_timestep_to = 1000 - torch.argmin(
                (scheduler_timesteps - denoising_step_to).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (scheduler_timesteps - denoising_step_from).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        if return_replay_trajectory:
            if len(replay_noisy_latents) != len(all_num_frames):
                raise RuntimeError(
                    "Replay trajectory did not capture one exit latent per generated block"
                )
            trajectory = {
                "noisy_latents": torch.cat(replay_noisy_latents, dim=1),
                "timesteps": torch.cat(replay_timesteps, dim=1),
                "rollout_x0": output[:, num_input_frames:].detach(),
                "full_rollout_x0": output.detach(),
            }
            return output, denoised_timestep_from, denoised_timestep_to, trajectory

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
