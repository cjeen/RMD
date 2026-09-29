from typing import Tuple

import torch
import torch.nn.functional as F

from model.frame_dmd import FrameDMD


class FrameDMDReplay(FrameDMD):
    """FrameDMD whose trainable generator path is one teacher-forcing replay."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self.replay_hide_first_frame_after = int(
            getattr(args, "tf_replay_hide_first_frame_after", 10)
        )
        self.replay_hide_first_frame_prob = float(
            getattr(args, "tf_replay_hide_first_frame_prob", 0.5)
        )
        self.replay_dmd_timestep_shift = float(
            getattr(args, "replay_dmd_timestep_shift", self.timestep_shift)
        )
        if self.replay_dmd_timestep_shift <= 0.0:
            raise ValueError("replay_dmd_timestep_shift must be positive")
        self.replay_hidden_prefix_frames = int(
            getattr(
                args,
                "tf_replay_hidden_prefix_frames",
                self.conditioning_frames,
            )
        )
        dmd_last_n_frames = getattr(args, "dmd_last_n_frames", None)
        self.dmd_last_n_frames = (
            int(dmd_last_n_frames) if dmd_last_n_frames is not None else None
        )
        dmd_random_n_frames = getattr(args, "dmd_random_n_frames", None)
        self.dmd_random_n_frames = (
            int(dmd_random_n_frames)
            if dmd_random_n_frames is not None
            else None
        )
        if not 0.0 <= self.replay_hide_first_frame_prob <= 1.0:
            raise ValueError("tf_replay_hide_first_frame_prob must be in [0, 1]")
        if not 1 <= self.replay_hidden_prefix_frames <= self.conditioning_frames:
            raise ValueError(
                "tf_replay_hidden_prefix_frames must be between 1 and "
                "conditioning_frames"
            )
        if self.dmd_last_n_frames is not None and self.dmd_last_n_frames <= 0:
            raise ValueError("dmd_last_n_frames must be positive when set")
        if self.dmd_random_n_frames is not None:
            if self.dmd_random_n_frames <= 0:
                raise ValueError("dmd_random_n_frames must be positive when set")
            if self.dmd_random_n_frames % self.score_chunk_size != 0:
                raise ValueError(
                    "dmd_random_n_frames must be divisible by score_chunk_size"
                )
        if (
            self.dmd_last_n_frames is not None
            and self.dmd_random_n_frames is not None
        ):
            raise ValueError(
                "dmd_last_n_frames and dmd_random_n_frames are mutually exclusive"
            )

    def _select_dmd_frames(
        self,
        replay_pred: torch.Tensor,
        rollout_generated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the temporal chunks that receive generator DMD supervision."""
        num_frames = rollout_generated.shape[1]
        if replay_pred.shape[1] != num_frames:
            raise ValueError("Replay and rollout must have the same frame count")

        dmd_last_n_frames = getattr(self, "dmd_last_n_frames", None)
        dmd_random_n_frames = getattr(self, "dmd_random_n_frames", None)
        if dmd_last_n_frames is not None:
            if dmd_last_n_frames > num_frames:
                raise ValueError(
                    f"dmd_last_n_frames={dmd_last_n_frames} exceeds the "
                    f"{num_frames} generated frames"
                )
            selected_indices = torch.arange(
                num_frames - dmd_last_n_frames,
                num_frames,
                device=rollout_generated.device,
            )
        elif dmd_random_n_frames is not None:
            if dmd_random_n_frames > num_frames:
                raise ValueError(
                    f"dmd_random_n_frames={dmd_random_n_frames} exceeds the "
                    f"{num_frames} generated frames"
                )
            if num_frames % self.score_chunk_size != 0:
                raise ValueError(
                    f"Cannot split {num_frames} frames into chunks of "
                    f"{self.score_chunk_size}"
                )
            num_chunks = num_frames // self.score_chunk_size
            selected_chunks = torch.randperm(
                num_chunks, device=rollout_generated.device
            )[: dmd_random_n_frames // self.score_chunk_size]
            selected_chunks = selected_chunks.sort().values
            offsets = torch.arange(
                self.score_chunk_size, device=rollout_generated.device
            )
            selected_indices = (
                selected_chunks[:, None] * self.score_chunk_size + offsets[None, :]
            ).reshape(-1)
        else:
            selected_indices = torch.arange(
                num_frames, device=rollout_generated.device
            )

        return (
            replay_pred.index_select(1, selected_indices),
            rollout_generated.index_select(1, selected_indices),
            selected_indices,
        )

    def _run_rollout_with_trajectory(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor,
    ):
        if (
            initial_latent is None
            or initial_latent.ndim != 5
            or initial_latent.shape[1] != self.conditioning_frames
        ):
            raise ValueError(
                "FrameDMD TF replay requires initial_latent "
                f"[B, {self.conditioning_frames}, C, H, W]"
            )
        batch_size, total_frames, channels, height, width = image_or_video_shape
        if total_frames != self.num_training_frames:
            raise ValueError(f"Expected {self.num_training_frames} total frames, got {total_frames}")
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        noise = torch.randn(
            [
                batch_size,
                total_frames - self.conditioning_frames,
                channels,
                height,
                width,
            ],
            device=self.device,
            dtype=self.dtype,
        )
        output, timestep_from, timestep_to, trajectory = (
            self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                initial_latent=initial_latent,
                return_replay_trajectory=True,
                **conditional_dict,
            )
        )
        # Replay is a full-sequence teacher-forcing forward and does not reuse
        # rollout KV state. Drop the retained caches before constructing its
        # 65k-token attention mask/activations.
        self.inference_pipeline.kv_cache1 = None
        self.inference_pipeline.kv_cache2 = None
        self.inference_pipeline.crossattn_cache = None
        generated = output[:, self.conditioning_frames:].to(self.dtype)
        expected_generated = total_frames - self.conditioning_frames
        if generated.shape[1] != expected_generated:
            raise RuntimeError(
                f"TF replay rollout produced {generated.shape[1]} frames, "
                f"expected {expected_generated}"
            )
        return generated, trajectory, timestep_from, timestep_to

    def _sample_first_frame_visibility(
        self, batch_size: int, num_generated_frames: int, device: torch.device
    ) -> Tuple[torch.Tensor, dict]:
        # Replay visibility follows generator rollout boundaries. ScoreNet may
        # deliberately evaluate smaller independent chunks (for example,
        # framewise scores over a three-frame generator rollout).
        rollout_chunk_size = int(
            getattr(self, "num_frame_per_block", self.score_chunk_size)
        )
        if rollout_chunk_size <= 0:
            raise ValueError("num_frame_per_block must be positive")
        generated_block_starts = torch.arange(
            self.conditioning_frames,
            self.conditioning_frames + num_generated_frames,
            rollout_chunk_size,
            device=device,
        )
        if num_generated_frames % rollout_chunk_size != 0:
            raise ValueError(
                f"Cannot split {num_generated_frames} generated frames into "
                f"rollout chunks of {rollout_chunk_size}"
            )
        candidates = generated_block_starts > self.replay_hide_first_frame_after
        hidden_blocks = candidates.unsqueeze(0) & (
            torch.rand(batch_size, candidates.numel(), device=device)
            < self.replay_hide_first_frame_prob
        )
        hidden = hidden_blocks.repeat_interleave(rollout_chunk_size, dim=1)
        generated_visibility = ~hidden
        full_visibility = torch.cat(
            [
                torch.ones(
                    batch_size,
                    self.conditioning_frames,
                    device=device,
                    dtype=torch.bool,
                ),
                generated_visibility,
            ],
            dim=1,
        )
        return full_visibility, {
            "tf_replay_hidden_frames": hidden.float().sum().detach(),
            "tf_replay_hidden_fraction": hidden.float().mean().detach(),
        }

    def _teacher_forcing_replay(
        self,
        initial_latent: torch.Tensor,
        rollout_generated: torch.Tensor,
        trajectory: dict,
        conditional_dict: dict,
    ):
        noisy_generated = trajectory["noisy_latents"]
        generated_timesteps = trajectory["timesteps"]
        if noisy_generated.shape != rollout_generated.shape:
            raise RuntimeError("Replay noisy-latent shape does not match rollout shape")
        if generated_timesteps.shape != rollout_generated.shape[:2]:
            raise RuntimeError("Replay timestep shape does not match rollout frames")

        replay_noisy = torch.cat([initial_latent.detach(), noisy_generated], dim=1)
        replay_clean = torch.cat([initial_latent.detach(), rollout_generated.detach()], dim=1)
        replay_timestep = torch.cat(
            [
                torch.zeros(
                    generated_timesteps.shape[0],
                    self.conditioning_frames,
                    device=generated_timesteps.device,
                    dtype=generated_timesteps.dtype,
                ),
                generated_timesteps,
            ],
            dim=1,
        )
        visibility, diagnostics = self._sample_first_frame_visibility(
            replay_noisy.shape[0], rollout_generated.shape[1], replay_noisy.device
        )
        # With hiding disabled, use the ordinary cached TF attention mask.
        # This is equivalent to an all-true visibility mask and avoids rebuilding
        # a dynamic flex-attention mask on every replay step.
        visibility_arg = (
            visibility if self.replay_hide_first_frame_prob > 0.0 else None
        )
        _, replay_pred_x0 = self.generator(
            noisy_image_or_video=replay_noisy,
            conditional_dict=conditional_dict,
            timestep=replay_timestep,
            clean_x=replay_clean,
            aug_t=torch.zeros_like(replay_timestep),
            teacher_forcing_first_frame_visibility_mask=visibility_arg,
            teacher_forcing_hidden_prefix_frames=self.replay_hidden_prefix_frames,
        )
        return replay_pred_x0[:, self.conditioning_frames:].to(self.dtype), diagnostics

    def _compute_replay_dmd_loss(
        self,
        replay_pred: torch.Tensor,
        rollout_generated: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        denoised_timestep_from: int,
        denoised_timestep_to: int,
    ):
        batch_size = rollout_generated.shape[0]
        replay_pred, rollout_generated, selected_indices = (
            self._select_dmd_frames(replay_pred, rollout_generated)
        )
        num_frames = rollout_generated.shape[1]
        replay_frames = self.flatten_independent_frames(replay_pred)
        score_frames = self.flatten_independent_frames(rollout_generated.detach())
        num_score_chunks = num_frames // self.score_chunk_size
        independent_cond = self.repeat_conditioning(
            conditional_dict, num_score_chunks, batch_size
        )
        independent_uncond = self.repeat_conditioning(
            unconditional_dict, num_score_chunks, batch_size
        )
        total = score_frames.shape[0]
        losses, gradient_norms, timesteps = [], [], []
        score_delta_abs_means, score_delta_rms_values = [], []
        normalizer_means, normalizer_mins, normalizer_maxs = [], [], []

        min_timestep = (
            denoised_timestep_to
            if self.ts_schedule and denoised_timestep_to is not None
            else self.min_score_timestep
        )
        max_timestep = (
            denoised_timestep_from
            if self.ts_schedule_max and denoised_timestep_from is not None
            else self.num_train_timestep
        )
        for start in range(0, total, self.independent_frame_chunk_size):
            end = min(total, start + self.independent_frame_chunk_size)
            score_chunk = score_frames[start:end]
            cond_chunk = self.slice_conditioning(independent_cond, start, end, total)
            uncond_chunk = self.slice_conditioning(independent_uncond, start, end, total)
            with torch.no_grad():
                timestep = self._get_timestep(
                    min_timestep,
                    max_timestep,
                    end - start,
                    self.score_chunk_size,
                    self.score_chunk_size,
                    uniform_timestep=True,
                )
                replay_shift = getattr(
                    self, "replay_dmd_timestep_shift", self.timestep_shift
                )
                if replay_shift > 1:
                    timestep = (
                        replay_shift * (timestep / 1000)
                        / (1 + (replay_shift - 1) * (timestep / 1000))
                        * 1000
                    )
                timestep = timestep.clamp(self.min_step, self.max_step)
                noise = torch.randn_like(score_chunk)
                noisy = self.dmd_scheduler.add_noise(
                    score_chunk.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                ).unflatten(0, score_chunk.shape[:2])
                grad, log = self._compute_kl_grad(
                    noisy_image_or_video=noisy,
                    estimated_clean_image_or_video=score_chunk,
                    timestep=timestep,
                    conditional_dict=cond_chunk,
                    unconditional_dict=uncond_chunk,
                )
                target = (score_chunk.double() - grad.double()).detach()

            weight = (end - start) / total
            losses.append(
                0.5 * F.mse_loss(replay_frames[start:end].double(), target) * weight
            )
            gradient_norms.append(log["dmdtrain_gradient_norm"] * weight)
            score_delta_abs_means.append(
                log["dmd_score_delta_abs_mean"] * weight
            )
            score_delta_rms_values.append(
                log["dmd_score_delta_rms"] * weight
            )
            normalizer_means.append(log["dmd_normalizer_mean"] * weight)
            normalizer_mins.append(log["dmd_normalizer_min"])
            normalizer_maxs.append(log["dmd_normalizer_max"])
            timesteps.append(log["timestep"])

        return torch.stack(losses).sum(), {
            "dmdtrain_gradient_norm": torch.stack(gradient_norms).sum(),
            "dmd_score_delta_abs_mean": torch.stack(
                score_delta_abs_means
            ).sum(),
            "dmd_score_delta_rms": torch.stack(score_delta_rms_values).sum(),
            "dmd_normalizer_mean": torch.stack(normalizer_means).sum(),
            "dmd_normalizer_min": torch.stack(normalizer_mins).min(),
            "dmd_normalizer_max": torch.stack(normalizer_maxs).max(),
            "timestep": torch.cat(timesteps, dim=0),
            "dmd_frames": torch.tensor(
                batch_size * num_frames, device=score_frames.device
            ),
            "dmd_score_chunks": torch.tensor(total, device=score_frames.device),
            "dmd_selected_frame_indices": selected_indices.detach(),
            "tf_replay_passes": torch.tensor(1, device=score_frames.device),
        }

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ):
        del clean_latent
        with torch.no_grad():
            rollout, trajectory, timestep_from, timestep_to = (
                self._run_rollout_with_trajectory(
                    image_or_video_shape, conditional_dict, initial_latent
                )
            )
        replay_pred, replay_log = self._teacher_forcing_replay(
            initial_latent, rollout, trajectory, conditional_dict
        )
        loss, log = self._compute_replay_dmd_loss(
            replay_pred=replay_pred,
            rollout_generated=rollout,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )
        log.update(replay_log)
        return loss, log

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ):
        del unconditional_dict, clean_latent
        with torch.no_grad():
            rollout, trajectory, timestep_from, timestep_to = (
                self._run_rollout_with_trajectory(
                    image_or_video_shape, conditional_dict, initial_latent
                )
            )
            replay_pred, replay_log = self._teacher_forcing_replay(
                initial_latent, rollout, trajectory, conditional_dict
            )
        loss, log = self._critic_loss_from_generated(
            generated_video=replay_pred,
            conditional_dict=conditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )
        log.update(replay_log)
        return loss, log
