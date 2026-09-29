from typing import Tuple

import torch
import torch.distributed as dist

from model.frame_dmd_replay import FrameDMDReplay


class TextFrameDMDReplay(FrameDMDReplay):
    """Text-to-video FrameDMD replay that excludes its first context chunk.

    By default the generator rolls out all 21 frames without a ground-truth
    initial latent. Optionally, a synchronized per-update draw can replace the
    first generated chunk with a ground-truth conditioning chunk. In both modes,
    the first chunk remains detached teacher-forcing context for later chunks,
    optionally hidden after a configured frame, while the standard generator DMD
    and fake-score objectives cover only the remaining chunks.
    """

    def __init__(self, args, device):
        super().__init__(args, device)
        if self.conditioning_frames != self.num_frame_per_block:
            raise ValueError(
                "TextFrameDMDReplay requires conditioning_frames == "
                "num_frame_per_block so the replay prefix is one complete "
                "generator rollout chunk"
            )
        self.hide_generated_first_chunk_in_replay = bool(
            getattr(
                args,
                "text_tf_replay_hide_generated_first_chunk",
                getattr(args, "text_tf_replay_hide_generated_first_frame", False),
            )
        )
        self.gt_first_conditioning_probability = float(
            getattr(args, "gt_first_conditioning_probability", 0.0)
        )
        if not 0.0 <= self.gt_first_conditioning_probability <= 1.0:
            raise ValueError(
                "gt_first_conditioning_probability must be in [0, 1]"
            )
        if self.gt_first_conditioning_probability > 0.0 and not args.i2v:
            raise ValueError(
                "gt_first_conditioning_probability > 0 requires i2v=true so "
                "the batch supplies a ground-truth first latent"
            )

    def _select_rollout_initial_latent(
        self, initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, bool]:
        """Choose GT or generated first context consistently on every rank."""
        probability = self.gt_first_conditioning_probability
        if probability <= 0.0:
            return None, False
        if initial_latent is None:
            raise ValueError(
                "GT-first sampling requires an initial latent from the batch"
            )
        if probability >= 1.0:
            return initial_latent, True

        draw = torch.zeros((), device=self.device, dtype=torch.float32)
        distributed = dist.is_available() and dist.is_initialized()
        if not distributed or dist.get_rank() == 0:
            draw.copy_(torch.rand((), device=self.device))
        if distributed:
            dist.broadcast(draw, src=0)
        use_gt_first = bool(draw.item() < probability)
        return (initial_latent if use_gt_first else None), use_gt_first

    def _run_rollout_with_trajectory(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict, int, int]:
        if initial_latent is not None:
            generated, trajectory, timestep_from, timestep_to = (
                FrameDMDReplay._run_rollout_with_trajectory(
                    self,
                    image_or_video_shape,
                    conditional_dict,
                    initial_latent,
                )
            )
            rollout = torch.cat([initial_latent.detach(), generated], dim=1)
            return rollout, trajectory, timestep_from, timestep_to
        batch_size, total_frames, channels, height, width = image_or_video_shape
        if total_frames != self.num_training_frames:
            raise ValueError(
                f"Expected {self.num_training_frames} frames, got {total_frames}"
            )
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        noise = torch.randn(
            [batch_size, total_frames, channels, height, width],
            device=self.device,
            dtype=self.dtype,
        )
        output, timestep_from, timestep_to, trajectory = (
            self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                initial_latent=None,
                return_replay_trajectory=True,
                **conditional_dict,
            )
        )
        self.inference_pipeline.kv_cache1 = None
        self.inference_pipeline.kv_cache2 = None
        self.inference_pipeline.crossattn_cache = None

        output = output.to(self.dtype)
        if output.shape != noise.shape:
            raise RuntimeError(
                f"Rollout shape {tuple(output.shape)} does not match noise shape "
                f"{tuple(noise.shape)}"
            )
        if trajectory["noisy_latents"].shape != output.shape:
            raise RuntimeError("Replay noisy latents do not cover all rollout frames")
        if trajectory["timesteps"].shape != output.shape[:2]:
            raise RuntimeError("Replay timesteps do not cover all rollout frames")
        return output, trajectory, timestep_from, timestep_to

    def _teacher_forcing_replay(
        self,
        rollout: torch.Tensor,
        trajectory: dict,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        if initial_latent is not None:
            replay_generated, diagnostics = (
                FrameDMDReplay._teacher_forcing_replay(
                    self,
                    initial_latent,
                    rollout[:, self.conditioning_frames :],
                    trajectory,
                    conditional_dict,
                )
            )
            replay = torch.cat(
                [initial_latent.detach(), replay_generated], dim=1
            )
            return replay, diagnostics
        replay_noisy = trajectory["noisy_latents"]
        replay_timestep = trajectory["timesteps"]
        visibility_arg = None
        diagnostics = {}
        if self.hide_generated_first_chunk_in_replay:
            # The first generated chunk is clean replay context. Query chunks
            # above the threshold cannot attend to that prefix, while all other
            # causal clean history remains visible.
            visibility, diagnostics = self._sample_first_frame_visibility(
                replay_noisy.shape[0],
                replay_noisy.shape[1] - self.conditioning_frames,
                replay_noisy.device,
            )
            visibility_arg = (
                visibility if self.replay_hide_first_frame_prob > 0.0 else None
            )
        _, replay_pred_x0 = self.generator(
            noisy_image_or_video=replay_noisy,
            conditional_dict=conditional_dict,
            timestep=replay_timestep,
            clean_x=rollout.detach(),
            aug_t=torch.zeros_like(replay_timestep),
            teacher_forcing_first_frame_visibility_mask=visibility_arg,
            teacher_forcing_hidden_prefix_frames=self.replay_hidden_prefix_frames,
        )
        if replay_pred_x0.shape != rollout.shape:
            raise RuntimeError("TF replay output shape does not match the rollout")
        return replay_pred_x0.to(self.dtype), diagnostics

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ):
        del clean_latent
        rollout_initial_latent, use_gt_first = (
            self._select_rollout_initial_latent(initial_latent)
        )
        with torch.no_grad():
            rollout, trajectory, timestep_from, timestep_to = (
                self._run_rollout_with_trajectory(
                    image_or_video_shape,
                    conditional_dict,
                    rollout_initial_latent,
                )
            )
        replay_pred, replay_log = self._teacher_forcing_replay(
            rollout,
            trajectory,
            conditional_dict,
            initial_latent=rollout_initial_latent,
        )

        # The first GT-or-generated context chunk has no standard chunk-ScoreNet
        # DMD loss in this base class.
        loss, log = self._compute_replay_dmd_loss(
            replay_pred=replay_pred[:, self.conditioning_frames:],
            rollout_generated=rollout[:, self.conditioning_frames:],
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )
        log["dmd_context_frames"] = torch.tensor(
            self.conditioning_frames, device=rollout.device
        )
        # Convert indices relative to the sliced replay into full-video indices.
        log["dmd_selected_frame_indices"] = (
            log["dmd_selected_frame_indices"] + self.conditioning_frames
        )
        log["gt_first_conditioning"] = torch.tensor(
            float(use_gt_first), device=rollout.device
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
        rollout_initial_latent, use_gt_first = (
            self._select_rollout_initial_latent(initial_latent)
        )
        with torch.no_grad():
            rollout, trajectory, timestep_from, timestep_to = (
                self._run_rollout_with_trajectory(
                    image_or_video_shape,
                    conditional_dict,
                    rollout_initial_latent,
                )
            )
            replay_pred, replay_log = self._teacher_forcing_replay(
                rollout,
                trajectory,
                conditional_dict,
                initial_latent=rollout_initial_latent,
            )

        # The standard fake score is likewise trained only after the prefix.
        loss, log = self._critic_loss_from_generated(
            generated_video=replay_pred[:, self.conditioning_frames:],
            conditional_dict=conditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )
        log.update(
            {
                "critic_context_frames": torch.tensor(
                    self.conditioning_frames, device=rollout.device
                ),
                "tf_replay_passes": torch.tensor(1, device=rollout.device),
                "gt_first_conditioning": torch.tensor(
                    float(use_gt_first), device=rollout.device
                ),
            }
        )
        log.update(replay_log)
        return loss, log
