from typing import Tuple

import torch
import torch.nn.functional as F

from model.dmd import DMD


class VideoDMDReplay(DMD):
    """Joint-video DMD with a causal full-sequence teacher-forcing replay."""

    def __init__(self, args, device):
        super().__init__(args, device)
        if self.num_training_frames != 21:
            raise ValueError("VideoDMDReplay requires num_training_frames=21")
        if self.num_training_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "VideoDMDReplay requires num_training_frames to be divisible "
                "by num_frame_per_block"
            )
        if int(getattr(args, "tf_replay_passes", 1)) != 1:
            raise ValueError("VideoDMDReplay currently supports exactly one TF replay")
        self.replay_hide_first_frame_after = int(
            getattr(args, "tf_replay_hide_first_frame_after", 10)
        )
        # Opt in: existing VideoDMD runs keep their original visibility.
        self.replay_hide_first_frame_prob = float(
            getattr(args, "tf_replay_hide_first_frame_prob", 0.0)
        )
        self.replay_hidden_prefix_frames = int(
            getattr(args, "tf_replay_hidden_prefix_frames", self.num_frame_per_block)
        )
        if not 0.0 <= self.replay_hide_first_frame_prob <= 1.0:
            raise ValueError("tf_replay_hide_first_frame_prob must be in [0, 1]")
        if not 1 <= self.replay_hidden_prefix_frames <= self.num_frame_per_block:
            raise ValueError("Hidden prefix must fit within the first rollout block")

    def _replay_visibility(self, batch_size, num_frames, device):
        if self.replay_hide_first_frame_prob == 0.0:
            return None
        # Match TextFrameDMDReplay: sample once per later query block,
        # use a strict block-start > threshold, and always keep block zero.
        starts = torch.arange(
            self.num_frame_per_block, num_frames, self.num_frame_per_block,
            device=device,
        )
        hidden = (starts > self.replay_hide_first_frame_after).unsqueeze(0) & (
            torch.rand(batch_size, starts.numel(), device=device)
            < self.replay_hide_first_frame_prob
        )
        return torch.cat([
            torch.ones(batch_size, self.num_frame_per_block,
                       dtype=torch.bool, device=device),
            (~hidden).repeat_interleave(self.num_frame_per_block, dim=1),
        ], dim=1)

    def _run_rollout_with_trajectory(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict, int, int]:
        if initial_latent is not None:
            raise ValueError("VideoDMDReplay is text-to-video and does not use a GT first frame")
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
        # Replay is a full-sequence forward and must not retain rollout caches.
        self.inference_pipeline.kv_cache1 = None
        self.inference_pipeline.kv_cache2 = None
        self.inference_pipeline.crossattn_cache = None

        output = output.to(self.dtype)
        if output.shape != noise.shape:
            raise RuntimeError(
                f"Rollout shape {tuple(output.shape)} does not match noise shape {tuple(noise.shape)}"
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
    ) -> torch.Tensor:
        replay_noisy = trajectory["noisy_latents"]
        replay_timestep = trajectory["timesteps"]
        visibility = self._replay_visibility(
            replay_noisy.shape[0], replay_noisy.shape[1], replay_noisy.device
        )
        _, replay_pred_x0 = self.generator(
            noisy_image_or_video=replay_noisy,
            conditional_dict=conditional_dict,
            timestep=replay_timestep,
            clean_x=rollout.detach(),
            aug_t=torch.zeros_like(replay_timestep),
            teacher_forcing_first_frame_visibility_mask=visibility,
            teacher_forcing_hidden_prefix_frames=self.replay_hidden_prefix_frames,
        )
        if replay_pred_x0.shape != rollout.shape:
            raise RuntimeError("TF replay output shape does not match the rollout")
        return replay_pred_x0.to(self.dtype)

    def _compute_replay_video_dmd_loss(
        self,
        replay_pred: torch.Tensor,
        rollout: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        denoised_timestep_from: int,
        denoised_timestep_to: int,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frames = rollout.shape[:2]
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
        with torch.no_grad():
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frames,
                self.num_frame_per_block,
                uniform_timestep=True,
            )
            if self.timestep_shift > 1:
                timestep = (
                    self.timestep_shift
                    * (timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (timestep / 1000))
                    * 1000
                )
            timestep = timestep.clamp(self.min_step, self.max_step)
            score_video = rollout.detach()
            noise = torch.randn_like(score_video)
            noisy_video = self.dmd_scheduler.add_noise(
                score_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frames))
            grad, log = self._compute_kl_grad(
                noisy_image_or_video=noisy_video,
                estimated_clean_image_or_video=score_video,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )
            target = (score_video.double() - grad.double()).detach()

        loss = 0.5 * F.mse_loss(replay_pred.double(), target)
        log.update(
            {
                "dmd_frames": torch.tensor(num_frames, device=rollout.device),
                "tf_replay_passes": torch.tensor(1, device=rollout.device),
            }
        )
        return loss, log

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        del clean_latent
        with torch.no_grad():
            rollout, trajectory, timestep_from, timestep_to = (
                self._run_rollout_with_trajectory(
                    image_or_video_shape, conditional_dict, initial_latent
                )
            )
        replay_pred = self._teacher_forcing_replay(
            rollout, trajectory, conditional_dict
        )
        return self._compute_replay_video_dmd_loss(
            replay_pred=replay_pred,
            rollout=rollout,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        del unconditional_dict, clean_latent
        with torch.no_grad():
            rollout, trajectory, timestep_from, timestep_to = (
                self._run_rollout_with_trajectory(
                    image_or_video_shape, conditional_dict, initial_latent
                )
            )
            replay_pred = self._teacher_forcing_replay(
                rollout, trajectory, conditional_dict
            )
        loss, log = self._critic_loss_from_generated(
            generated_image=replay_pred,
            conditional_dict=conditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )
        log["tf_replay_passes"] = torch.tensor(1, device=replay_pred.device)
        return loss, log
