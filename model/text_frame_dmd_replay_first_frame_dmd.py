import torch
import torch.nn.functional as F

from model.text_frame_dmd_replay import TextFrameDMDReplay
from utils.wan_wrapper import WanDiffusionWrapper


class TextFrameDMDReplayFirstFrameDMD(TextFrameDMDReplay):
    """Text FrameDMD replay with a separate base-model DMD for its first chunk."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self.critic_training_distribution = getattr(
            args, "critic_training_distribution", "replay"
        )
        if self.critic_training_distribution not in ("replay", "rollout"):
            raise ValueError("critic_training_distribution must be replay or rollout")
        self.first_frame_loss_only = bool(
            getattr(args, "first_frame_loss_only", False)
        )
        self.first_chunk_dmd_frames = (
            1 if self.first_frame_loss_only else self.conditioning_frames
        )
        self.first_frame_dmd_loss_weight = float(
            getattr(args, "first_frame_dmd_loss_weight", 1.0)
        )
        if self.first_frame_dmd_loss_weight <= 0.0:
            raise ValueError("first_frame_dmd_loss_weight must be positive")
        self.first_frame_real_guidance_scale = float(
            getattr(
                args,
                "first_frame_real_guidance_scale",
                self.real_guidance_scale,
            )
        )

        first_frame_real_name = getattr(
            args, "first_frame_real_name", "Wan2.1-T2V-14B"
        )
        first_frame_fake_name = getattr(
            args, "first_frame_fake_name", "Wan2.1-T2V-1.3B"
        )
        self.first_frame_real_score = WanDiffusionWrapper(
            model_name=first_frame_real_name,
            is_causal=False,
            **getattr(args, "first_frame_real_score_model_kwargs", {}),
        )
        self.first_frame_real_score.model.requires_grad_(False)
        self.first_frame_fake_score = WanDiffusionWrapper(
            model_name=first_frame_fake_name,
            is_causal=False,
            **getattr(args, "first_frame_fake_score_model_kwargs", {}),
        )
        self.first_frame_fake_score.model.requires_grad_(True)
        if args.gradient_checkpointing:
            self.first_frame_fake_score.enable_gradient_checkpointing()

        if getattr(args, "use_score_model_schedulers", False):
            self.first_frame_dmd_scheduler = (
                self.first_frame_real_score.get_scheduler()
            )
            self.first_frame_critic_scheduler = (
                self.first_frame_fake_score.get_scheduler()
            )
        else:
            self.first_frame_dmd_scheduler = self.scheduler
            self.first_frame_critic_scheduler = self.scheduler

        for score_scheduler in (
            self.first_frame_dmd_scheduler,
            self.first_frame_critic_scheduler,
        ):
            if getattr(score_scheduler, "alphas_cumprod", None) is not None:
                score_scheduler.alphas_cumprod = (
                    score_scheduler.alphas_cumprod.to(device)
                )

    @staticmethod
    def combine_frame_losses(
        first_frame_loss: torch.Tensor,
        replay_loss: torch.Tensor,
        replay_frames: int,
        first_frame_frames: int = 1,
        first_frame_weight: float = 1.0,
    ) -> torch.Tensor:
        """Combine prefix and later losses as a normalized weighted average."""
        if replay_frames <= 0 or first_frame_frames <= 0:
            raise ValueError("prefix and replay frame counts must be positive")
        if first_frame_weight <= 0.0:
            raise ValueError("first_frame_weight must be positive")
        weighted_first_frames = first_frame_frames * first_frame_weight
        return (
            first_frame_loss * weighted_first_frames
            + replay_loss * replay_frames
        ) / (replay_frames + weighted_first_frames)

    def _compute_first_frame_dmd_loss(
        self,
        replay_pred: torch.Tensor,
        rollout: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        denoised_timestep_from: int,
        denoised_timestep_to: int,
    ):
        first_chunk_frames = self.first_chunk_dmd_frames
        if (
            replay_pred.shape[1] != first_chunk_frames
            or rollout.shape[1] != first_chunk_frames
        ):
            raise ValueError(
                "First-chunk DMD expects exactly "
                f"{first_chunk_frames} frames"
            )

        score_frame = rollout.detach()
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
                score_frame.shape[0],
                first_chunk_frames,
                first_chunk_frames,
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
            noise = torch.randn_like(score_frame)
            noisy = self.first_frame_dmd_scheduler.add_noise(
                score_frame.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, score_frame.shape[:2])
            grad, log = self._compute_kl_grad(
                noisy_image_or_video=noisy,
                estimated_clean_image_or_video=score_frame,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                real_score=self.first_frame_real_score,
                fake_score=self.first_frame_fake_score,
                real_guidance_scale=self.first_frame_real_guidance_scale,
            )
            target = (score_frame.double() - grad.double()).detach()

        loss = 0.5 * F.mse_loss(replay_pred.double(), target)
        log["dmd_frames"] = torch.tensor(
            first_chunk_frames, device=score_frame.device
        )
        return loss, log

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
            rollout, trajectory, conditional_dict
        )

        first_chunk_frames = self.conditioning_frames
        first_chunk_dmd_frames = getattr(
            self, "first_chunk_dmd_frames", first_chunk_frames
        )
        replay_loss, later_log = self._compute_replay_dmd_loss(
            replay_pred=replay_pred[:, first_chunk_frames:],
            rollout_generated=rollout[:, first_chunk_frames:],
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )
        first_frame_loss, first_log = self._compute_first_frame_dmd_loss(
            replay_pred=replay_pred[:, :first_chunk_dmd_frames],
            rollout=rollout[:, :first_chunk_dmd_frames],
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )

        replay_frames = replay_pred.shape[1] - first_chunk_frames
        expected_replay_frames = self.num_training_frames - first_chunk_frames
        if replay_frames != expected_replay_frames:
            raise RuntimeError(
                f"Expected {expected_replay_frames} replay-supervised frames, "
                f"got {replay_frames}"
            )
        loss = self.combine_frame_losses(
            first_frame_loss,
            replay_loss,
            replay_frames,
            first_frame_frames=first_chunk_dmd_frames,
            first_frame_weight=getattr(
                self, "first_frame_dmd_loss_weight", 1.0
            ),
        )

        log = dict(later_log)
        for key in (
            "dmdtrain_gradient_norm",
            "dmd_score_delta_abs_mean",
            "dmd_score_delta_rms",
            "dmd_normalizer_mean",
        ):
            log[key] = self.combine_frame_losses(
                first_log[key],
                later_log[key],
                replay_frames,
                first_frame_frames=first_chunk_dmd_frames,
            )
        log["dmd_normalizer_min"] = torch.minimum(
            first_log["dmd_normalizer_min"],
            later_log["dmd_normalizer_min"],
        )
        log["dmd_normalizer_max"] = torch.maximum(
            first_log["dmd_normalizer_max"],
            later_log["dmd_normalizer_max"],
        )
        first_timestep = first_log["timestep"]
        later_timestep = later_log["timestep"]
        if first_timestep.shape[1:] == later_timestep.shape[1:]:
            combined_timestep = torch.cat(
                [first_timestep, later_timestep], dim=0
            )
        else:
            # A first-frame-only prefix has width 1 while later score chunks
            # have width 3. Keep all sampled values without padding the log.
            combined_timestep = torch.cat(
                [first_timestep.reshape(-1), later_timestep.reshape(-1)]
            )
        log.update(
            {
                "timestep": combined_timestep,
                "dmd_frames": torch.tensor(
                    replay_frames + first_chunk_dmd_frames,
                    device=rollout.device,
                ),
                "dmd_score_chunks": torch.tensor(
                    replay_frames // self.score_chunk_size + 1,
                    device=rollout.device,
                ),
                "first_frame_dmd_loss": first_frame_loss.detach(),
                "first_frame_dmd_loss_weight": torch.tensor(
                    getattr(self, "first_frame_dmd_loss_weight", 1.0),
                    device=rollout.device,
                ),
                "replay_dmd_loss": replay_loss.detach(),
                "first_frame_dmd_gradient_norm": first_log[
                    "dmdtrain_gradient_norm"
                ],
                "replay_dmd_gradient_norm": later_log[
                    "dmdtrain_gradient_norm"
                ],
                "first_frame_dmd_score_delta_abs_mean": first_log[
                    "dmd_score_delta_abs_mean"
                ],
                "replay_dmd_score_delta_abs_mean": later_log[
                    "dmd_score_delta_abs_mean"
                ],
            }
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
                rollout, trajectory, conditional_dict
            )

        first_chunk_frames = self.conditioning_frames
        first_chunk_dmd_frames = getattr(
            self, "first_chunk_dmd_frames", first_chunk_frames
        )
        # Keep the replay forward in both variants so this ablation changes
        # only critic targets, without changing random-number consumption.
        critic_samples = (
            rollout
            if getattr(self, "critic_training_distribution", "replay") == "rollout"
            else replay_pred
        )
        replay_critic_loss, later_log = self._critic_loss_from_generated(
            generated_video=critic_samples[:, first_chunk_frames:],
            conditional_dict=conditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
        )
        first_frame_critic_loss, first_log = self._critic_loss_from_generated(
            generated_video=critic_samples[:, :first_chunk_dmd_frames],
            conditional_dict=conditional_dict,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
            fake_score=self.first_frame_fake_score,
            critic_scheduler=self.first_frame_critic_scheduler,
            score_chunk_size=first_chunk_dmd_frames,
        )

        loss = replay_critic_loss + first_frame_critic_loss
        log = dict(later_log)
        log.update(
            {
                "replay_critic_loss": replay_critic_loss.detach(),
                "first_frame_critic_loss": first_frame_critic_loss.detach(),
                "first_frame_critic_timestep": first_log[
                    "critic_timestep"
                ],
                "first_frame_critic_frames": first_log["critic_frames"],
                "critic_context_frames": torch.tensor(
                    0, device=rollout.device
                ),
                "tf_replay_passes": torch.tensor(1, device=rollout.device),
            }
        )
        log.update(replay_log)
        return loss, log
