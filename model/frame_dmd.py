from typing import Tuple

import torch

from model.dmd import DMD
from utils.wan_wrapper import WanDiffusionWrapper


class FrameDMD(DMD):
    """GT-first self-forcing rollout with frame-independent DMD score matching."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self._load_score_initialization(args)
        self.score_chunk_size = int(getattr(args, "score_chunk_size", 1))
        self.conditioning_frames = int(getattr(args, "conditioning_frames", 1))
        self.independent_frame_chunk_size = int(
            getattr(args, "independent_frame_chunk_size", 1)
        )
        if self.score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive")
        if self.conditioning_frames <= 0:
            raise ValueError("conditioning_frames must be positive")
        if self.independent_frame_chunk_size <= 0:
            raise ValueError("independent_frame_chunk_size must be positive")
        if self.num_training_frames != 21:
            raise ValueError("GT-first independent DMD requires num_training_frames=21")
        generated_frames = self.num_training_frames - self.conditioning_frames
        if generated_frames <= 0 or generated_frames % self.score_chunk_size != 0:
            raise ValueError(
                f"Cannot split {generated_frames} generated frames into "
                f"score chunks of {self.score_chunk_size}"
            )

    def _load_score_initialization(self, args) -> None:
        """Load the score initialization used by the matching Dual-Matching run."""
        real_score_lora_path = getattr(args, "real_score_lora_path", None)
        if real_score_lora_path:
            try:
                from peft import LoraConfig
                from peft.mapping import inject_adapter_in_model
                from peft.utils import set_peft_model_state_dict
            except (ImportError, ModuleNotFoundError) as exc:
                raise ModuleNotFoundError(
                    "FrameDMD real-score initialization requires peft>=0.12.0"
                ) from exc

            adapter_name = getattr(args, "real_score_lora_adapter_name", "default")
            lora_config = LoraConfig(
                r=int(getattr(args, "real_score_lora_rank", 128)),
                lora_alpha=int(getattr(args, "real_score_lora_rank", 128)),
                bias="none",
                init_lora_weights="gaussian",
                target_modules=list(
                    getattr(
                        args,
                        "real_score_lora_target_modules",
                        ["q", "k", "v", "o", "ffn.0", "ffn.2"],
                    )
                ),
            )
            injected_model = inject_adapter_in_model(
                lora_config,
                self.real_score.model,
                adapter_name=adapter_name,
            )
            if injected_model is not None:
                self.real_score.model = injected_model
            adapter_state = torch.load(
                real_score_lora_path,
                map_location="cpu",
                weights_only=True,
            )
            set_peft_model_state_dict(
                self.real_score.model,
                adapter_state,
                adapter_name=adapter_name,
            )
            self.real_score.model.requires_grad_(False)
            print(f"Loaded frozen real-score LoRA from {real_score_lora_path}")

        fake_score_ckpt = getattr(args, "fake_score_ckpt", None)
        if fake_score_ckpt:
            checkpoint = torch.load(
                fake_score_ckpt,
                map_location="cpu",
                weights_only=True,
            )
            if "generator" in checkpoint:
                checkpoint = checkpoint["generator"]
            elif "model" in checkpoint:
                checkpoint = checkpoint["model"]
            self.fake_score.load_state_dict(checkpoint, strict=True)
            self.fake_score.model.requires_grad_(True)
            print(f"Loaded fake-score checkpoint from {fake_score_ckpt}")

    def flatten_independent_frames(self, latent: torch.Tensor) -> torch.Tensor:
        """Turn [B,F,C,H,W] into independent K-frame score batch elements."""
        if latent.ndim != 5:
            raise ValueError(f"Expected [B, F, C, H, W], got {tuple(latent.shape)}")
        batch_size, num_frames = latent.shape[:2]
        if num_frames % self.score_chunk_size != 0:
            raise ValueError(
                f"Cannot split {num_frames} frames into chunks of "
                f"{self.score_chunk_size}"
            )
        num_chunks = num_frames // self.score_chunk_size
        return latent.reshape(
            batch_size * num_chunks,
            self.score_chunk_size,
            *latent.shape[2:],
        )

    @staticmethod
    def repeat_conditioning(conditional_dict: dict, num_frames: int, batch_size: int) -> dict:
        repeated = {}
        for key, value in conditional_dict.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                repeated[key] = value.repeat_interleave(num_frames, dim=0)
            else:
                repeated[key] = value
        return repeated

    @staticmethod
    def slice_conditioning(conditional_dict: dict, start: int, end: int, total: int) -> dict:
        sliced = {}
        for key, value in conditional_dict.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == total:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, None, int, int]:
        if initial_latent is None:
            raise ValueError("GT-first independent DMD requires an initial GT latent")
        if (
            initial_latent.ndim != 5
            or initial_latent.shape[1] != self.conditioning_frames
        ):
            raise ValueError(
                f"Expected initial latent with {self.conditioning_frames} frames, "
                f"got {tuple(initial_latent.shape)}"
            )

        batch_size, total_frames, channels, height, width = image_or_video_shape
        if total_frames != self.num_training_frames:
            raise ValueError(
                f"Expected {self.num_training_frames} total frames, got {total_frames}"
            )
        if initial_latent.shape[0] != batch_size:
            raise ValueError("Initial latent batch size does not match image_or_video_shape")

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
        output, denoised_timestep_from, denoised_timestep_to = (
            self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                initial_latent=initial_latent,
                **conditional_dict,
            )
        )
        if output.shape[1] != total_frames:
            raise RuntimeError(
                f"GT-first rollout returned {output.shape[1]} frames, expected {total_frames}"
            )

        # The conditioning prefix is data, not generated; exclude it from score loss.
        return (
            output[:, self.conditioning_frames:].to(self.dtype),
            None,
            denoised_timestep_from,
            denoised_timestep_to,
        )

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: torch.Tensor = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ):
        if gradient_mask is not None:
            raise ValueError("GT-first independent DMD does not use a gradient mask")
        batch_size, num_frames = image_or_video.shape[:2]
        independent_frames = self.flatten_independent_frames(image_or_video)
        num_score_chunks = num_frames // self.score_chunk_size
        independent_cond = self.repeat_conditioning(
            conditional_dict, num_score_chunks, batch_size
        )
        independent_uncond = self.repeat_conditioning(
            unconditional_dict, num_score_chunks, batch_size
        )
        total_score_chunks = independent_frames.shape[0]
        weighted_losses = []
        weighted_gradient_norms = []
        timesteps = []
        for start in range(
            0, total_score_chunks, self.independent_frame_chunk_size
        ):
            end = min(
                total_score_chunks, start + self.independent_frame_chunk_size
            )
            weight = (end - start) / total_score_chunks
            chunk_loss, chunk_log = super().compute_distribution_matching_loss(
                image_or_video=independent_frames[start:end],
                conditional_dict=self.slice_conditioning(
                    independent_cond, start, end, total_score_chunks
                ),
                unconditional_dict=self.slice_conditioning(
                    independent_uncond, start, end, total_score_chunks
                ),
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
            )
            weighted_losses.append(chunk_loss * weight)
            weighted_gradient_norms.append(chunk_log["dmdtrain_gradient_norm"] * weight)
            timesteps.append(chunk_log["timestep"])

        return torch.stack(weighted_losses).sum(), {
            "dmdtrain_gradient_norm": torch.stack(weighted_gradient_norms).sum(),
            "timestep": torch.cat(timesteps, dim=0),
            "dmd_frames": torch.tensor(
                batch_size * num_frames, device=independent_frames.device
            ),
            "dmd_score_chunks": torch.tensor(
                total_score_chunks, device=independent_frames.device
            ),
            "dmd_chunk_size": torch.tensor(
                self.score_chunk_size, device=independent_frames.device
            ),
        }

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
            generated_video, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )

        return self._critic_loss_from_generated(
            generated_video=generated_video,
            conditional_dict=conditional_dict,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )

    def _critic_loss_from_generated(
        self,
        generated_video: torch.Tensor,
        conditional_dict: dict,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        fake_score=None,
        critic_scheduler=None,
        score_chunk_size=None,
    ):
        """Fit the fake score to already-generated frames (frame 0 is absent)."""

        fake_score = self.fake_score if fake_score is None else fake_score
        critic_scheduler = (
            self.critic_scheduler
            if critic_scheduler is None
            else critic_scheduler
        )
        score_chunk_size = (
            self.score_chunk_size
            if score_chunk_size is None
            else int(score_chunk_size)
        )
        if score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive")

        batch_size, num_frames = generated_video.shape[:2]
        if num_frames % score_chunk_size != 0:
            raise ValueError(
                f"Cannot split {num_frames} frames into chunks of "
                f"{score_chunk_size}"
            )
        num_score_chunks = num_frames // score_chunk_size
        generated_frames = generated_video.reshape(
            batch_size * num_score_chunks,
            score_chunk_size,
            *generated_video.shape[2:],
        )
        independent_cond = self.repeat_conditioning(
            conditional_dict, num_score_chunks, batch_size
        )
        independent_shape = generated_frames.shape[:2]

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
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            generated_frames.shape[0],
            score_chunk_size,
            score_chunk_size,
            uniform_timestep=True,
        )
        if self.timestep_shift > 1:
            critic_timestep = (
                self.timestep_shift
                * (critic_timestep / 1000)
                / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000))
                * 1000
            )
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_frames)
        noisy_generated_frames = critic_scheduler.add_noise(
            generated_frames.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, independent_shape)
        total_score_chunks = generated_frames.shape[0]
        weighted_losses = []
        for start in range(
            0, total_score_chunks, self.independent_frame_chunk_size
        ):
            end = min(
                total_score_chunks, start + self.independent_frame_chunk_size
            )
            chunk_noisy = noisy_generated_frames[start:end]
            chunk_timestep = critic_timestep[start:end]
            _, chunk_pred = fake_score(
                noisy_image_or_video=chunk_noisy,
                conditional_dict=self.slice_conditioning(
                    independent_cond, start, end, total_score_chunks
                ),
                timestep=chunk_timestep,
            )

            if self.args.denoising_loss_type == "flow":
                flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                    scheduler=critic_scheduler,
                    x0_pred=chunk_pred.flatten(0, 1),
                    xt=chunk_noisy.flatten(0, 1),
                    timestep=chunk_timestep.flatten(0, 1),
                )
                pred_fake_noise = None
            else:
                flow_pred = None
                pred_fake_noise = critic_scheduler.convert_x0_to_noise(
                    x0=chunk_pred.flatten(0, 1),
                    xt=chunk_noisy.flatten(0, 1),
                    timestep=chunk_timestep.flatten(0, 1),
                )

            chunk_loss = self.denoising_loss_func(
                x=generated_frames[start:end].flatten(0, 1),
                x_pred=chunk_pred.flatten(0, 1),
                noise=critic_noise[start:end].flatten(0, 1),
                noise_pred=pred_fake_noise,
                alphas_cumprod=critic_scheduler.alphas_cumprod,
                timestep=chunk_timestep.flatten(0, 1),
                flow_pred=flow_pred,
            )
            weighted_losses.append(
                chunk_loss * ((end - start) / total_score_chunks)
            )

        denoising_loss = torch.stack(weighted_losses).sum()
        return denoising_loss, {
            "critic_timestep": critic_timestep.detach(),
            "critic_frames": torch.tensor(num_frames, device=generated_frames.device),
            "critic_score_chunks": torch.tensor(
                total_score_chunks, device=generated_frames.device
            ),
            "critic_chunk_size": torch.tensor(
                score_chunk_size, device=generated_frames.device
            ),
        }
