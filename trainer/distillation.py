"""Two-stage RMD score distillation; update schedules match the release runs."""
import gc
import os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from model import TextFrameDMDReplayFirstFrameDMD, VideoDMDReplay
from utils.dataset import PrecomputedWanJsonlDataset, TextDataset, cycle
from utils.distributed import fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.device import empty_cache, get_default_device
from utils.misc import set_seed
from utils.training_schedule import training_phases


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = get_default_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            import wandb
            self.wandb = wandb
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        model_cls = {
            "rmd": TextFrameDMDReplayFirstFrameDMD,
            "video_dmd": VideoDMDReplay,
        }[config.distribution_loss]
        self.model = model_cls(config, device=self.device)

        # This is deliberately NPU-only and opt-in. GPU jobs retain their
        # existing placement behavior even if the flag is accidentally present.
        self.npu_real_score_cpu_offload = (
            self.device.type == "npu"
            and bool(getattr(config, "npu_real_score_cpu_offload", False))
        )
        if self.is_main_process and self.npu_real_score_cpu_offload:
            print(
                "NPU CPU offload enabled for frozen real-score models.",
                flush=True,
            )

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=self.npu_real_score_cpu_offload,
        )
        if self.npu_real_score_cpu_offload:
            empty_cache()

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.has_first_frame_critic = hasattr(
            self.model, "first_frame_fake_score"
        )
        if self.has_first_frame_critic:
            self.model.first_frame_real_score = fsdp_wrap(
                self.model.first_frame_real_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(
                    config,
                    "first_frame_real_score_fsdp_wrap_strategy",
                    config.real_score_fsdp_wrap_strategy,
                ),
                cpu_offload=self.npu_real_score_cpu_offload,
            )
            if self.npu_real_score_cpu_offload:
                empty_cache()
            self.model.first_frame_fake_score = fsdp_wrap(
                self.model.first_frame_fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=getattr(
                    config,
                    "first_frame_fake_score_fsdp_wrap_strategy",
                    config.fake_score_fsdp_wrap_strategy,
                ),
            )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=getattr(
                config, "weight_decay_generator", config.weight_decay
            )
        )
        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=getattr(
                config, "weight_decay_critic", config.weight_decay
            )
        )
        self.first_frame_critic_optimizer = None
        if self.has_first_frame_critic:
            self.first_frame_critic_optimizer = torch.optim.AdamW(
                [
                    param
                    for param in self.model.first_frame_fake_score.parameters()
                    if param.requires_grad
                ],
                lr=getattr(
                    config,
                    "lr_first_frame_critic",
                    config.lr_critic if hasattr(config, "lr_critic") else config.lr,
                ),
                betas=(config.beta1_critic, config.beta2_critic),
                weight_decay=getattr(
                    config, "weight_decay_critic", config.weight_decay
                ),
            )
        # Step 3: Initialize the dataloader
        if getattr(self.config, "data_format", None) == "precomputed_wan_jsonl":
            dataset = PrecomputedWanJsonlDataset(config.data_meta_path)
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        if len(sampler) < config.batch_size:
            raise ValueError("Dataset must provide at least one batch per rank")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=getattr(config, "num_workers", 8))

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # Initialize weights only; optimizer/RNG states are not resumed.
        initialization_checkpoint = None
        initialization_checkpoint_path = None
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            initialization_checkpoint = torch.load(
                config.generator_ckpt, map_location="cpu"
            )
            initialization_checkpoint_path = config.generator_ckpt
            state_dict = initialization_checkpoint
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        if getattr(config, "critic_ckpt", False):
            print(f"Loading pretrained critic from {config.critic_ckpt}")
            if config.critic_ckpt == initialization_checkpoint_path:
                critic_checkpoint = initialization_checkpoint
            else:
                critic_checkpoint = torch.load(
                    config.critic_ckpt, map_location="cpu"
                )
            state_dict = critic_checkpoint
            if "critic" in state_dict:
                state_dict = state_dict["critic"]
            elif "fake_score" in state_dict:
                state_dict = state_dict["fake_score"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.fake_score.load_state_dict(state_dict, strict=True)
            if critic_checkpoint is not initialization_checkpoint:
                del critic_checkpoint

        if self.has_first_frame_critic and getattr(
            config, "first_frame_critic_ckpt", False
        ):
            print(
                "Loading first-frame critic from "
                f"{config.first_frame_critic_ckpt}"
            )
            first_frame_critic_checkpoint = torch.load(
                config.first_frame_critic_ckpt, map_location="cpu"
            )
            state_dict = first_frame_critic_checkpoint
            if "first_frame_critic" in state_dict:
                state_dict = state_dict["first_frame_critic"]
            elif "critic" in state_dict:
                state_dict = state_dict["critic"]
            elif "fake_score" in state_dict:
                state_dict = state_dict["fake_score"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.first_frame_fake_score.load_state_dict(
                state_dict, strict=True
            )
            del first_frame_critic_checkpoint

        del initialization_checkpoint

        ##############################################################################################################

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)
        first_frame_critic_state_dict = None
        if self.has_first_frame_critic:
            first_frame_critic_state_dict = fsdp_state_dict(
                self.model.first_frame_fake_score
            )

        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict,
        }
        if first_frame_critic_state_dict is not None:
            state_dict["first_frame_critic"] = first_frame_critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        use_precomputed_features = (
            getattr(self.config, "data_format", None) == "precomputed_wan_jsonl"
        )
        if use_precomputed_features:
            clean_latent = None
            initial_latent_frames = int(
                getattr(self.config, "initial_latent_frames", 1)
            )
            if initial_latent_frames <= 0:
                raise ValueError("initial_latent_frames must be positive")
            image_latent = (
                batch["clean_latent"][:, :initial_latent_frames].to(
                    device=self.device, dtype=self.dtype
                )
                if self.config.i2v
                else None
            )
        elif self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            if use_precomputed_features:
                conditional_dict = {
                    "prompt_embeds": batch["prompt_embeds"].to(
                        device=self.device, dtype=self.dtype
                    )
                }
            else:
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})
            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        first_frame_critic_grad_norm = None
        if self.has_first_frame_critic:
            first_frame_critic_grad_norm = (
                self.model.first_frame_fake_score.clip_grad_norm_(
                    self.max_grad_norm_critic
                )
            )
            combined_critic_grad_norm = torch.sqrt(
                critic_grad_norm.float().square()
                + first_frame_critic_grad_norm.float().square()
            )
        else:
            combined_critic_grad_norm = critic_grad_norm

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": combined_critic_grad_norm,
                                "replay_critic_grad_norm": critic_grad_norm})
        if first_frame_critic_grad_norm is not None:
            critic_log_dict["first_frame_critic_grad_norm"] = (
                first_frame_critic_grad_norm
            )
        return critic_log_dict


    def train(self):
        repeated_batch = None
        for step in range(self.config.max_steps):
            train_generator, train_critic = training_phases(
                self.config.distribution_loss, step
            )
            # Stage 1 reuses one prompt batch over five critic and one G update.
            if self.config.distribution_loss == "rmd" and step % 6 == 0:
                repeated_batch = next(self.dataloader)
            metrics = {}
            if train_generator:
                self.generator_optimizer.zero_grad(set_to_none=True)
                batch = repeated_batch if repeated_batch is not None else next(self.dataloader)
                metrics.update(self.fwdbwd_one_step(batch, True))
                self.generator_optimizer.step()
            if train_critic:
                optimizers = [self.critic_optimizer]
                if self.first_frame_critic_optimizer is not None:
                    optimizers.append(self.first_frame_critic_optimizer)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                batch = repeated_batch if repeated_batch is not None else next(self.dataloader)
                metrics.update(self.fwdbwd_one_step(batch, False))
                for optimizer in optimizers:
                    optimizer.step()
            self.step = step + 1
            if self.is_main_process:
                scalars = {
                    key: value.detach().float().mean().item()
                    for key, value in metrics.items() if torch.is_tensor(value)
                }
                print(f"step={self.step} generator={train_generator} critic={train_critic} {scalars}", flush=True)
                if not self.disable_wandb:
                    self.wandb.log(scalars, step=self.step)
            if not self.config.no_save and (
                    self.step % self.config.log_iters == 0
                    or self.step == self.config.max_steps):
                self.save()
            if self.step % self.config.gc_interval == 0:
                gc.collect()
                empty_cache()
        dist.barrier()
