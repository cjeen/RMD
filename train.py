"""Entry point for the two RMD training stages."""
import argparse
import os
from pathlib import Path

from utils.config import load_config, validate_training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--logdir", default="logs/train")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable optional W&B logging")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config_path, args.set)
    validate_training_config(config)
    if args.validate_only:
        print("Configuration and input paths validated (no accelerator execution).")
        return
    from trainer import ScoreDistillationTrainer
    import torch.distributed as dist
    if args.wandb:
        import wandb
    from omegaconf import OmegaConf

    config.config_name = Path(args.config_path).stem
    config.logdir = args.logdir
    config.no_save = args.no_save
    config.no_visualize = True
    config.disable_wandb = not args.wandb
    config.wandb_save_dir = args.logdir
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    try:
        trainer = ScoreDistillationTrainer(config)
        if int(os.environ.get("RANK", "0")) == 0:
            # Includes the actual random seed selected when seed=0.
            saved_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            saved_config.wandb_key = None
            OmegaConf.save(saved_config, Path(args.logdir) / "resolved_config.yaml")
        trainer.train()
    finally:
        if args.wandb:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
