"""Load public configs independently of the caller's working directory."""
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=()):
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/default_config.yaml"),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(list(overrides)),
    )
    OmegaConf.resolve(config)
    return config


def validate_training_config(config):
    if config.distribution_loss not in ("rmd", "video_dmd"):
        raise ValueError("Only rmd and video_dmd are released")
    if config.num_frame_per_block != 1 or config.num_training_frames != 21:
        raise ValueError("Expected num_frame_per_block=1 and num_training_frames=21")
    if config.i2v or list(config.image_or_video_shape)[1:] != [21, 16, 60, 104]:
        raise ValueError("The release recipe is text-to-video at 832x480")
    if config.max_steps <= 0 or config.batch_size <= 0:
        raise ValueError("max_steps and batch_size must be positive")
    expected_ratio = 6 if config.distribution_loss == "rmd" else 5
    if config.dfake_gen_update_ratio != expected_ratio:
        raise ValueError(f"Expected dfake_gen_update_ratio={expected_ratio}")
    expected_phase = "five_critic_one_generator" if config.distribution_loss == "rmd" else ""
    if config.get("phase_schedule", "") != expected_phase:
        raise ValueError(f"Expected phase_schedule={expected_phase!r}")
    paths = [config.generator_ckpt]
    if config.data_format == "precomputed_wan_jsonl":
        paths.append(config.data_meta_path)
    elif config.data_format == "text":
        paths.append(config.data_path)
    else:
        raise ValueError("data_format must be text or precomputed_wan_jsonl")
    if config.distribution_loss == "rmd":
        paths.extend([config.real_score_lora_path, config.fake_score_ckpt])
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
