"""RMD text-to-video inference with a fully sliding KV window."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/rmd/stage2/model.pt")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--prompts", default="prompts/example.txt")
    parser.add_argument("--precomputed", action="store_true", help="Prompts is an embedding JSONL")
    parser.add_argument("--output", default="outputs/rmd")
    parser.add_argument("--latent-frames", type=int, default=241,
                        help="Decoded frame count is 4*N-3; 241 gives ~60 seconds at 16 FPS")
    parser.add_argument("--kv-window", type=int, default=21)
    parser.add_argument("--decode-chunk", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    if min(args.latent_frames, args.kv_window, args.decode_chunk, args.samples) < 1:
        parser.error("Frame counts, window, decode chunk and samples must be positive")

    import imageio.v2 as imageio
    import torch
    from pipeline import CausalInferencePipeline
    from utils.config import load_config
    from utils.dataset import PrecomputedWanJsonlDataset, TextDataset
    from utils.device import get_default_device, set_current_device, empty_cache
    from utils.distributed import canonicalize_wrapped_module_state_dict
    from utils.misc import set_seed
    from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    set_current_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = get_default_device()
    if device.type == "cpu":
        raise RuntimeError("Video inference requires a CUDA GPU or Ascend NPU")
    config = load_config(args.config)
    if config.num_frame_per_block != 1:
        raise ValueError("Expected num_frame_per_block=1 for the pretrained model")
    config.model_kwargs.local_attn_size = args.kv_window
    with torch.no_grad():
        pipeline = CausalInferencePipeline(
            config, device=device, load_text_encoder=not args.precomputed
        )
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("generator", checkpoint.get("model", checkpoint))
        pipeline.generator.load_state_dict(canonicalize_wrapped_module_state_dict(state), strict=True)
        del checkpoint, state
        pipeline = pipeline.to(dtype=torch.bfloat16).eval()
        low_memory = get_cuda_free_memory_gb(device) < 40
        if pipeline.text_encoder is not None:
            if low_memory:
                DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
            else:
                pipeline.text_encoder.to(device)
        pipeline.generator.to(device)
        pipeline.vae.to(device)
        dataset = (PrecomputedWanJsonlDataset(args.prompts) if args.precomputed
                   else TextDataset(args.prompts))
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        # Striding avoids DistributedSampler padding and duplicate outputs.
        for index in range(rank, len(dataset), world):
            record = dataset[index]
            for sample in range(args.samples):
                set_seed(args.seed + index * args.samples + sample)
                noise = torch.randn(1, args.latent_frames, 16, 60, 104,
                                    device=device, dtype=torch.bfloat16)
                conditioning = (
                    {"prompt_embeds": record["prompt_embeds"].unsqueeze(0).to(device, torch.bfloat16)}
                    if args.precomputed else {"text_prompts": [record["prompts"]]}
                )
                video = pipeline.inference(
                    noise=noise, low_memory=low_memory,
                    vae_decode_chunk_size=args.decode_chunk, **conditioning
                )
                path = output / f"{index:05d}_{sample:02d}.mp4"
                # Encode frame by frame; avoid another full float video allocation.
                with imageio.get_writer(path, fps=16, codec="libx264", macro_block_size=1) as writer:
                    for frame in video[0]:
                        writer.append_data((frame.permute(1, 2, 0).clamp(0, 1) * 255)
                                           .to(torch.uint8).cpu().numpy())
                print(path, flush=True)
                del noise, video
                empty_cache()


if __name__ == "__main__":
    main()
