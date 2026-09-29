#!/usr/bin/env python3
"""Export a new, standalone Hugging Face folder without modifying source files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

ROOT = Path(__file__).resolve().parents[1]


def canonicalize(state):
    result = {}
    for key, tensor in state.items():
        for prefix in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
            key = key.replace(prefix, "")
        if key in result:
            raise ValueError(f"Duplicate canonical key: {key}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor state entry: {key}")
        result[key] = tensor
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(sources, output, model_card):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing release: {output}")
    for path in (*sources.values(), model_card):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    manifest = {"format_version": 1, "chunk_size": 1, "files": []}

    def record(relative, **metadata):
        path = output / relative
        manifest["files"].append({
            "path": relative, "size_bytes": path.stat().st_size,
            "sha256": sha256(path), **metadata,
        })
        print(f"Verified {relative}", flush=True)

    for stage, step in (("stage1", 900), ("stage2", 800)):
        print(f"Reading {stage}, step {step} (memory mapped)", flush=True)
        checkpoint = torch.load(sources[stage], map_location="cpu", weights_only=True, mmap=True)
        key, filename = "generator", "model.pt"
        if key not in checkpoint:
            raise KeyError(f"{stage}: missing {key}")
        state = canonicalize(checkpoint[key])
        relative = f"{stage}/{filename}"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generator": state}, path)
        # Compare every exported tensor to the source without changing dtype.
        restored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["generator"]
        if state.keys() != restored.keys():
            raise ValueError("Exported keys differ from source")
        for name, tensor in state.items():
            if tensor.dtype != restored[name].dtype or not torch.equal(tensor, restored[name]):
                raise ValueError(f"Tensor mismatch: {stage}/{key}/{name}")
        record(relative, stage=stage, step=step, source_key=key,
               tensor_count=len(state), parameters=sum(v.numel() for v in state.values()))
        del state, restored
        del checkpoint
    for key, relative in (
        ("fake", "training/fake_score.pt"),
        ("lora", "training/teacher_lora/adapter_model.bin"),
    ):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "fake":
            # The research checkpoint also contains run metadata; publish weights only.
            checkpoint = torch.load(sources[key], map_location="cpu", weights_only=True, mmap=True)
            state = canonicalize(checkpoint.get("generator", checkpoint.get("model", checkpoint)))
            torch.save({"generator": state}, path)
            restored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["generator"]
            if state.keys() != restored.keys():
                raise ValueError("Fake-score export keys differ")
            for name, tensor in state.items():
                if tensor.dtype != restored[name].dtype or not torch.equal(tensor, restored[name]):
                    raise ValueError(f"Fake-score tensor mismatch: {name}")
            record(relative, role=key, metadata_removed=True)
            del checkpoint, state, restored
        else:
            shutil.copyfile(sources[key], path)
            record(relative, role=key)
            if manifest["files"][-1]["sha256"] != sha256(sources[key]):
                raise ValueError(f"Copy mismatch: {relative}")
    for filename in ("stage1.yaml", "stage2.yaml", "inference.yaml", "default_config.yaml"):
        path = output / "configs" / filename
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "configs" / filename, path)
        record(f"configs/{filename}")
    shutil.copyfile(model_card, output / "README.md")
    shutil.copyfile(ROOT / "configs/teacher_adapter.json", output / "training/teacher_lora/adapter_metadata.json")
    shutil.copyfile(ROOT / "LICENSE", output / "CODE_LICENSE")
    for relative in ("README.md", "CODE_LICENSE", "training/teacher_lora/adapter_metadata.json"):
        record(relative)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Release ready: {output}; {len(manifest['files'])} verified files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True, help="Full stage-1 step-900 checkpoint")
    parser.add_argument("--stage2", type=Path, required=True, help="Full stage-2 step-800 checkpoint")
    parser.add_argument("--fake-score", type=Path, required=True)
    parser.add_argument("--teacher-lora", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True, help="Independently maintained Hugging Face README")
    parser.add_argument("--output", type=Path, default=ROOT / "release/huggingface")
    args = parser.parse_args()
    export({
        "stage1": args.stage1.resolve(), "stage2": args.stage2.resolve(),
        "fake": args.fake_score.resolve(),
        "lora": args.teacher_lora.resolve(),
    }, args.output.resolve(), args.model_card.resolve())
