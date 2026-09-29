"""Prompt datasets for text-only, on-policy distillation."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, prompt_path):
        self.prompts = Path(prompt_path).read_text(encoding="utf-8").splitlines()
        if not self.prompts or any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("Prompt file must contain nonempty lines")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompts": self.prompts[idx], "idx": idx}


class PrecomputedWanJsonlDataset(Dataset):
    """JSONL: prompt + text_embedding_path; paths relative to the JSONL parent.

    Original metadata may also include latent_path. T2V training does
    not use ground-truth latents, so they are deliberately not loaded.
    Embeddings must be padded UMT5 outputs shaped [512, 4096].
    """
    def __init__(self, meta_path):
        self.meta_path = Path(meta_path)
        with self.meta_path.open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Empty metadata: {meta_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        path = Path(record["text_embedding_path"])
        if not path.is_absolute():
            path = self.meta_path.parent / path
        embed = np.load(path, allow_pickle=False)
        if embed.shape != (512, 4096):
            raise ValueError(f"{path}: expected (512, 4096), got {embed.shape}")
        return {
            "prompts": record["prompt"],
            "prompt_embeds": torch.from_numpy(embed).float().contiguous(),
            "idx": idx,
        }


def cycle(dataloader):
    # Preserve the original run's sampler order on each repetition.
    while True:
        yield from dataloader
