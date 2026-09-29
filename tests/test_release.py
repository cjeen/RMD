import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils.dataset import PrecomputedWanJsonlDataset
from utils.training_schedule import training_phases
from test_rmd_scoring import load_method, REPOSITORY


class ReleaseTests(unittest.TestCase):
    def test_first_frame_mask_matches_zero_based_boundary(self):
        method = load_method(
            REPOSITORY / "model/frame_dmd_replay.py",
            "FrameDMDReplay", "_sample_first_frame_visibility",
        )
        obj = SimpleNamespace(num_frame_per_block=1, score_chunk_size=1,
                              conditioning_frames=1, replay_hide_first_frame_after=10,
                              replay_hide_first_frame_prob=1.0)
        visible, _ = method(obj, 2, 20, torch.device("cpu"))
        self.assertEqual(tuple(visible.shape), (2, 21))
        self.assertTrue(visible[:, :11].all())
        self.assertFalse(visible[:, 11:].any())

    def test_asymmetric_exit_keeps_continuations_at_last_step(self):
        method = load_method(
            REPOSITORY / "pipeline/self_forcing_training.py",
            "SelfForcingTrainingPipeline", "generate_and_sync_list",
        )
        method.__globals__["dist"] = torch.distributed
        obj = SimpleNamespace(fixed_first_block_exit_steps=None, exit_last_n_steps=None,
                              last_step_only=True, first_denoising_step_only=False,
                              random_first_block_exit=True, first_block_first_denoising_step=False)
        first_exits = set()
        for seed in range(20):
            torch.manual_seed(seed)
            exits = method(obj, 21, 4, torch.device("cpu"))
            self.assertEqual(exits[1:], [3] * 20)
            first_exits.add(exits[0])
        self.assertEqual(first_exits, {0, 1, 2, 3})

    def test_reference_update_counts_and_order(self):
        first = [training_phases("rmd", step) for step in range(900)]
        self.assertEqual(first[:6], [(False, True)] * 5 + [(True, False)])
        self.assertEqual(sum(g for g, c in first), 150)
        self.assertEqual(sum(c for g, c in first), 750)
        second = [training_phases("video_dmd", step) for step in range(800)]
        self.assertEqual(second[:5], [(True, True)] + [(False, True)] * 4)
        self.assertEqual(sum(g for g, c in second), 160)
        self.assertEqual(sum(c for g, c in second), 800)

    def test_embeddings_need_no_video_and_resolve_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = np.zeros((512, 4096), dtype=np.float16)
            np.save(root / "embed.npy", expected)
            (root / "train.jsonl").write_text(json.dumps({
                "prompt": "A fox", "text_embedding_path": "embed.npy",
                "latent_path": "/intentionally/missing/video.npy",
            }) + "\n")
            record = PrecomputedWanJsonlDataset(root / "train.jsonl")[0]
            self.assertEqual(record["prompts"], "A fox")
            self.assertEqual(record["prompt_embeds"].dtype, torch.float32)
            self.assertEqual(tuple(record["prompt_embeds"].shape), (512, 4096))
            self.assertNotIn("clean_latent", record)


if __name__ == "__main__":
    unittest.main()
