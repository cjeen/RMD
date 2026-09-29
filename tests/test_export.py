import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportTests(unittest.TestCase):
    def test_roundtrip_preserves_precision_sources_and_detects_corruption(self):
        exporter = load_script("export_release")
        verifier = load_script("verify_release")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {}
            for name in ("stage1", "stage2"):
                path = root / f"{name}.pt"
                torch.save({
                    "generator": {"model.weight": torch.arange(4, dtype=torch.bfloat16)},
                    "critic": {"excluded": torch.ones(2)},
                }, path)
                sources[name] = path
            for name in ("fake", "lora"):
                path = root / f"{name}.pt"
                state = {"model.weight": torch.ones(2)}
                if name == "fake":
                    state = {"generator": state, "metadata": {"private_path": "/private/experiment"}}
                torch.save(state, path)
                sources[name] = path
            before = {name: exporter.sha256(path) for name, path in sources.items()}
            output = root / "release"
            model_card = root / "model_card.md"
            model_card.write_text("# Test model release\n")
            exporter.export(sources, output, model_card)
            self.assertEqual((output / "README.md").read_text(), model_card.read_text())
            self.assertEqual(before, {n: exporter.sha256(p) for n, p in sources.items()})
            self.assertEqual(verifier.verify(output), 11)
            self.assertFalse((output / "training/causal_ode.pt").exists())
            self.assertFalse((output / "images/teaser.png").exists())
            regular = torch.load(output / "stage2/model.pt", weights_only=True)
            self.assertFalse(list(output.rglob("*ema*")))
            self.assertEqual(set(regular), {"generator"})
            self.assertEqual(regular["generator"]["model.weight"].dtype, torch.bfloat16)
            fake = torch.load(output / "training/fake_score.pt", weights_only=True)
            self.assertEqual(set(fake), {"generator"})
            with self.assertRaises(FileExistsError):
                exporter.export(sources, output, model_card)
            (output / "stage2/model.pt").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                verifier.verify(output)


if __name__ == "__main__":
    unittest.main()
