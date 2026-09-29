import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Tuple
import unittest
from unittest.mock import patch

import torch


REPOSITORY = Path(__file__).resolve().parents[1]


def load_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            method = next(
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == method_name
            )
            break
    if method is None:
        raise AssertionError(f"Could not find {class_name}.{method_name}")
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "Tuple": Tuple}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class _ConstantScore:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self, noisy_image_or_video, conditional_dict, timestep):
        del conditional_dict, timestep
        self.calls += 1
        prediction = torch.full_like(noisy_image_or_video, self.value)
        return None, prediction


class _ConditionedScore:
    def __init__(self, conditional_value, unconditional_value):
        self.conditional_value = conditional_value
        self.unconditional_value = unconditional_value

    def __call__(self, noisy_image_or_video, conditional_dict, timestep):
        del timestep
        value = (
            self.conditional_value
            if conditional_dict["conditional"]
            else self.unconditional_value
        )
        return None, torch.full_like(noisy_image_or_video, value)


class TextFrameDMDReplayFirstFrameDMDTest(unittest.TestCase):
    def test_dmd_fp32_arithmetic_preserves_bf16_score_inputs(self):
        compute_grad = load_method(REPOSITORY / "model" / "dmd.py", "DMD", "_compute_kl_grad")
        clean = torch.tensor([0.125, 1.375], dtype=torch.bfloat16).reshape(1, 1, 1, 1, 2)
        fake = torch.full_like(clean, 0.773)
        real = torch.full_like(clean, 0.00332)
        seen_dtypes = []

        def score(prediction):
            def forward(noisy_image_or_video, **kwargs):
                seen_dtypes.append(noisy_image_or_video.dtype)
                return None, prediction
            return forward

        for enabled in (False, True):
            for normalization in (False, True):
                with self.subTest(enabled=enabled, normalization=normalization):
                    model = SimpleNamespace(
                        real_score=score(real), fake_score=score(fake),
                        real_guidance_scale=0.0, fake_guidance_scale=0.0,
                        skip_real_uncond_when_guidance_zero=True,
                        dmd_grad_fp32=enabled,
                    )
                    grad, _ = compute_grad(
                        model, clean, clean, torch.zeros(1, 1), {}, {},
                        normalization=normalization,
                    )
                    f, r, x = (v.float() if enabled else v for v in (fake, real, clean))
                    expected = f - r
                    if normalization:
                        expected = expected / (x - r).abs().mean(
                            dim=[1, 2, 3, 4], keepdim=True
                        )
                    self.assertEqual(grad.dtype, torch.float32 if enabled else torch.bfloat16)
                    torch.testing.assert_close(grad, expected, rtol=0, atol=0)
        self.assertTrue(all(dtype == torch.bfloat16 for dtype in seen_dtypes))
        # The test inputs must expose precision lost by subtracting in BF16.
        self.assertFalse(torch.equal(fake.float() - real.float(), (fake - real).float()))

    def test_mixed_first_frame_probability_boundaries(self):
        select_initial = load_method(
            REPOSITORY / "model" / "text_frame_dmd_replay.py",
            "TextFrameDMDReplay",
            "_select_rollout_initial_latent",
        )
        gt = torch.ones(1, 1, 1, 1, 1)

        generated_model = SimpleNamespace(
            gt_first_conditioning_probability=0.0
        )
        selected, use_gt = select_initial(generated_model, gt)
        self.assertIsNone(selected)
        self.assertFalse(use_gt)

        gt_model = SimpleNamespace(gt_first_conditioning_probability=1.0)
        selected, use_gt = select_initial(gt_model, gt)
        self.assertIs(selected, gt)
        self.assertTrue(use_gt)

        with self.assertRaisesRegex(ValueError, "initial latent"):
            select_initial(gt_model, None)

    def test_gt_rollout_is_normalized_to_full_21_frame_shape(self):
        rollout_method = load_method(
            REPOSITORY / "model" / "text_frame_dmd_replay.py",
            "TextFrameDMDReplay",
            "_run_rollout_with_trajectory",
        )
        generated = torch.full((1, 20, 1, 1, 1), 2.0)
        gt = torch.ones(1, 1, 1, 1, 1)

        class FakeFrameDMDReplay:
            @staticmethod
            def _run_rollout_with_trajectory(*args):
                return generated, {"source": "gt"}, 1000, 0

        rollout_method.__globals__["FrameDMDReplay"] = FakeFrameDMDReplay
        rollout, trajectory, timestep_from, timestep_to = rollout_method(
            SimpleNamespace(), [1, 21, 1, 1, 1], {}, gt
        )

        self.assertEqual(tuple(rollout.shape), (1, 21, 1, 1, 1))
        self.assertTrue(torch.equal(rollout[:, :1], gt))
        self.assertTrue(torch.equal(rollout[:, 1:], generated))
        self.assertEqual(trajectory, {"source": "gt"})
        self.assertEqual((timestep_from, timestep_to), (1000, 0))

    def test_half_probability_draw_is_broadcast_and_controls_the_branch(self):
        select_initial = load_method(
            REPOSITORY / "model" / "text_frame_dmd_replay.py",
            "TextFrameDMDReplay",
            "_select_rollout_initial_latent",
        )

        class FakeDistributed:
            broadcasts = 0

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_rank():
                return 0

            @classmethod
            def broadcast(cls, value, src):
                del value
                self.assertEqual(src, 0)
                cls.broadcasts += 1

        select_initial.__globals__["dist"] = FakeDistributed
        model = SimpleNamespace(
            gt_first_conditioning_probability=0.5,
            device=torch.device("cpu"),
        )
        gt = torch.ones(1, 1, 1, 1, 1)

        with patch.object(torch, "rand", return_value=torch.tensor(0.25)):
            selected, use_gt = select_initial(model, gt)
        self.assertIs(selected, gt)
        self.assertTrue(use_gt)

        with patch.object(torch, "rand", return_value=torch.tensor(0.75)):
            selected, use_gt = select_initial(model, gt)
        self.assertIsNone(selected)
        self.assertFalse(use_gt)
        self.assertEqual(FakeDistributed.broadcasts, 2)

    def test_gt_replay_preserves_context_and_all_20_continuation_frames(self):
        replay_method = load_method(
            REPOSITORY / "model" / "text_frame_dmd_replay.py",
            "TextFrameDMDReplay",
            "_teacher_forcing_replay",
        )
        gt = torch.ones(1, 1, 1, 1, 1)
        rollout = torch.cat(
            [gt, torch.full((1, 20, 1, 1, 1), 2.0)], dim=1
        )
        replay_generated = torch.full((1, 20, 1, 1, 1), 3.0)
        captured = {}

        class FakeFrameDMDReplay:
            @staticmethod
            def _teacher_forcing_replay(
                model, initial_latent, rollout_generated, trajectory, conditional
            ):
                del model, trajectory, conditional
                captured["initial"] = initial_latent
                captured["generated"] = rollout_generated
                return replay_generated, {"source": "gt"}

        replay_method.__globals__["FrameDMDReplay"] = FakeFrameDMDReplay
        replay, diagnostics = replay_method(
            SimpleNamespace(conditioning_frames=1),
            rollout,
            {},
            {},
            initial_latent=gt,
        )

        self.assertEqual(tuple(replay.shape), (1, 21, 1, 1, 1))
        self.assertTrue(torch.equal(replay[:, :1], gt))
        self.assertTrue(torch.equal(replay[:, 1:], replay_generated))
        self.assertTrue(torch.equal(captured["generated"], rollout[:, 1:]))
        self.assertEqual(diagnostics, {"source": "gt"})

    def test_combine_frame_losses_is_exact_21_frame_average(self):
        combine_frame_losses = load_method(
            REPOSITORY
            / "model"
            / "text_frame_dmd_replay_first_frame_dmd.py",
            "TextFrameDMDReplayFirstFrameDMD",
            "combine_frame_losses",
        )
        first = torch.tensor(21.0, requires_grad=True)
        replay = torch.tensor(2.0, requires_grad=True)

        loss = combine_frame_losses(first, replay, replay_frames=20)

        self.assertAlmostEqual(loss.item(), 61.0 / 21.0, places=6)
        loss.backward()
        self.assertAlmostEqual(first.grad.item(), 1.0 / 21.0, places=6)
        self.assertAlmostEqual(replay.grad.item(), 20.0 / 21.0, places=6)

    def test_combine_frame_losses_supports_normalized_first_frame_weight(self):
        combine_frame_losses = load_method(
            REPOSITORY
            / "model"
            / "text_frame_dmd_replay_first_frame_dmd.py",
            "TextFrameDMDReplayFirstFrameDMD",
            "combine_frame_losses",
        )
        first = torch.tensor(21.0, requires_grad=True)
        replay = torch.tensor(2.0, requires_grad=True)

        loss = combine_frame_losses(
            first,
            replay,
            replay_frames=20,
            first_frame_weight=3.0,
        )

        self.assertAlmostEqual(loss.item(), 103.0 / 23.0, places=6)
        loss.backward()
        self.assertAlmostEqual(first.grad.item(), 3.0 / 23.0, places=6)
        self.assertAlmostEqual(replay.grad.item(), 20.0 / 23.0, places=6)

    def test_generator_routes_frame_zero_and_later_frames_separately(self):
        model_path = (
            REPOSITORY
            / "model"
            / "text_frame_dmd_replay_first_frame_dmd.py"
        )
        generator_loss = load_method(
            model_path,
            "TextFrameDMDReplayFirstFrameDMD",
            "generator_loss",
        )
        combine_frame_losses = load_method(
            model_path,
            "TextFrameDMDReplayFirstFrameDMD",
            "combine_frame_losses",
        )
        replay_prediction = torch.ones(
            1, 21, 1, 1, 1, requires_grad=True
        )
        rollout = torch.zeros_like(replay_prediction)
        routed_shapes = {}

        def standard_log(frames):
            scalar = torch.tensor(1.0)
            return {
                "dmdtrain_gradient_norm": scalar,
                "dmd_score_delta_abs_mean": scalar,
                "dmd_score_delta_rms": scalar,
                "dmd_normalizer_mean": scalar,
                "dmd_normalizer_min": scalar,
                "dmd_normalizer_max": scalar,
                "timestep": torch.zeros(frames, 1),
            }

        def compute_replay(**kwargs):
            routed_shapes["replay"] = kwargs["replay_pred"].shape[1]
            return kwargs["replay_pred"].mean(), standard_log(20)

        def compute_first(**kwargs):
            routed_shapes["first"] = kwargs["replay_pred"].shape[1]
            return kwargs["replay_pred"].mean(), standard_log(1)

        model = SimpleNamespace(
            _run_rollout_with_trajectory=lambda *args: (
                rollout,
                {},
                1000,
                0,
            ),
            _teacher_forcing_replay=lambda *args: (replay_prediction, {}),
            _compute_replay_dmd_loss=compute_replay,
            _compute_first_frame_dmd_loss=compute_first,
            combine_frame_losses=combine_frame_losses,
            conditioning_frames=1,
            num_training_frames=21,
            score_chunk_size=1,
        )

        loss, _ = generator_loss(
            model,
            image_or_video_shape=[1, 21, 1, 1, 1],
            conditional_dict={},
            unconditional_dict={},
            clean_latent=None,
        )
        loss.backward()

        self.assertEqual(routed_shapes, {"replay": 20, "first": 1})
        self.assertTrue(
            torch.allclose(
                replay_prediction.grad,
                torch.full_like(replay_prediction, 1.0 / 21.0),
            )
        )



if __name__ == "__main__":
    unittest.main()
