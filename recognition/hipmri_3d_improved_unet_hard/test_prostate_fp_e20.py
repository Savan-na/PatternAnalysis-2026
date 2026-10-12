"""CPU numerical and configuration tests for the V4 E20 paired experiment."""

import math
import unittest

import torch

from . import experiment_prostate_fp_e20 as v4


class TestV4E20(unittest.TestCase):

    def test_frozen_settings(self):
        cfg = v4.settings()

        self.assertEqual(
            cfg["experiment_id"],
            "v4_prostate_fp_tversky_paired_e20_v1"
        )
        self.assertEqual(cfg["parent_epoch"], 12)
        self.assertEqual(cfg["end_epoch"], 20)
        self.assertEqual(cfg["steps_per_epoch"], 143)
        self.assertEqual(cfg["validation_epochs"], [14, 16, 18, 20])
        self.assertEqual(cfg["primary_comparison_epoch"], 20)
        self.assertEqual(cfg["max_epochs_per_job"], 1)
        self.assertEqual(cfg["alpha_control"], 0.5)
        self.assertEqual(cfg["alpha_treatment"], 0.7)

        hashes = v4.local_hashes()
        self.assertEqual(set(hashes), set(v4.LOCAL_FILES))
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_head_loss_and_gradients(self):
        torch.manual_seed(3710)

        for scenario in ("present", "absent"):
            with self.subTest(scenario=scenario):
                logits = torch.randn(
                    1, 6, 8, 8, 8, requires_grad=True
                )
                target = torch.randint(0, 6, (1, 8, 8, 8))

                if scenario == "present":
                    target[0, 0, 0, 0] = 5
                else:
                    target[target == 5] = 0

                reference = v4.v3.v2.head_loss(logits, target)
                control = v4.head_loss(logits, target, 0.5)
                treatment = v4.head_loss(logits, target, 0.7)

                for expected, actual in zip(reference, control[:4]):
                    torch.testing.assert_close(
                        expected, actual, atol=0, rtol=0
                    )

                p = logits.softmax(dim=1)[:, 5]
                y = (target == 5).to(logits.dtype)

                tp = (p * y).sum()
                fp = (p * (1 - y)).sum()
                fn = ((1 - p) * y).sum()
                eps = 1e-5

                dice = (2 * tp + eps) / (
                    p.sum() + y.sum() + eps
                )

                tversky_half = (tp + eps / 2) / (
                    tp + 0.5 * fp + 0.5 * fn + eps / 2
                )

                torch.testing.assert_close(
                    dice, tversky_half, atol=1e-6, rtol=1e-6
                )

                tversky_07 = (tp + eps / 2) / (
                    tp + 0.7 * fp + 0.3 * fn + eps / 2
                )

                expected_overlap = (
                    reference[2] + (dice - tversky_07) / 5
                )

                torch.testing.assert_close(
                    treatment[2],
                    expected_overlap,
                    atol=1e-6,
                    rtol=1e-6
                )

                torch.testing.assert_close(
                    treatment[1],
                    reference[1],
                    atol=0,
                    rtol=0
                )

                gradient = torch.autograd.grad(
                    treatment[0], logits
                )[0]

                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(
                    float(gradient.abs().sum()), 0
                )

    def test_deep_supervision_control_parity(self):
        torch.manual_seed(3710)

        main = torch.randn(
            1, 6, 16, 16, 16, requires_grad=True
        )
        aux_half = torch.randn(
            1, 6, 8, 8, 8, requires_grad=True
        )
        aux_quarter = torch.randn(
            1, 6, 4, 4, 4, requires_grad=True
        )
        target = torch.randint(0, 6, (1, 16, 16, 16))

        outputs = (main, [aux_half, aux_quarter])

        reference = v4.v3.v2.combined_loss(outputs, target)
        control = v4.loss(outputs, target, "control")

        for expected, actual in zip(reference, control[:4]):
            torch.testing.assert_close(
                expected, actual, atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
