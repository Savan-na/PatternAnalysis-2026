import unittest
from importlib.util import find_spec



@unittest.skipIf(find_spec("torch") is None, "PyTorch is not installed")
class NetworkTests(unittest.TestCase):
    def test_2d_logits_and_gradients(self):
        import torch
        from hipmri_rewrite.models import UNet2D
        from hipmri_rewrite.losses import ce_dice_loss

        model = UNet2D(base_channels=4)
        image = torch.randn(1, 1, 32, 32)
        target = torch.randint(0, 6, (1, 32, 32), dtype=torch.long)
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 6, 32, 32))
        ce_dice_loss(logits, target).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_3d_standard_and_improved_logits_and_gradients(self):
        import torch
        from hipmri_rewrite.models import ImprovedUNet3D, UNet3D
        from hipmri_rewrite.losses import ce_dice_loss

        image = torch.randn(1, 1, 32, 32, 32)
        target = torch.randint(0, 6, (1, 32, 32, 32), dtype=torch.long)
        for model in (UNet3D(base_channels=2), ImprovedUNet3D(base_channels=2, dropout=0)):
            with self.subTest(model=type(model).__name__):
                logits = model(image)
                self.assertEqual(tuple(logits.shape), (1, 6, 32, 32, 32))
                ce_dice_loss(logits, target).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_improved_supervision_is_additive(self):
        import torch
        from hipmri_rewrite.models import ImprovedUNet3D

        model = ImprovedUNet3D(base_channels=2, dropout=0).eval()
        with torch.no_grad():
            combined, pieces = model(torch.randn(1, 1, 32, 32, 32), return_components=True)
        torch.testing.assert_close(combined, sum(pieces))

    def test_2d_and_3d_inference_return_original_xyz_grid(self):
        import numpy as np
        import torch
        from pathlib import Path
        from hipmri_rewrite.data import Volume
        from hipmri_rewrite.inference import predict_volume
        from hipmri_rewrite.models import UNet2D, UNet3D
        from hipmri_rewrite.splits import Case

        case = Case("K019", 1, "train", Path("image"), Path("mask"))
        image = np.random.default_rng(3710).standard_normal((17, 18, 17)).astype(np.float32)
        volume = Volume(case, image, np.zeros(image.shape, dtype=np.uint8), np.eye(4), (1., 1., 1.))
        for mode, model in (("2d", UNet2D(base_channels=2)), ("3d", UNet3D(base_channels=2))):
            with self.subTest(mode=mode):
                predicted = predict_volume(model, volume, mode=mode, device=torch.device("cpu"), slice_batch=4)
                self.assertEqual(predicted.shape, image.shape)
                self.assertTrue(np.all(predicted < 6))


if __name__ == "__main__":
    unittest.main()
