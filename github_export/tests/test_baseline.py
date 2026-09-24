"""Meaningful checks for model shape, gradients, overlap metrics, and real data."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/models"))

import torch
from train_unet import LumenDataset, overlap_scores, read_splits, seed_everything, segmentation_loss
from unet import SmallUNet


class BaselineTests(unittest.TestCase):
    def test_overlap_edge_cases(self):
        # Perfect, disjoint, both empty, and partially overlapping predictions.
        prediction = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]]).reshape(4, 1, 2, 2)
        truth = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]]).reshape(4, 1, 2, 2).float()
        dice, iou = overlap_scores(prediction.float() * 20 - 10, truth)
        torch.testing.assert_close(dice, torch.tensor([1.0, 0.0, 1.0, 0.5]))
        torch.testing.assert_close(iou, torch.tensor([1.0, 0.0, 1.0, 1 / 3]))

    def test_full_resolution_and_gradients(self):
        seed_everything(42)
        model = SmallUNet()
        logits = model(torch.rand(1, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (1, 1, 256, 256))
        loss = segmentation_loss(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_reproducible_initialization(self):
        seed_everything(42)
        first = SmallUNet().state_dict()
        seed_everything(42)
        second = SmallUNet().state_dict()
        self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))

    def test_existing_splits_and_array_contract(self):
        splits = read_splits()  # Also checks duplicate/source separation.
        self.assertEqual({s: len(r) for s, r in splits.items()}, {"train": 311, "validation": 99, "test": 966})
        for split in ("train", "validation"):
            image, mask, sample_id = LumenDataset(splits[split])[0]
            self.assertEqual(tuple(image.shape), (3, 256, 256))
            self.assertEqual(tuple(mask.shape), (1, 256, 256))
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(sample_id.startswith("lumen_"))


if __name__ == "__main__":
    unittest.main()
