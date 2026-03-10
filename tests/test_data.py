"""Tests for trackshot.data.dataset.

Run with::

    pytest tests/test_data.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trackshot.data.dataset import BallDataset, collate_fn, get_dataloader

# ---------------------------------------------------------------------------
# Paths — relative to project root (where pytest is invoked from)
# ---------------------------------------------------------------------------

TRAIN_PATH = Path("data/raw/train")
VAL_PATH = Path("data/raw/valid")
TEST_PATH = Path("data/raw/test")

# Expected image counts from the DFL Soccer Ball Detection dataset
EXPECTED_COUNTS = {
    "train": 230,
    "val": 65,
    "test": 35,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _requires_data(path: Path):
    """Skip test if the raw data directory is not present."""
    if not path.exists():
        pytest.skip(f"Raw data not found at {path} — download the Kaggle dataset first.")


# ---------------------------------------------------------------------------
# 1. Split lengths
# ---------------------------------------------------------------------------


class TestSplitLengths:
    def test_train_length(self):
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=None)
        assert len(ds) == EXPECTED_COUNTS["train"], (
            f"Expected {EXPECTED_COUNTS['train']} train images, got {len(ds)}"
        )

    def test_val_length(self):
        _requires_data(VAL_PATH)
        ds = BallDataset(root=VAL_PATH, img_size=None)
        assert len(ds) == EXPECTED_COUNTS["val"], (
            f"Expected {EXPECTED_COUNTS['val']} val images, got {len(ds)}"
        )

    def test_test_length(self):
        _requires_data(TEST_PATH)
        ds = BallDataset(root=TEST_PATH, img_size=None)
        assert len(ds) == EXPECTED_COUNTS["test"], (
            f"Expected {EXPECTED_COUNTS['test']} test images, got {len(ds)}"
        )


# ---------------------------------------------------------------------------
# 2. Image tensor shape and dtype
# ---------------------------------------------------------------------------


class TestImageTensor:
    def test_shape_with_resize(self):
        """Resized images should be exactly (3, img_size, img_size)."""
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=640)
        image, _ = ds[0]
        assert image.shape == (3, 640, 640), f"Unexpected shape: {image.shape}"

    def test_dtype_float32(self):
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        image, _ = ds[0]
        assert image.dtype == torch.float32, f"Expected float32, got {image.dtype}"

    def test_values_in_unit_range(self):
        """Pixel values must be in [0, 1]."""
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        for idx in range(min(10, len(ds))):
            image, _ = ds[idx]
            assert image.min() >= 0.0, f"Sample {idx}: min value {image.min()} < 0"
            assert image.max() <= 1.0, f"Sample {idx}: max value {image.max()} > 1"

    def test_three_channels(self):
        """Images must be RGB (3 channels)."""
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        image, _ = ds[0]
        assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"


# ---------------------------------------------------------------------------
# 3. Label format
# ---------------------------------------------------------------------------


class TestLabelFormat:
    def test_label_tensor_shape(self):
        """Labels must be (N, 5) — N boxes, 5 values each."""
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        for idx in range(min(20, len(ds))):
            _, labels = ds[idx]
            assert labels.ndim == 2, f"Sample {idx}: labels must be 2D, got {labels.ndim}D"
            assert labels.shape[1] == 5, f"Sample {idx}: expected 5 columns, got {labels.shape[1]}"

    def test_class_index_is_zero(self):
        """Single-class dataset: all class indices must be 0."""
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        for idx in range(len(ds)):
            _, labels = ds[idx]
            if labels.shape[0] > 0:
                assert (labels[:, 0] == 0).all(), (
                    f"Sample {idx}: unexpected class index in {labels[:, 0]}"
                )

    def test_spatial_values_normalized(self):
        """x_c, y_c, w, h must all be in (0, 1]."""
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        for idx in range(len(ds)):
            _, labels = ds[idx]
            if labels.shape[0] > 0:
                spatial = labels[:, 1:]  # (N, 4): x_c, y_c, w, h
                assert spatial.min() > 0.0, f"Sample {idx}: spatial value <= 0: {spatial.min()}"
                assert spatial.max() <= 1.0, f"Sample {idx}: spatial value > 1: {spatial.max()}"

    def test_label_dtype_float32(self):
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        _, labels = ds[0]
        assert labels.dtype == torch.float32, f"Expected float32, got {labels.dtype}"


# ---------------------------------------------------------------------------
# 4. DataLoader / collate
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_collate_fn_output_types(self):
        """collate_fn must return (Tensor, list[Tensor])."""
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        batch = [ds[i] for i in range(4)]
        images, labels = collate_fn(batch)
        assert isinstance(images, torch.Tensor), "images must be a Tensor"
        assert isinstance(labels, list), "labels must be a list"
        assert images.shape == (4, 3, 64, 64)

    def test_dataloader_iterates(self):
        """DataLoader must yield at least one batch without error."""
        _requires_data(TRAIN_PATH)
        ds = BallDataset(root=TRAIN_PATH, img_size=64)
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
        images, labels = next(iter(loader))
        assert images.shape[0] == 4
        assert len(labels) == 4
