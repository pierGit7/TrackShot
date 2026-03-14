"""Visualize dataset samples with bounding box overlays.

Usage::

    python -m trackshot.data.visualize              # defaults: train split, 16 samples
    python -m trackshot.data.visualize --split val --n 9
    python -m trackshot.data.visualize --split test --n 4 --out logs/test_samples.png

Outputs:
    - A grid image saved to ``logs/dataset_sample.png`` (or --out path)
    - Per-split statistics printed to stdout
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
from dataset import BallDataset
import numpy as np
import torch

# BGR colour for bounding boxes (green)
BOX_COLOR = (0, 200, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_boxes(image_tensor: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """Convert a tensor image + YOLO labels to an annotated BGR numpy array.

    Args:
        image_tensor: ``(3, H, W)`` float32 tensor, values in ``[0, 1]``.
        labels: ``(N, 5)`` float32 tensor — ``[cls, x_c, y_c, w, h]`` normalized.

    Returns:
        Annotated ``(H, W, 3)`` uint8 BGR numpy array.
    """
    # Tensor → HWC uint8 RGB → BGR
    img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    for row in labels:
        _, x_c, y_c, bw, bh = row.tolist()
        # Denormalize to pixel coordinates
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        cv2.putText(
            img_bgr,
            "ball",
            (x1, max(y1 - 4, 10)),
            FONT,
            FONT_SCALE,
            BOX_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return img_bgr


def make_grid(images: list[np.ndarray], ncols: int, cell_size: int = 320) -> np.ndarray:
    """Arrange a list of BGR images into a grid.

    Args:
        images: List of BGR numpy arrays (any size — will be resized to ``cell_size``).
        ncols: Number of columns in the grid.
        cell_size: Each cell is resized to ``(cell_size, cell_size)`` pixels.

    Returns:
        Single BGR numpy array of the assembled grid.
    """
    nrows = math.ceil(len(images) / ncols)
    grid_h = nrows * cell_size
    grid_w = ncols * cell_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row, col = divmod(i, ncols)
        resized = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
        y0, y1 = row * cell_size, (row + 1) * cell_size
        x0, x1 = col * cell_size, (col + 1) * cell_size
        grid[y0:y1, x0:x1] = resized

    return grid


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def print_stats(split: str, dataset: BallDataset) -> None:
    """Print dataset statistics for a split."""
    total = len(dataset)
    n_with_ball = 0
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []

    img_size = dataset.img_size or 640  # fallback for display

    for _, labels in dataset:
        if labels.shape[0] > 0:
            n_with_ball += 1
            for row in labels:
                _, _, _, bw, bh = row.tolist()
                bbox_widths.append(bw * img_size)
                bbox_heights.append(bh * img_size)

    print(f"\n{'─' * 50}")
    print(f"  Split : {split}")
    print(f"  Total images       : {total}")
    print(f"  Images with ball   : {n_with_ball}  ({100 * n_with_ball / max(total, 1):.1f}%)")
    if bbox_widths:
        print(
            f"  BBox width  (px)   : mean={np.mean(bbox_widths):.1f}  "
            f"min={np.min(bbox_widths):.1f}  max={np.max(bbox_widths):.1f}"
        )
        print(
            f"  BBox height (px)   : mean={np.mean(bbox_heights):.1f}  "
            f"min={np.min(bbox_heights):.1f}  max={np.max(bbox_heights):.1f}"
        )
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize TrackerShort dataset samples")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to visualize (default: train)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=16,
        help="Number of samples to display in the grid (default: 16)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Resize images to this size before drawing (default: 640)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="logs/dataset_sample.png",
        help="Output path for the grid image (default: logs/dataset_sample.png)",
    )
    args = parser.parse_args()

    split_paths = {
        "train": "data/raw/train",
        "val": "data/raw/valid",
        "test": "data/raw/test",
    }

    root = split_paths[args.split]
    dataset = BallDataset(root=root, img_size=args.img_size)

    print_stats(args.split, dataset)

    # Sample indices evenly across the dataset
    n = min(args.n, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, n).long().tolist()

    annotated: list[np.ndarray] = []
    for idx in indices:
        image, labels = dataset[idx]
        annotated.append(draw_boxes(image, labels))

    ncols = min(4, n)
    grid = make_grid(annotated, ncols=ncols, cell_size=320)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"\nGrid saved to: {out_path}  ({grid.shape[1]}x{grid.shape[0]} px)\n")


if __name__ == "__main__":
    main()
