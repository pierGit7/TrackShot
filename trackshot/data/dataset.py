from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BallDataset(Dataset):
    """YOLO-format dataset for single-class ball detection.

    Expects the following directory layout::

        root/
        ├── images/
        │   ├── img_001.jpg
        │   └── ...
        └── labels/
            ├── img_001.txt   # YOLO format: class x_c y_c w h (normalized)
            └── ...

    Images that have no corresponding label file are treated as
    background samples and return an empty label tensor.

    Args:
        root: Path to split directory (e.g. ``data/raw/train``).
        img_size: Resize images to ``(img_size, img_size)`` before returning.
            Set to ``None`` to keep original resolution.
        transform: Optional additional ``torchvision.transforms`` applied
            *after* the base resize + to-tensor pipeline.
    """

    def __init__(
        self,
        root: str | Path,
        img_size: int | None = 640,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.transform = transform

        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

        # Base transform: resize (if requested) + convert to float tensor [0, 1]
        base_transforms = []
        if self.img_size is not None:
            base_transforms.append(transforms.Resize((self.img_size, self.img_size)))
        base_transforms.append(transforms.ToTensor())  # → [0, 1] float32
        self._base_transform = transforms.Compose(base_transforms)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]

        # --- Image ---
        pil_image = Image.open(img_path).convert("RGB")
        image: torch.Tensor = cast(torch.Tensor, self._base_transform(pil_image))
        if self.transform is not None:
            image = self.transform(image)

        # --- Labels ---
        label_path = self.labels_dir / (img_path.stem + ".txt")
        labels = self._load_labels(label_path)

        return image, labels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_labels(label_path: Path) -> torch.Tensor:
        """Return a ``(N, 5)`` float32 tensor of YOLO labels.

        Each row is ``[class_idx, x_center, y_center, width, height]``
        with all spatial values normalized to ``[0, 1]``.

        Returns an empty ``(0, 5)`` tensor when the label file is missing
        or empty (background / no-ball frame).
        """
        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        rows = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                rows.append([float(v) for v in parts])

        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32)

        return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Custom collate that stacks images and keeps labels as a list.

    Labels cannot be stacked into a single tensor because each image may
    contain a different number of bounding boxes.

    Returns:
        images: ``(B, 3, H, W)`` float32 tensor.
        labels: list of ``(N_i, 5)`` tensors, one per image.
    """
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), list(labels)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def get_dataloader(cfg, split: str = "train") -> DataLoader:
    """Build a :class:`DataLoader` for the requested split.

    Args:
        cfg: Hydra/OmegaConf data config node.  Expected keys:
            ``train_path``, ``val_path``, ``test_path``,
            ``img_size``, ``batch_size``, ``num_workers``.
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    split_map = {
        "train": cfg.train_path,
        "val": cfg.val_path,
        "test": cfg.test_path,
    }
    if split not in split_map:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(split_map)}")

    root = split_map[split]
    dataset = BallDataset(root=root, img_size=cfg.img_size)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
