"""Trackshot training entry point.

Uses the `ultralytics` YOLOv8 training pipeline, which handles:
  - data augmentation
  - the YOLO detection loss (box + objectness + class)
  - validation loop and mAP metrics every epoch
  - checkpointing (best.pt / last.pt) under runs/

After training (or when a checkpoint is reused), a lightweight validation pass
is run using our own `get_dataloader` from `dataset.py` to verify the model on
the val split and report per-batch inference results.

Checkpoint behaviour
--------------------
By default training is **skipped** if any ``.pt`` file exists in the
``checkpoints/`` directory.  The most recently modified file is used.

To force a full retrain regardless::

    python train.py training.force=true

Normal usage (reuses existing checkpoint if present)::

    python train.py
    python train.py training.epochs=50   # epoch override only applies when training runs
"""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def _find_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the most recently modified trained checkpoint in *checkpoints_dir*.

    Only considers ``best.pt`` and ``last.pt`` — the conventional output names
    used by ultralytics — so pretrained base weights (e.g. ``yolov8n.pt``) that
    live in the same directory are never mistaken for a finished training run.

    Returns ``None`` if the directory doesn't exist or contains no match.
    """
    if not checkpoints_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in checkpoints_dir.iterdir() if p.name in {"best.pt", "last.pt"}),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _write_data_yaml(cfg: DictConfig, dest: Path) -> Path:
    """Generate a YOLO-format data.yaml from the Hydra data config.

    ultralytics requires an absolute path to a data YAML that describes the
    dataset splits.  We derive it from the same ``cfg.data`` config node that
    ``get_dataloader`` uses, so there is a single source of truth.

    Returns the path to the written file.
    """
    # Resolve paths relative to the original working directory (Hydra changes
    # cwd to the run output dir, so we reconstruct from HydraConfig).
    from hydra.core.hydra_config import HydraConfig

    runtime_cwd = Path(HydraConfig.get().runtime.cwd)

    def abs_path(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else runtime_cwd / path)

    data_yaml = {
        "path": abs_path(cfg.data.train_path),  # base; not used directly by splits below
        "train": abs_path(cfg.data.train_path),
        "val": abs_path(cfg.data.val_path),
        "test": abs_path(cfg.data.test_path),
        "nc": int(cfg.data.nc),
        "names": OmegaConf.to_container(cfg.data.names, resolve=True),
    }

    import yaml  # PyYAML – pulled in transitively by ultralytics

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return dest


def _run_inference_example(
    model,
    cfg: DictConfig,
    runtime_cwd: Path,
    device_str: str,
) -> None:
    """Run inference on a single val image and save the annotated result.

    Scans up to 20 evenly-spaced images from :class:`BallDataset` and selects
    the first one where the model detects a ball with confidence >= 90%.
    Falls back to the middle image if no candidate clears the threshold.
    Uses ``result.plot()`` from ultralytics to draw the bounding box and
    confidence score, then saves the annotated frame to
    ``logs/inference_result.png``.

    Args:
        model:       Loaded :class:`ultralytics.YOLO` model.
        cfg:         Hydra config node (needs ``cfg.data``).
        runtime_cwd: Project root path (used to resolve data paths and logs/).
        device_str:  ``"0"`` for CUDA or ``"cpu"``.
    """
    import cv2
    import torch

    from trackshot.data.dataset import BallDataset

    print("─" * 54)
    print("Inference example  (1 image from val split)")
    print("─" * 54)

    # Resolve val path relative to project root
    val_root = Path(cfg.data.val_path)
    if not val_root.is_absolute():
        val_root = runtime_cwd / val_root

    dataset = BallDataset(root=val_root, img_size=cfg.data.img_size)

    # Walk all val images and pick the first one where the model detects a ball
    # with confidence >= 50%. Scanning all images ensures no good frame is skipped.
    # Falls back to the middle image if nothing clears the threshold.
    CONF_THRESHOLD = 0.50
    candidate_indices = list(range(len(dataset)))

    chosen_labels: torch.Tensor | None = None
    chosen_stem: str = ""
    chosen_result = None

    for idx in candidate_indices:
        image_tensor, labels = dataset[idx]
        img_stem = dataset.image_paths[idx].stem

        results = model.predict(
            source=image_tensor.unsqueeze(0),  # (3,H,W) → (1,3,H,W) BCHW
            imgsz=cfg.data.img_size,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=device_str,
        )
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes):
            chosen_labels = labels
            chosen_stem = img_stem
            chosen_result = result
            break

    if chosen_result is None:
        # No candidate had a detection above threshold — fall back to middle image
        print(f"  No detection above {CONF_THRESHOLD:.0%} in any candidate; using middle image.")
        fallback_idx = len(dataset) // 2
        fallback_tensor, chosen_labels = dataset[fallback_idx]
        chosen_stem = dataset.image_paths[fallback_idx].stem
        fallback_results = model.predict(
            source=fallback_tensor.unsqueeze(0),  # (3,H,W) → (1,3,H,W) BCHW
            imgsz=cfg.data.img_size,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=device_str,
        )
        chosen_result = fallback_results[0]

    result = chosen_result
    assert chosen_labels is not None
    labels = chosen_labels
    img_stem = chosen_stem

    # Build confidence list for the summary line
    boxes = result.boxes
    confs: list[float] = (
        [round(float(c), 2) for c in boxes.conf.tolist()]
        if boxes is not None and len(boxes)
        else []
    )
    n_pred = len(confs)
    n_gt = labels.shape[0]

    print(f"  {img_stem:<20} | GT: {n_gt} | pred: {n_pred} | conf: {confs}")

    # ultralytics .plot() returns an annotated BGR uint8 numpy array
    # with bounding boxes and confidence scores drawn on the image
    annotated = result.plot()

    out_path = runtime_cwd / "logs" / "inference_result.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    print(f"\nResult saved → {out_path.relative_to(runtime_cwd)}")
    print("─" * 54)


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    import torch

    from trackshot.data.dataset import get_dataloader
    from trackshot.models.architecture import build_model

    print("=" * 60)
    print("Trackshot – YOLOv8n ball detection training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device_str = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {'CUDA' if device_str == '0' else 'CPU'}\n")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_model(cfg.model)

    # ------------------------------------------------------------------
    # Checkpoint detection
    #
    # Skip training when a .pt file already exists in checkpoints/ and
    # the --force flag has not been set.
    # ------------------------------------------------------------------
    from hydra.core.hydra_config import HydraConfig

    runtime_cwd = Path(HydraConfig.get().runtime.cwd)
    checkpoints_dir = runtime_cwd / "checkpoints"

    existing_checkpoint = _find_checkpoint(checkpoints_dir)

    if existing_checkpoint and not cfg.training.force:
        print(f"Checkpoint found: {existing_checkpoint}")
        print("Skipping training. Use 'training.force=true' to retrain.\n")
        best_weights = existing_checkpoint
    else:
        if cfg.training.force:
            print("Force flag set — retraining from scratch.\n")

        # ------------------------------------------------------------------
        # Write data.yaml for ultralytics (derived from cfg.data)
        # ------------------------------------------------------------------
        data_yaml_path = _write_data_yaml(cfg, Path("data.yaml"))
        print(f"data.yaml written to: {data_yaml_path}\n")

        # ------------------------------------------------------------------
        # Train with ultralytics
        #
        # ultralytics handles:
        #   • data loading + augmentation
        #   • YOLO detection loss (CIoU box + BCE objectness + BCE class)
        #   • validation mAP metrics every epoch
        #   • best.pt / last.pt checkpointing under runs/detect/
        # ------------------------------------------------------------------
        results = model.train(
            data=str(data_yaml_path),
            epochs=cfg.training.epochs,
            imgsz=cfg.data.img_size,
            batch=cfg.data.batch_size,
            lr0=cfg.training.lr,
            device=device_str,
            seed=cfg.seed,
            project=str(runtime_cwd / "runs" / "detect"),  # absolute → no Hydra cwd doubling
            name="trackshot",
            exist_ok=True,
            verbose=True,
        )

        trained_best = Path(results.save_dir) / "weights" / "best.pt"

        # Mirror best.pt into checkpoints/ so future runs find it
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_weights = checkpoints_dir / "best.pt"
        shutil.copy(trained_best, best_weights)

        print("\n" + "=" * 60)
        print("Training complete.")
        print(f"Best weights saved to: {best_weights}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Post-training validation pass using our own BallDataset / DataLoader
    #
    # This exercises the get_dataloader() path from dataset.py and shows
    # per-batch inference results on the val split.
    # ------------------------------------------------------------------
    print("Running post-training validation with BallDataset dataloader …\n")

    val_loader = get_dataloader(cfg.data, split="val")

    # Load best weights for inference
    from ultralytics import YOLO as _YOLO

    inference_model = _YOLO(str(best_weights))

    total_batches = 0
    total_detections = 0

    for batch_idx, (images, labels) in enumerate(val_loader):
        # images: (B, 3, H, W) float32  –  same as BallDataset collate_fn output
        # labels: list[Tensor(N_i, 5)]   –  per-image YOLO labels (class+xywh)
        batch_results = inference_model.predict(
            source=images,  # ultralytics accepts batched tensor
            imgsz=cfg.data.img_size,
            conf=0.25,
            verbose=False,
            device=device_str,
        )

        batch_detections = sum(len(r.boxes) for r in batch_results)
        total_detections += batch_detections
        total_batches += 1

        if batch_idx < 3:  # print first 3 batches
            gt_boxes = sum(l.shape[0] for l in labels)
            print(
                f"  Batch {batch_idx:>3} | images: {images.shape[0]} "
                f"| GT boxes: {gt_boxes:>3} | detections: {batch_detections:>3}"
            )

    print(f"\nVal split – {total_batches} batches, {total_detections} total detections.")

    # ------------------------------------------------------------------
    # Inference example — 8 annotated val samples saved as a grid image
    # ------------------------------------------------------------------
    _run_inference_example(inference_model, cfg, runtime_cwd, device_str)

    print("\nDone.")


if __name__ == "__main__":
    main()
