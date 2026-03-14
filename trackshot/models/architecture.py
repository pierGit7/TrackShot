"""Model architecture for Trackshot.

The production model is YOLOv8n via the `ultralytics` library.  It is the
smallest YOLO variant and the intended export target for TFLite deployment on
the ESP32-S3.

`build_model` is the single public entry point used by `train.py`.
"""

from __future__ import annotations

from omegaconf import DictConfig
from ultralytics import YOLO


def build_model(cfg: DictConfig) -> YOLO:
    """Return a :class:`ultralytics.YOLO` model ready for training.

    Args:
        cfg: The ``model`` config node.  Expected keys:

            * ``model_type`` – YOLO variant string, e.g. ``"yolov8n"``.
            * ``weights``    – Path / hub tag for pre-trained weights, e.g.
              ``"yolov8n.pt"``.  Set to ``null`` to train from scratch using
              the bundled YAML architecture definition (``"yolov8n.yaml"``).
            * ``nc``         – Number of detection classes.

    Returns:
        A :class:`~ultralytics.YOLO` instance.  The ``.train()`` method is
        called by the training script; ``build_model`` only constructs and
        (optionally) loads pre-trained weights.
    """
    weights: str | None = cfg.get("weights", None)

    if weights:
        model = YOLO(weights)  # loads pretrained backbone + head
    else:
        # Train from scratch using the architecture YAML
        model = YOLO(f"{cfg.model_type}.yaml")

    return model
