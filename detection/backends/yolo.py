"""Local YOLO backend powered by ultralytics."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base import DetectionBackend, ensure_confidence


class LocalYOLOBackend(DetectionBackend):
    name = "local-yolo"

    def __init__(self, common_settings, target_settings) -> None:
        super().__init__(common_settings, target_settings)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for local YOLO inference. "
                "Install it via 'pip install ultralytics'."
            ) from exc

        model_path = target_settings.get("model") or common_settings.models_yolo.default_model
        weights_path = Path(model_path)
        if not weights_path.is_absolute():
            weights_path = Path(os.getcwd()) / weights_path
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Local YOLO weights not found at {weights_path}. "
                "Configure model paths under `detection.models_yolo` in config/detection.yaml."
            )

        self._device = common_settings.models_yolo.device
        self._model = YOLO(str(weights_path))
        self._names = self._model.model.names if hasattr(self._model, "model") else self._model.names

    def infer(self, frame, *, frame_idx: int, model_id: str, confidence: float) -> Dict[str, Any]:
        conf = ensure_confidence(confidence) or 0.25
        results = self._model(
            frame,
            verbose=False,
            conf=conf,
            device=self._device,
        )
        if not results:
            return {"predictions": []}
        result = results[0]
        if not hasattr(result, "boxes") or result.boxes is None:
            return {"predictions": []}

        boxes = result.boxes
        preds: List[Dict[str, Any]] = []

        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        classes = _to_numpy(boxes.cls).astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(float, box)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + w * 0.5
            cy = y1 + h * 0.5
            conf_val = float(confs[idx]) if confs.size > idx else conf
            cls_id = int(classes[idx]) if classes.size > idx else 0
            class_name = self._names.get(cls_id, str(cls_id)) if isinstance(self._names, dict) else str(cls_id)
            preds.append(
                {
                    "x": cx,
                    "y": cy,
                    "width": w,
                    "height": h,
                    "confidence": conf_val,
                    "class": class_name,
                    "class_id": cls_id,
                }
            )
        return {"predictions": preds, "backend": self.name, "frame": frame_idx}


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    try:
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=np.float32)
    except Exception:  # pragma: no cover - fallback path
        return np.zeros((0,), dtype=np.float32)


__all__ = ["LocalYOLOBackend"]
