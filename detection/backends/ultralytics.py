"""Ultralytics Platform inference backend."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.ultralytics_client import UltralyticsClient

from .base import DetectionBackend, ensure_confidence


class UltralyticsBackend(DetectionBackend):
    name = "ultralytics"

    def __init__(self, common_settings, target_settings) -> None:
        super().__init__(common_settings, target_settings)
        api_key_secret = common_settings.ultralytics_api_key
        if not api_key_secret:
            raise RuntimeError(
                "ULTRALYTICS_API_KEY environment variable must be set when using the 'ultralytics' backend."
            )

        target = str(target_settings.get("target") or "").strip()
        endpoint_url = target_settings.get("endpoint")
        if not endpoint_url and target:
            endpoint_url = common_settings.ultralytics_endpoints.get(target)
        if not endpoint_url:
            model_id = str(target_settings.get("model") or "").strip()
            if model_id:
                endpoint_url = f"https://platform.ultralytics.com/api/models/{model_id}/predict"
        if not endpoint_url:
            raise RuntimeError(
                "Ultralytics endpoint is not configured. Set detection.ultralytics_endpoints.<target> "
                "or detection.models_ultralytics.<target>."
            )

        self._endpoint_url = str(endpoint_url)
        self._iou = common_settings.ultralytics_iou
        self._imgsz = common_settings.ultralytics_imgsz
        self._timeout = common_settings.ultralytics_timeout_sec
        self._client = UltralyticsClient(api_key=api_key_secret.get_secret_value())

    def infer(self, frame, *, frame_idx: int, model_id: str, confidence: float) -> Dict[str, Any]:
        payload = self._client.infer_frame(
            frame,
            endpoint_url=self._endpoint_url,
            confidence=ensure_confidence(confidence) or 0.25,
            iou=self._iou,
            imgsz=self._imgsz,
            timeout=self._timeout,
        )
        preds = _normalise_predictions(payload)
        return {
            "predictions": preds,
            "backend": self.name,
            "frame": frame_idx,
            "model": model_id,
            "raw_metadata": payload.get("metadata", {}),
        }


def _normalise_predictions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: Iterable[Any]
    if isinstance(payload.get("images"), list):
        flattened: List[Any] = []
        for image in payload["images"]:
            if isinstance(image, dict) and isinstance(image.get("results"), list):
                flattened.extend(image["results"])
        results = flattened
    elif isinstance(payload.get("results"), list):
        results = payload["results"]
    else:
        results = []

    preds: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        pred = _normalise_one_prediction(item)
        if pred is not None:
            preds.append(pred)
    return preds


def _normalise_one_prediction(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    box = item.get("box") if isinstance(item.get("box"), dict) else {}
    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    class_id = item.get("class")
    class_name = str(item.get("name") or class_id or "")
    pred: Dict[str, Any] = {
        "x": x1 + width * 0.5,
        "y": y1 + height * 0.5,
        "width": width,
        "height": height,
        "confidence": float(item.get("confidence", 0.0) or 0.0),
        "class": class_name,
    }
    if class_id is not None:
        try:
            pred["class_id"] = int(class_id)
        except (TypeError, ValueError):
            pred["class_id"] = class_id

    points = _normalise_points(
        item.get("points")
        or item.get("segments")
        or item.get("segment")
        or item.get("mask")
        or item.get("polygon")
    )
    if points:
        pred["points"] = points
    return pred


def _normalise_points(value: Any) -> Optional[List[Dict[str, float]]]:
    if not value:
        return None
    if isinstance(value, dict):
        xs = value.get("x")
        ys = value.get("y")
        if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys):
            return [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]
        points = value.get("points")
        if points is not value:
            return _normalise_points(points)
        return None
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1 and isinstance(value[0], list):
            return _normalise_points(value[0])
        first = value[0]
        if isinstance(first, dict) and "x" in first and "y" in first:
            return [{"x": float(p["x"]), "y": float(p["y"])} for p in value]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return [{"x": float(p[0]), "y": float(p[1])} for p in value]
    return None


__all__ = ["UltralyticsBackend"]
