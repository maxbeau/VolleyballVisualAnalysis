from typing import Any, Dict, List, Tuple


def refine_detections(
    detections: List[Dict[str, Any]],
    frame_shape: Tuple[int, int],
    settings,
) -> List[Dict[str, Any]]:
    """Drop implausible boxes and reward court-aligned predictions."""
    if not detections:
        return detections
    height, width = frame_shape
    min_box_ratio = float(getattr(settings.players, "MIN_BOX_RATIO", 0.02))
    max_aspect = float(getattr(settings.players, "MAX_ASPECT_RATIO", 4.0))
    aspect_bypass_conf = float(getattr(settings.players, "ASPECT_BYPASS_CONF", 0.9))
    aspect_bypass_height = float(getattr(settings.players, "ASPECT_BYPASS_HEIGHT_RATIO", 0.22))
    conf_bonus = float(getattr(settings.players, "CONF_BONUS_INSIDE_COURT", 0.0))

    min_wh = min(width, height) * min_box_ratio
    min_area = (min_wh ** 2)

    refined: List[Dict[str, Any]] = []
    for det in detections:
        try:
            w = float(det.get("width", 0.0))
            h = float(det.get("height", 0.0))
            conf = float(det.get("confidence", 0.0))
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        area = w * h
        if area < min_area:
            continue
        aspect = max(w, h) / max(1e-6, min(w, h))
        if aspect > max_aspect:
            keep_by_conf = conf >= aspect_bypass_conf
            keep_by_height = (h / max(1e-6, height)) >= aspect_bypass_height
            court_meta = det.get("_court") if isinstance(det.get("_court"), dict) else {}
            keep_by_outside = bool(court_meta.get("outside")) or bool(court_meta.get("soft_inside"))
            if not (keep_by_conf or keep_by_height or keep_by_outside):
                continue
        det_out = det
        if conf_bonus > 0.0 and isinstance(det.get("_court"), dict):
            det_out = det.copy()
            det_out["confidence"] = float(min(1.0, conf + conf_bonus))
        refined.append(det_out)
    return refined


__all__ = ["refine_detections"]
