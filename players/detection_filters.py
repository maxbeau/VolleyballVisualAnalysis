from typing import Any, Dict, List, Tuple


def _to_tlbr(det: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x = float(det.get("x", 0.0))
    y = float(det.get("y", 0.0))
    w = float(det.get("width", 0.0))
    h = float(det.get("height", 0.0))
    half_w = w * 0.5
    half_h = h * 0.5
    return (x - half_w, y - half_h, x + half_w, y + half_h)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


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

    if not refined:
        return refined

    dup_iou = float(getattr(settings.players, "DET_DUP_IOU", 0.6))
    dup_iou = max(0.0, min(1.0, dup_iou))
    if dup_iou <= 0.0:
        return refined

    refined.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    kept_boxes: List[Tuple[float, float, float, float]] = []
    for det in refined:
        box = _to_tlbr(det)
        should_keep = True
        for kb in kept_boxes:
            if _iou(box, kb) >= dup_iou:
                should_keep = False
                break
        if should_keep:
            kept.append(det)
            kept_boxes.append(box)
    return kept


__all__ = ["refine_detections"]
