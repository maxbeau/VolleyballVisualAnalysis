import json
from typing import List, Dict, Any
import numpy as np

from court.utils import order_corners, corners_from_prediction




def _pick_best_court_pred(preds):
    best = None
    best_conf = -1.0
    for pred in preds or []:
        try:
            label = str(pred.get("class", ""))
        except Exception:
            label = ""
        label_lower = label.strip().lower()
        if label_lower and "court" not in label_lower:
            continue
        conf_val = pred.get("confidence", 0.0)
        try:
            conf = float(conf_val)
        except Exception:
            conf = 0.0
        if conf > best_conf:
            best_conf = conf
            best = pred
    if best is None:
        # fallback: accept any prediction if we never saw an explicit court label
        for pred in preds or []:
            try:
                conf = float(pred.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            if conf > best_conf:
                best_conf = conf
                best = pred
    return best
def load_detections(detections_jsonl: str) -> List[Dict[str, Any]]:
    dets: List[Dict[str, Any]] = []
    with open(detections_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fr = int(rec.get("frame", 0))
            best = rec.get("pred")
            corners = rec.get("corners")
            if not corners:
                if not best:
                    preds = rec.get("predictions")
                    if isinstance(preds, list) and preds:
                        best = _pick_best_court_pred(preds)
                if isinstance(best, dict):
                    try:
                        corners = corners_from_prediction(best)
                    except Exception:
                        corners = None
            if corners and len(corners) >= 4:
                dets.append({"frame": fr, "corners": order_corners([(float(x), float(y)) for x, y in corners[:4]])})
    dets.sort(key=lambda r: r["frame"])
    # Robust outlier filter on detections: drop keyframes with TL too far from median
    if dets:
        tls = np.array([d["corners"][0] for d in dets], dtype=np.float64)
        med = np.median(tls, axis=0)
        diffs = np.linalg.norm(tls - med, axis=1)
        thr = 20.0  # conservative hard threshold (px)
        filtered = [d for d, dist in zip(dets, diffs) if dist <= thr]
        if len(filtered) >= max(2, int(0.6 * len(dets))):
            dets = filtered
    return dets


__all__ = ["load_detections"]
