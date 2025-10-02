import os
import json
import copy
import argparse
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple

import cv2

from core.utils import ensure_dir
try:
    # Prefer new unified config path
    from config import settings
except Exception:  # fallback for older import path
    from core.config import settings
from core.roboflow_client import RoboflowClient
from court.utils import (
    corners_from_prediction,
    compute_homography,
    build_court_model_template,
    template_precision_score,
)


class LRUCache:
    """Simple LRU cache to bound in-memory detection results."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._store: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any) -> Optional[Any]:
        val = self._store.get(key)
        if val is not None:
            self._store.move_to_end(key)
        return val

    def put(self, key: Any, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


def _parse_multi_scales(raw: Optional[str]) -> List[float]:
    scales: List[float] = []
    if raw:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = float(token)
            except ValueError:
                continue
            if value <= 0.0:
                continue
            if all(abs(value - existing) > 1e-6 for existing in scales):
                scales.append(value)
    if not scales:
        scales = [1.0]
    if all(abs(1.0 - existing) > 1e-6 for existing in scales):
        scales.insert(0, 1.0)
    return scales


def _result_has_scales(result: Dict[str, Any], scales: List[float]) -> bool:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        return False
    if not meta.get("multiscale"):
        return False
    stored = meta.get("scales")
    if not isinstance(stored, list):
        return False
    try:
        stored_vals = {float(val) for val in stored}
    except (TypeError, ValueError):
        return False
    for scale in scales:
        if not any(abs(float(scale) - stored_scale) < 1e-6 for stored_scale in stored_vals):
            return False
    return True


def _rescale_prediction(pred: Dict[str, Any], scale_x: float, scale_y: float) -> Dict[str, Any]:
    safe_scale_x = scale_x if abs(scale_x) > 1e-6 else 1.0
    safe_scale_y = scale_y if abs(scale_y) > 1e-6 else 1.0
    adjusted = copy.deepcopy(pred)
    if "x" in adjusted:
        adjusted["x"] = float(adjusted["x"]) / safe_scale_x
    if "y" in adjusted:
        adjusted["y"] = float(adjusted["y"]) / safe_scale_y
    if "width" in adjusted:
        adjusted["width"] = float(adjusted["width"]) / safe_scale_x
    if "height" in adjusted:
        adjusted["height"] = float(adjusted["height"]) / safe_scale_y
    if "points" in adjusted and isinstance(adjusted["points"], list):
        new_points: List[Any] = []
        for pt in adjusted["points"]:
            if isinstance(pt, dict):
                new_pt = dict(pt)
                if "x" in new_pt:
                    new_pt["x"] = float(new_pt["x"]) / safe_scale_x
                if "y" in new_pt:
                    new_pt["y"] = float(new_pt["y"]) / safe_scale_y
                new_points.append(new_pt)
            else:
                new_points.append(pt)
        adjusted["points"] = new_points
    return adjusted


def _run_multiscale_detection(
    frame,
    client: RoboflowClient,
    *,
    model_id: str,
    confidence: float,
    scales: List[float],
    base_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_h, base_w = frame.shape[:2]
    aggregated: List[Dict[str, Any]] = []
    best_base = copy.deepcopy(base_result) if base_result is not None else {"model_id": model_id}

    for scale in scales:
        if scale <= 0.0:
            continue
        try:
            if abs(scale - 1.0) < 1e-6 and base_result is not None:
                scaled_frame = frame
                result = base_result
            else:
                new_w = max(32, int(round(base_w * scale)))
                new_h = max(32, int(round(base_h * scale)))
                scaled_frame = (
                    frame
                    if (new_w == base_w and new_h == base_h)
                    else cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                )
                result = client.infer_frame(scaled_frame, model_id=model_id, confidence=confidence)
        except Exception:
            continue

        preds = result.get("predictions", []) if isinstance(result, dict) else []
        if not preds:
            continue

        scale_x = scaled_frame.shape[1] / float(base_w)
        scale_y = scaled_frame.shape[0] / float(base_h)
        for pred in preds:
            rescaled_pred = _rescale_prediction(pred, scale_x, scale_y)
            meta = rescaled_pred.get("meta", {}) if isinstance(rescaled_pred.get("meta"), dict) else {}
            meta.update({
                "scale": float(scale),
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
            })
            rescaled_pred["meta"] = meta
            aggregated.append(rescaled_pred)

    aggregated.sort(key=lambda p: float(p.get("confidence", 0.0)), reverse=True)

    best_base["predictions"] = aggregated
    meta = best_base.get("meta") if isinstance(best_base.get("meta"), dict) else {}
    meta.update({
        "multiscale": True,
        "scales": [float(s) for s in scales],
    })
    best_base["meta"] = meta
    return best_base


def choose_best_pred(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Selects the prediction with the highest confidence."""
    preds = result.get("predictions", []) if isinstance(result, dict) else []
    return max(preds, key=lambda p: float(p.get("confidence", 0.0))) if preds else None


def capture_court(
    model_id: str,
    confidence: float,
    interval_sec: float,
    cache_dir: str,
    combined_jsonl: str,
    save_jpegs: bool,
    use_template_score: bool,
    template_min_precision: float,
    template_line_px: int,
    gate_by_template: bool,
) -> None:
    """Captures court detections from a video at a low frame rate."""
    ensure_dir(cache_dir)
    ensure_dir(os.path.dirname(combined_jsonl) or ".")

    video_path = settings.common.VIDEO_PATH
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if not settings.common.ROBOFLOW_API_KEY:
        raise EnvironmentError("ROBOFLOW_API_KEY not found in .env or environment")
    client = RoboflowClient(api_key=settings.common.ROBOFLOW_API_KEY)

    step = max(1, int(round(interval_sec * fps)))
    
    result_cache = LRUCache(int(os.getenv("COURT_RESULT_CACHE_SIZE", "128")))
    cache_hits = 0
    cache_misses = 0
    multi_scale_raw = os.getenv("COURT_MULTI_SCALES", "1.0,0.85,1.15")
    multi_scales = _parse_multi_scales(multi_scale_raw)
    use_multiscale = any(abs(scale - 1.0) > 1e-6 for scale in multi_scales) or len(multi_scales) > 1

    with open(combined_jsonl, "w", encoding="utf-8") as out_f:
        for next_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            raw_json_path = os.path.join(cache_dir, f"frame_{next_idx:06d}.json")

            cache_key = next_idx
            result = result_cache.get(cache_key)
            if result is not None:
                cache_hits += 1
            else:
                cache_misses += 1
                disk_result: Optional[Dict[str, Any]] = None
                if os.path.exists(raw_json_path):
                    with open(raw_json_path, "r", encoding="utf-8") as jf:
                        disk_result = json.load(jf)

                if use_multiscale:
                    if disk_result is not None and _result_has_scales(disk_result, multi_scales):
                        result = disk_result
                    else:
                        base_result = disk_result if disk_result is not None else None
                        result = _run_multiscale_detection(
                            frame,
                            client,
                            model_id=model_id,
                            confidence=confidence,
                            scales=multi_scales,
                            base_result=base_result,
                        )
                        with open(raw_json_path, "w", encoding="utf-8") as jf:
                            json.dump(result, jf, ensure_ascii=False)
                else:
                    if disk_result is not None:
                        result = disk_result
                    else:
                        result = client.infer_frame(frame, model_id=model_id, confidence=confidence)
                        with open(raw_json_path, "w", encoding="utf-8") as jf:
                            json.dump(result, jf, ensure_ascii=False)

                result_cache.put(cache_key, result)
            
            if save_jpegs:
                img_path = os.path.join(cache_dir, f"frame_{next_idx:06d}.jpg")
                if not os.path.exists(img_path):
                    cv2.imwrite(img_path, frame)

            best = choose_best_pred(result)
            corners = corners_from_prediction(best) if best else None

            tpl_prec: Optional[float] = None
            tpl_pass: Optional[bool] = None
            if use_template_score and corners:
                try:
                    # Compute homography image->model and build a template
                    H_img2model, model_size = compute_homography(corners)
                    Wm, Hm = model_size
                    template_mask = build_court_model_template(Wm, Hm, line_px=int(template_line_px), orientation="horizontal")
                    # Score using model->image
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    H_model2img = None
                    try:
                        import numpy as np
                        H_model2img = np.linalg.inv(H_img2model)
                    except Exception:
                        H_model2img = None
                    if H_model2img is not None:
                        tpl_prec = float(template_precision_score(gray, H_model2img, template_mask))
                        tpl_pass = tpl_prec >= float(template_min_precision)
                except Exception:
                    tpl_prec, tpl_pass = None, None

            rec = {
                "frame": next_idx,
                "time_sec": next_idx / fps if fps else None,
                "image_size": {"w": width, "h": height},
                "model_id": model_id,
                "pred": best,
                "corners": corners,
                "raw_json": os.path.relpath(raw_json_path),
                "cached_jpeg": os.path.relpath(img_path) if save_jpegs else None,
            }
            if tpl_prec is not None:
                rec["tpl_prec"] = tpl_prec
            if tpl_pass is not None:
                rec["tpl_pass"] = tpl_pass

            if gate_by_template and use_template_score and (tpl_pass is False):
                # Skip writing this detection when gated out
                continue

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cap.release()
    print(
        "Court capture done. Frames processed: {} (hits: {}, misses: {}), output: {}".format(
            total_frames,
            cache_hits,
            cache_misses,
            combined_jsonl,
        )
    )


def main():
    """Main entry point for the court detection script."""
    parser = argparse.ArgumentParser(description="Capture low-rate court detections to JSONL.")
    parser.add_argument("--model-id", default=os.getenv("COURT_MODEL_ID", "volleyball-court-lurkn/1"))
    parser.add_argument("--confidence", type=float, default=settings.common.OVERLAY_MIN_CONF)
    parser.add_argument("--interval-sec", type=float, default=float(os.getenv("COURT_INTERVAL_SEC", 5.0)))
    # Template precision scoring/gating
    parser.add_argument("--use-template-score", dest="use_template_score", action="store_true", default=None, help="Compute template precision score for each detection")
    parser.add_argument("--no-template-score", dest="use_template_score", action="store_false", help="Disable template precision scoring")
    parser.add_argument("--template-min-precision", type=float, default=None, help="Minimum template precision to accept detection when gating")
    parser.add_argument("--template-line-px", type=int, default=None, help="Template line thickness in pixels")
    parser.add_argument("--gate-by-template", action="store_true", help="Drop detections that fail template precision threshold")
    args = parser.parse_args()

    capture_court(
        model_id=args.model_id,
        confidence=args.confidence,
        interval_sec=args.interval_sec,
        cache_dir=os.getenv("COURT_CACHE_DIR", os.path.join(settings.common.CACHE_DIR, "court")),
        combined_jsonl=settings.court.DETECTIONS_JSONL,
        save_jpegs=settings.court.SAVE_JPEGS,
        use_template_score=(settings.court.DET_USE_TEMPLATE_SCORE if args.use_template_score is None else bool(args.use_template_score)),
        template_min_precision=(settings.court.DET_MIN_TEMPLATE_PREC if args.template_min_precision is None else float(args.template_min_precision)),
        template_line_px=(settings.court.DET_TEMPLATE_LINE_PX if args.template_line_px is None else int(args.template_line_px)),
        gate_by_template=bool(args.gate_by_template),
    )


if __name__ == "__main__":
    main()
