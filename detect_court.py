import os
import json
import math
import time
import argparse
from statistics import median
from typing import Dict, List, Tuple, Optional

import cv2

from utils import load_env_file, ensure_dir, pick_video_path
from roboflow_client import RoboflowClient


def parse_polygon_from_pred(pred: Dict) -> Optional[List[Tuple[float, float]]]:
    pts = pred.get("points")
    if pts is None:
        return None
    # Accept formats: [{"x":..,"y":..}, ...] or [[x,y], ...] or {"x":[...],"y":[...]}
    if isinstance(pts, list) and len(pts) > 0:
        first = pts[0]
        if isinstance(first, dict) and "x" in first and "y" in first:
            return [(float(p["x"]), float(p["y"])) for p in pts]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return [(float(p[0]), float(p[1])) for p in pts]
    if isinstance(pts, dict) and "x" in pts and "y" in pts:
        xs, ys = pts["x"], pts["y"]
        if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys):
            return [(float(x), float(y)) for x, y in zip(xs, ys)]
    return None


def corners_from_polygon_extremes(poly: List[Tuple[float, float]]):
    # Pick TL, TR, BR, BL directly from polygon by extreme sums/diffs (no minAreaRect)
    # This better preserves the projective trapezoid shape of the court outline.
    import numpy as np

    pts = np.array(poly, dtype=np.float32)
    s = pts[:, 0] + pts[:, 1]  # x+y
    d = pts[:, 0] - pts[:, 1]  # x-y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1])), (float(br[0]), float(br[1])), (float(bl[0]), float(bl[1]))]


def rect_from_bbox(pred: Dict):
    # Fallback: use axis-aligned bbox if polygon missing
    x = float(pred.get("x", 0.0))
    y = float(pred.get("y", 0.0))
    w = float(pred.get("width", 0.0))
    h = float(pred.get("height", 0.0))
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def order_corners(pts: List[Tuple[float, float]]):
    import numpy as np

    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1])), (float(br[0]), float(br[1])), (float(bl[0]), float(bl[1]))]


def corners_from_prediction(pred: Dict) -> Optional[List[Tuple[float, float]]]:
    poly = parse_polygon_from_pred(pred)
    if poly and len(poly) >= 4:
        # Use polygon extremes to preserve trapezoid, then order
        corners = corners_from_polygon_extremes(poly)
        return order_corners(corners)
    # fallback to bbox corners (rectangle)
    corners = rect_from_bbox(pred)
    return order_corners(corners)


def magnitude_corners_delta(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    # mean point-to-point distance
    import numpy as np

    pa = np.array(a, dtype=float)
    pb = np.array(b, dtype=float)
    return float(np.linalg.norm(pa - pb, axis=1).mean())


def integrate_corners_median(samples: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    # samples are ordered TL,TR,BR,BL
    out = []
    for i in range(4):
        xs = [p[i][0] for p in samples]
        ys = [p[i][1] for p in samples]
        out.append((median(xs), median(ys)))
    return out


def integrate_corners_ema(samples: List[List[Tuple[float, float]]], alpha: float = 0.3) -> List[Tuple[float, float]]:
    if not samples:
        return []
    ema = [list(samples[0][i]) for i in range(4)]
    for s in samples[1:]:
        for i in range(4):
            ema[i][0] = alpha * s[i][0] + (1 - alpha) * ema[i][0]
            ema[i][1] = alpha * s[i][1] + (1 - alpha) * ema[i][1]
    return [(float(x), float(y)) for x, y in ema]


def detect_court(
    api_key: str,
    model_id: str = "volleyball-court-lurkn/1",
    confidence: float = 0.25,
    interval_sec: float = 5.0,
    min_interval_sec: float = 2.0,
    change_thresh_px: float = 20.0,
    cache_dir: str = "outputs/court_preds",
    combined_jsonl: str = "outputs/court_detections.jsonl",
    integrated_json: str = "outputs/court_corners_integrated.json",
) -> None:
    ensure_dir(cache_dir)
    ensure_dir(os.path.dirname(combined_jsonl) or ".")
    ensure_dir(os.path.dirname(integrated_json) or ".")

    env = load_env_file()
    video_path, _ = pick_video_path(env)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    client = RoboflowClient(api_key=api_key, base_url=env.get("ROBOFLOW_API_URL", "https://detect.roboflow.com"))

    samples: List[Dict] = []  # each: {frame, time_sec, corners, confidence, raw_json_path}
    records: List[Dict] = []

    next_idx = 0
    last_corners: Optional[List[Tuple[float, float]]] = None
    dyn_interval = interval_sec

    with open(combined_jsonl, "w", encoding="utf-8") as out_f:
        while next_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            tmp_path = os.path.join(cache_dir, f"frame_{next_idx:06d}.jpg")
            json_path = os.path.join(cache_dir, f"frame_{next_idx:06d}.json")

            if not os.path.exists(tmp_path):
                cv2.imwrite(tmp_path, frame)

            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as jf:
                    result = json.load(jf)
            else:
                result = client.infer_image(tmp_path, model_id=model_id, confidence=confidence)
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(result, jf, ensure_ascii=False)

            # choose best prediction (highest confidence)
            preds = result.get("predictions", []) if isinstance(result, dict) else []
            best = None
            for p in preds:
                if best is None or float(p.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
                    best = p

            corners = None
            if best is not None:
                corners = corners_from_prediction(best)

            rec = {
                "frame": next_idx,
                "time_sec": next_idx / fps if fps else None,
                "image_size": {"w": width, "h": height},
                "model_id": model_id,
                "confidence": confidence,
                "raw_json": os.path.relpath(json_path),
                "corners": corners,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if corners is not None:
                samples.append({
                    "frame": next_idx,
                    "time_sec": rec["time_sec"],
                    "corners": corners,
                })
                # dynamic interval adjustment
                if last_corners is not None:
                    delta = magnitude_corners_delta(corners, last_corners)
                    dyn_interval = min_interval_sec if delta >= change_thresh_px else interval_sec
                last_corners = corners

            # step to next
            step = max(1, int(round(dyn_interval * fps)))
            next_idx += step

    cap.release()

    # integrate
    corners_list = [s["corners"] for s in samples]
    integrated_med = integrate_corners_median(corners_list) if corners_list else None
    integrated_ema = integrate_corners_ema(corners_list, alpha=0.3) if corners_list else None

    with open(integrated_json, "w", encoding="utf-8") as f:
        json.dump({
            "samples": len(corners_list),
            "method": "median+ema",
            "median": integrated_med,
            "ema": integrated_ema,
            "all_samples": samples,  # allow you to decide how many to integrate later
        }, f, ensure_ascii=False, indent=2)

    print(
        f"Court detection done. Video frames: {total_frames}, samples: {len(samples)}, integrated saved: {integrated_json}"
    )


def main():
    env = load_env_file()
    api_key = env.get("ROBOFLOW_API_KEY") or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY not found in .env or environment")

    parser = argparse.ArgumentParser(description="Detect court contours periodically and integrate corners")
    parser.add_argument("--model-id", default=os.environ.get("COURT_MODEL_ID", "volleyball-court-lurkn/1"))
    parser.add_argument("--confidence", type=float, default=float(env.get("ROBOFLOW_CONFIDENCE", 0.25)))
    parser.add_argument("--interval-sec", type=float, default=float(env.get("COURT_INTERVAL_SEC", 5.0)))
    parser.add_argument("--min-interval-sec", type=float, default=float(env.get("COURT_INTERVAL_SEC_MIN", 2.0)))
    parser.add_argument("--change-thresh-px", type=float, default=float(env.get("COURT_CHANGE_THRESH_PX", 20.0)))
    parser.add_argument("--cache-dir", default=env.get("COURT_CACHE_DIR", "outputs/court_preds"))
    parser.add_argument("--combined-jsonl", default=env.get("COURT_COMBINED_JSONL", "outputs/court_detections.jsonl"))
    parser.add_argument("--integrated-json", default=env.get("COURT_INTEGRATED_JSON", "outputs/court_corners_integrated.json"))
    args = parser.parse_args()

    detect_court(
        api_key=api_key,
        model_id=args.model_id,
        confidence=args.confidence,
        interval_sec=args.interval_sec,
        min_interval_sec=args.min_interval_sec,
        change_thresh_px=args.change_thresh_px,
        cache_dir=args.cache_dir,
        combined_jsonl=args.combined_jsonl,
        integrated_json=args.integrated_json,
    )


if __name__ == "__main__":
    main()
