import os
import json
import argparse
from typing import Dict, Any, Optional, List, Tuple

import cv2

from utils import load_env_file, ensure_dir, pick_video_path
from roboflow_client import RoboflowClient
from court_utils import corners_from_prediction


def choose_best_pred(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    preds = result.get("predictions", []) if isinstance(result, dict) else []
    best = None
    for p in preds:
        if best is None or float(p.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
            best = p
    return best


def capture_court(
    api_key: str,
    model_id: str,
    confidence: float,
    interval_sec: float,
    cache_dir: str,
    combined_jsonl: str,
) -> None:
    ensure_dir(cache_dir)
    ensure_dir(os.path.dirname(combined_jsonl) or ".")

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

    step = max(1, int(round(interval_sec * fps)))
    next_idx = 0

    save_jpegs = (str(env.get("COURT_SAVE_JPEGS", "false")).strip().lower() in ("1", "true", "yes", "on"))

    with open(combined_jsonl, "w", encoding="utf-8") as out_f:
        while next_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            img_path = os.path.join(cache_dir, f"frame_{next_idx:06d}.jpg")
            raw_json_path = os.path.join(cache_dir, f"frame_{next_idx:06d}.json")

            if save_jpegs and not os.path.exists(img_path):
                cv2.imwrite(img_path, frame)

            if os.path.exists(raw_json_path):
                with open(raw_json_path, "r", encoding="utf-8") as jf:
                    result = json.load(jf)
            else:
                # In-memory inference to avoid disk I/O
                result = client.infer_frame(frame, model_id=model_id, confidence=confidence)
                with open(raw_json_path, "w", encoding="utf-8") as jf:
                    json.dump(result, jf, ensure_ascii=False)

            best = choose_best_pred(result) or None
            corners = corners_from_prediction(best) if best is not None else None

            rec = {
                "frame": int(next_idx),
                "time_sec": (float(next_idx) / float(fps)) if fps else None,
                "image_size": {"w": int(width), "h": int(height)},
                "model_id": model_id,
                "pred": best,
                "corners": corners,
                "raw_json": os.path.relpath(raw_json_path),
                "cached_jpeg": os.path.relpath(img_path) if save_jpegs else None,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            next_idx += step

    cap.release()
    print(f"Court capture done. Frames: {total_frames}, written: {combined_jsonl}")


def main():
    env = load_env_file()
    api_key = env.get("ROBOFLOW_API_KEY") or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY not found in .env or environment")

    parser = argparse.ArgumentParser(description="Capture low-rate court detections to JSONL")
    parser.add_argument("--model-id", default=os.environ.get("COURT_MODEL_ID", env.get("COURT_MODEL_ID", "volleyball-court-lurkn/1")))
    parser.add_argument("--confidence", type=float, default=float(env.get("ROBOFLOW_CONFIDENCE", 0.25)))
    parser.add_argument("--interval-sec", type=float, default=float(env.get("COURT_INTERVAL_SEC", 5.0)))
    parser.add_argument("--cache-dir", default=env.get("COURT_CACHE_DIR", "outputs/court_preds"))
    parser.add_argument("--combined-jsonl", default=env.get("COURT_COMBINED_JSONL", "outputs/court_detections.jsonl"))
    args = parser.parse_args()

    capture_court(
        api_key=api_key,
        model_id=args.model_id,
        confidence=args.confidence,
        interval_sec=args.interval_sec,
        cache_dir=args.cache_dir,
        combined_jsonl=args.combined_jsonl,
    )


if __name__ == "__main__":
    main()
