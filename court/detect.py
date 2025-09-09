import os
import json
import argparse
from typing import Dict, Any, Optional, List, Tuple

import cv2

from core.utils import ensure_dir
from core.config import settings
from core.roboflow_client import RoboflowClient
from court.utils import corners_from_prediction


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
) -> None:
    """Captures court detections from a video at a low frame rate."""
    ensure_dir(cache_dir)
    ensure_dir(os.path.dirname(combined_jsonl) or ".")

    video_path = settings.VIDEO_PATH
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if not settings.ROBOFLOW_API_KEY:
        raise EnvironmentError("ROBOFLOW_API_KEY not found in .env or environment")
    client = RoboflowClient(api_key=settings.ROBOFLOW_API_KEY)

    step = max(1, int(round(interval_sec * fps)))
    
    with open(combined_jsonl, "w", encoding="utf-8") as out_f:
        for next_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            raw_json_path = os.path.join(cache_dir, f"frame_{next_idx:06d}.json")
            
            if os.path.exists(raw_json_path):
                with open(raw_json_path, "r", encoding="utf-8") as jf:
                    result = json.load(jf)
            else:
                result = client.infer_frame(frame, model_id=model_id, confidence=confidence)
                with open(raw_json_path, "w", encoding="utf-8") as jf:
                    json.dump(result, jf, ensure_ascii=False)
            
            if save_jpegs:
                img_path = os.path.join(cache_dir, f"frame_{next_idx:06d}.jpg")
                if not os.path.exists(img_path):
                    cv2.imwrite(img_path, frame)

            best = choose_best_pred(result)
            corners = corners_from_prediction(best) if best else None

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
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cap.release()
    print(f"Court capture done. Frames processed: {total_frames}, output: {combined_jsonl}")


def main():
    """Main entry point for the court detection script."""
    parser = argparse.ArgumentParser(description="Capture low-rate court detections to JSONL.")
    parser.add_argument("--model-id", default=os.getenv("COURT_MODEL_ID", "volleyball-court-lurkn/1"))
    parser.add_argument("--confidence", type=float, default=settings.OVERLAY_MIN_CONF)
    parser.add_argument("--interval-sec", type=float, default=float(os.getenv("COURT_INTERVAL_SEC", 5.0)))
    args = parser.parse_args()

    capture_court(
        model_id=args.model_id,
        confidence=args.confidence,
        interval_sec=args.interval_sec,
        cache_dir=os.getenv("COURT_CACHE_DIR", "outputs/court_preds"),
        combined_jsonl=settings.COURT_DETECTIONS_JSONL,
        save_jpegs=settings.COURT_SAVE_JPEGS,
    )


if __name__ == "__main__":
    main()
