import os
import json
import math
import time
import argparse
import re
import cv2
from typing import Dict, List
from utils import load_env_file, ensure_dir, pick_video_path
from roboflow_client import RoboflowClient


def infer_ball_on_video(
    api_key: str,
    model_id: str = "volleyball_v2/3",
    confidence: float = 0.25,
    fps_sample: float = 10.0,
    max_frames: int = 0,
    cache_dir: str = "outputs/preds",
    combined_jsonl: str = "outputs/ball_detections.jsonl",
    cache_only: bool = False,
) -> None:
    # Prepare IO
    ensure_dir(cache_dir)
    ensure_dir(os.path.dirname(combined_jsonl) or ".")

    env = load_env_file()
    video_path, _ = pick_video_path(env)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Determine sampling stride
    stride = max(1, int(round(vid_fps / max(0.1, fps_sample))))
    if fps_sample <= 0:
        stride = 1

    # Roboflow-only client (isolated module for easy future swap)
    client = RoboflowClient(api_key=api_key, base_url=env.get("ROBOFLOW_API_URL", "https://detect.roboflow.com"))

    # Iterate and sample frames
    frame_idx = 0
    sampled_count = 0
    t0 = time.time()
    def write_records(records: List[dict]):
        with open(combined_jsonl, "w", encoding="utf-8") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    records: List[dict] = []
    # Optional memory-I/O toggles
    def parse_bool(s, default=False):
        if s is None:
            return default
        return str(s).strip().lower() in ("1", "true", "yes", "on", "y")

    save_jpegs = parse_bool(env.get("BALL_SAVE_JPEGS", "false"), False)
    save_frame_json = parse_bool(env.get("BALL_SAVE_FRAME_JSON", "false"), False)

    if cache_only:
        # Rebuild combined jsonl from existing per-frame json cache
        pattern = re.compile(r"frame_(\d{6})\.json$")
        cached = []
        for name in os.listdir(cache_dir):
            m = pattern.match(name)
            if not m:
                continue
            idx = int(m.group(1))
            cached.append((idx, os.path.join(cache_dir, name)))
        cached.sort(key=lambda x: x[0])
        for idx, jp in cached:
            with open(jp, "r", encoding="utf-8") as jf:
                result = json.load(jf)
            preds = result.get("predictions", []) if isinstance(result, dict) else []
            records.append({
                "frame": idx,
                "time_sec": idx / vid_fps if vid_fps else None,
                "image_size": {"w": width, "h": height},
                "model_id": model_id,
                "confidence": confidence,
                "predictions": preds,
            })
        write_records(records)
        print(f"Cache-only mode: rebuilt {len(records)} records from {cache_dir} into {combined_jsonl}")
    else:
        with open(combined_jsonl, "w", encoding="utf-8") as out_f:
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    frame_idx += 1
                    continue

                # Optional JPEG caching (disabled by default)
                if save_jpegs:
                    tmp_path = os.path.join(cache_dir, f"frame_{frame_idx:06d}.jpg")
                    if not os.path.exists(tmp_path):
                        cv2.imwrite(tmp_path, frame)

                # Cache JSON path per frame (idempotent)
                json_path = os.path.join(cache_dir, f"frame_{frame_idx:06d}.json")
                if save_frame_json and os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as jf:
                        result = json.load(jf)
                else:
                    # Inference call on in-memory frame to avoid disk I/O
                    result = client.infer_frame(frame, model_id=model_id, confidence=confidence)
                    # Persist per-frame prediction for reuse (optional)
                    if save_frame_json:
                        with open(json_path, "w", encoding="utf-8") as jf:
                            json.dump(result, jf, ensure_ascii=False)

                # Build compact record for tracking pipeline
                preds = result.get("predictions", []) if isinstance(result, dict) else []
                record = {
                    "frame": frame_idx,
                    "time_sec": frame_idx / vid_fps if vid_fps else None,
                    "image_size": {"w": width, "h": height},
                    "model_id": model_id,
                    "confidence": confidence,
                    "predictions": preds,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                sampled_count += 1
                frame_idx += 1

                if max_frames and sampled_count >= max_frames:
                    break

    cap.release()
    dt = time.time() - t0
    print(
        f"Done. Video: {video_path} | frames: {total_frames} | sampled: {sampled_count} | stride: {stride} | time: {dt:.1f}s"
    )
    print(f"Per-frame cache: {cache_dir}")
    print(f"Combined detections: {combined_jsonl}")


def main():
    env = load_env_file()
    api_key = env.get("ROBOFLOW_API_KEY") or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY not found in .env or environment")

    parser = argparse.ArgumentParser(description="Detect ball using Roboflow with per-frame cache and JSONL output")
    parser.add_argument("--model-id", default=env.get("ROBOFLOW_MODEL_ID", "volleyball_v2/3"))
    parser.add_argument("--confidence", type=float, default=float(env.get("ROBOFLOW_CONFIDENCE", 0.25)))
    parser.add_argument("--infer-fps", type=float, default=float(env.get("INFER_FPS", 8)))
    parser.add_argument("--max-frames", type=int, default=int(env.get("MAX_FRAMES", 0)))
    parser.add_argument("--cache-dir", default=env.get("CACHE_DIR", "outputs/preds"))
    parser.add_argument("--combined-jsonl", default=env.get("COMBINED_JSONL", "outputs/ball_detections.jsonl"))
    parser.add_argument("--cache-only", action="store_true", help="Rebuild combined JSONL from existing cache; no API calls")
    args = parser.parse_args()

    infer_ball_on_video(
        api_key=api_key,
        model_id=args.model_id,
        confidence=args.confidence,
        fps_sample=args.infer_fps,
        max_frames=args.max_frames,
        cache_dir=args.cache_dir,
        combined_jsonl=args.combined_jsonl,
        cache_only=args.cache_only,
    )


if __name__ == "__main__":
    main()
