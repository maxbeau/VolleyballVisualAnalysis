import os
import json
import time
import argparse
import re
from typing import Dict, List, Optional, Tuple

import cv2

from core.utils import ensure_dir
from core.config import settings
from core.roboflow_client import RoboflowClient


def infer_players_on_video(
    model_id: str,
    confidence: float,
    fps_sample: float,
    max_frames: Optional[int],
    cache_dir: str,
    combined_jsonl: str,
    save_frame_json: bool,
    cache_only: bool,
) -> None:
    """Run Roboflow players model over the video with sampling and cache outputs.

    Produces per-frame JSON cache in `cache_dir` and a combined JSONL file with
    predictions keyed by frame index.
    """
    ensure_dir(cache_dir)
    ensure_dir(os.path.dirname(combined_jsonl) or ".")

    video_path = settings.VIDEO_PATH
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Determine sampling stride (frame step)
    stride = max(1, int(round(vid_fps / max(0.1, fps_sample)))) if fps_sample > 0 else 1

    # Roboflow client
    if not settings.ROBOFLOW_API_KEY:
        raise EnvironmentError("ROBOFLOW_API_KEY not found in .env or environment")
    client = RoboflowClient(api_key=settings.ROBOFLOW_API_KEY)

    t0 = time.time()

    if cache_only:
        records = _rebuild_from_cache(cache_dir, model_id, confidence, vid_fps, width, height)
        sampled_count = len(records)
    else:
        records, sampled_count = _run_inference_loop(
            cap, stride, model_id, confidence, max_frames, cache_dir,
            save_frame_json, vid_fps, width, height, client, total_frames
        )

    # Write combined JSONL
    with open(combined_jsonl, "w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cap.release()
    dt = time.time() - t0
    print(
        f"Players detection done. Video: {video_path} | frames: {total_frames} | sampled: {sampled_count} | stride: {stride} | time: {dt:.1f}s"
    )
    print(f"Per-frame cache: {cache_dir}")
    print(f"Combined detections: {combined_jsonl}")


def _rebuild_from_cache(cache_dir, model_id, confidence, vid_fps, width, height) -> List[Dict]:
    print(f"Cache-only mode: rebuilding from {cache_dir}...")
    pattern = re.compile(r"frame_(\d{6})\.json$")
    cached_files = []
    for name in os.listdir(cache_dir):
        m = pattern.match(name)
        if m:
            cached_files.append((int(m.group(1)), os.path.join(cache_dir, name)))

    cached_files.sort(key=lambda x: x[0])

    records: List[Dict] = []
    for idx, jp in cached_files:
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
    return records


def _run_inference_loop(cap, stride, model_id, confidence, max_frames, cache_dir, save_frame_json, vid_fps, width, height, client, total_frames) -> Tuple[List[Dict], int]:
    records: List[Dict] = []
    frame_idx = 0
    sampled_count = 0

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

        json_path = os.path.join(cache_dir, f"frame_{frame_idx:06d}.json")

        if save_frame_json and os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as jf:
                result = json.load(jf)
        else:
            print(f"Inferring frame {frame_idx}/{total_frames} with model {model_id}")
            result = client.infer_frame(frame, model_id=model_id, confidence=confidence)
            if save_frame_json:
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(result, jf, ensure_ascii=False)

        preds = result.get("predictions", []) if isinstance(result, dict) else []
        records.append({
            "frame": frame_idx,
            "time_sec": frame_idx / vid_fps if vid_fps else None,
            "image_size": {"w": width, "h": height},
            "model_id": model_id,
            "confidence": confidence,
            "predictions": preds,
        })

        sampled_count += 1
        frame_idx += 1

        if max_frames and sampled_count >= max_frames:
            break

    return records, sampled_count


def main():
    parser = argparse.ArgumentParser(description="Detect players using Roboflow with per-frame cache and JSONL output")
    parser.add_argument("--model-id", default=os.getenv("PLAYERS_MODEL_ID", "players-dataset-fusrb/1"))
    parser.add_argument("--confidence", type=float, default=settings.OVERLAY_MIN_CONF)
    parser.add_argument("--infer-fps", type=float, default=settings.PLAYERS_INFER_FPS)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true", help="Rebuild combined JSONL from existing cache; no API calls")
    args = parser.parse_args()

    infer_players_on_video(
        model_id=args.model_id,
        confidence=args.confidence,
        fps_sample=args.infer_fps,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        cache_dir=settings.PLAYERS_CACHE_DIR,
        combined_jsonl=settings.PLAYERS_DETECTIONS_JSONL,
        save_frame_json=settings.PLAYERS_SAVE_FRAME_JSON,
        cache_only=args.cache_only,
    )


if __name__ == "__main__":
    main()
