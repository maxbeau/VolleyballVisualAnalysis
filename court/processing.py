import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import cv2

from core.config import settings
from core.utils import ensure_dir
from court.smoothing import smooth_corners_timeseries


def load_court_samples(jsonl_path: str) -> Dict[int, List[Tuple[float, float]]]:
    samples: Dict[int, List[Tuple[float, float]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frame_idx = int(rec.get("frame", -1))
            if frame_idx < 0:
                continue
            corners = rec.get("corners")
            if corners and isinstance(corners, list) and len(corners) >= 4:
                pts = []
                for p in corners[:4]:
                    pts.append((float(p[0]), float(p[1])))
                samples[frame_idx] = pts
    return samples


def process_court(
    detections_jsonl: str,
    tracking_jsonl: str,
    q_var: float,
    r_var: float,
    hold_ttl: int,
) -> None:
    video_path = settings.VIDEO_PATH
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    ensure_dir(os.path.dirname(tracking_jsonl) or ".")
    samples = load_court_samples(detections_jsonl)
    timeseries = smooth_corners_timeseries(samples, total_frames, q_var=q_var, r_var=r_var, hold_ttl=hold_ttl)
    with open(tracking_jsonl, "w", encoding="utf-8") as f:
        for k in sorted(timeseries.keys()):
            rec = {"frame": int(k), "corners": timeseries[k]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Court tracking saved: {tracking_jsonl} (frames: {len(timeseries)}/{total_frames})")


def main():
    detections_jsonl = settings.COURT_DETECTIONS_JSONL
    tracking_jsonl = settings.COURT_TRACKING_JSONL
    q_var = 400.0
    r_var = 36.0
    hold_ttl = 0

    parser = argparse.ArgumentParser(description="Process court detections into per-frame tracking (Kalman+RTS)")
    parser.add_argument("--detections-jsonl", default=detections_jsonl)
    parser.add_argument("--tracking-jsonl", default=tracking_jsonl)
    parser.add_argument("--q-var", type=float, default=q_var)
    parser.add_argument("--r-var", type=float, default=r_var)
    parser.add_argument("--hold-ttl", type=int, default=hold_ttl)
    args = parser.parse_args()

    process_court(
        detections_jsonl=args.detections_jsonl,
        tracking_jsonl=args.tracking_jsonl,
        q_var=args.q_var,
        r_var=args.r_var,
        hold_ttl=args.hold_ttl,
    )


if __name__ == "__main__":
    main()

