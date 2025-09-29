import os
import json
from typing import Dict, List, Any, Optional

import cv2

from core.utils import ensure_dir
from players.tracker import TrackerConfig, OCSortTracker
from players.court_constraints import build_court_constraint, CourtConstraint
from players.detection_filters import refine_detections


def load_detections(jsonl_path: str) -> Dict[int, List[Dict[str, Any]]]:
    ts: Dict[int, List[Dict[str, Any]]] = {}
    if not os.path.exists(jsonl_path):
        return ts
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            i = int(rec.get("frame", -1))
            preds = rec.get("predictions")
            if i >= 0 and isinstance(preds, list):
                ts[i] = preds
    return ts


def run_player_tracking(
    *,
    video_path: str,
    detections_jsonl: str,
    tracks_jsonl: str,
    cfg: TrackerConfig,
    court_tracking_jsonl: Optional[str] = None,
    max_frames: int = 0,
):
    """Runs OC-SORT player tracking on a video and detections."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    dets = load_detections(detections_jsonl)
    court_constraint = build_court_constraint(court_tracking_jsonl) if court_tracking_jsonl else None

    ensure_dir(os.path.dirname(tracks_jsonl) or ".")

    tracker = OCSortTracker(cfg)

    total_limit = total
    if max_frames > 0:
        total_limit = min(total_limit, max_frames)

    with open(tracks_jsonl, "w", encoding="utf-8") as out_f:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            preds = dets.get(i, [])
            if court_constraint is not None:
                preds = court_constraint.filter_predictions(preds)

            tracks = tracker.update(frame, i, preds)
            row = {
                "frame": i,
                "tracks": tracks,
                "image_size": {"w": width, "h": height},
                "fps": fps,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            i += 1
            if i >= total_limit:
                break

    cap.release()
    print(f"Players tracking saved: {tracks_jsonl}")
