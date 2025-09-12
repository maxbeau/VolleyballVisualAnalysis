import os
import json
import argparse
from typing import Dict, List, Any

import cv2

from core.config import settings
from core.utils import ensure_dir
from players.tracker import TrackerConfig, ByteTrackReID
from players.ocr import JerseyOCR
from players.reid_embedder import build_reid_embedder


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


def main():
    parser = argparse.ArgumentParser(description="Track players with ByteTrack-style tracker + simple ReID")
    parser.add_argument("--players-jsonl", default=settings.PLAYERS_DETECTIONS_JSONL)
    parser.add_argument("--out", default=settings.PLAYERS_TRACKS_JSONL)
    parser.add_argument("--track-thresh", type=float, default=settings.PLAYERS_TRACK_THRESH)
    parser.add_argument("--match-iou", type=float, default=settings.PLAYERS_MATCH_IOU)
    parser.add_argument("--low-track-thresh", type=float, default=settings.PLAYERS_LOW_TRACK_THRESH)
    parser.add_argument("--reid-weight", type=float, default=settings.PLAYERS_REID_WEIGHT)
    parser.add_argument("--max-age", type=int, default=settings.PLAYERS_MAX_AGE)
    parser.add_argument("--min-hits", type=int, default=settings.PLAYERS_MIN_HITS)
    args = parser.parse_args()

    vid_path = settings.VIDEO_PATH
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    dets = load_detections(args.players_jsonl)

    ensure_dir(os.path.dirname(args.out) or ".")
    cfg = TrackerConfig(
        track_thresh=args.track_thresh,
        low_track_thresh=args.low_track_thresh,
        match_iou_thresh=args.match_iou,
        reid_weight=args.reid_weight,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )
    # Apply OCR settings
    cfg.ocr_enable = bool(getattr(settings, "PLAYERS_OCR_ENABLE", False))
    cfg.ocr_min_conf = float(getattr(settings, "PLAYERS_OCR_MIN_CONF", 0.5))

    embedder = build_reid_embedder(settings)
    # Jersey OCR (optional)
    jersey_fn = None
    if cfg.ocr_enable:
        try:
            jersey_reader = JerseyOCR()
            jersey_fn = jersey_reader.infer_digits
        except Exception as e:
            print(f"Jersey OCR unavailable: {e}")
            jersey_fn = None
    tracker = ByteTrackReID(cfg, embedder=embedder, jersey_ocr=jersey_fn)

    with open(args.out, "w", encoding="utf-8") as out_f:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            preds = dets.get(i, [])
            tracks = tracker.update(frame, i, preds)
            row = {
                "frame": i,
                "tracks": tracks,
                "image_size": {"w": width, "h": height},
                "fps": fps,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            i += 1
            if i >= total:
                break

    cap.release()
    print(f"Players tracking saved: {args.out}")


if __name__ == "__main__":
    main()
