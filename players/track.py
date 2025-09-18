import os
import json
import argparse
from typing import Dict, List, Any

import cv2

try:
    # Prefer unified config path
    from config import settings
except Exception:  # fallback for older import path
    from core.config import settings
from core.utils import ensure_dir
from players.tracker import TrackerConfig, ByteTrackReID
 
from players.reid_embedder import build_reid_embedder
from players.court_constraints import build_court_constraint
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


def main():
    parser = argparse.ArgumentParser(description="Track players with ByteTrack-style tracker + simple ReID")
    parser.add_argument("--players-jsonl", default=settings.players.DETECTIONS_JSONL)
    parser.add_argument("--out", default=settings.players.TRACKS_JSONL)
    parser.add_argument("--track-thresh", type=float, default=settings.players.TRACK_THRESH)
    parser.add_argument("--match-iou", type=float, default=settings.players.MATCH_IOU)
    parser.add_argument("--low-track-thresh", type=float, default=settings.players.LOW_TRACK_THRESH)
    parser.add_argument("--reid-weight", type=float, default=settings.players.REID_WEIGHT)
    parser.add_argument("--max-age", type=int, default=settings.players.MAX_AGE)
    parser.add_argument("--min-hits", type=int, default=settings.players.MIN_HITS)
    parser.add_argument("--max-frames", type=int, default=0, help="Process at most this many frames; 0 for all")
    # Advanced tracking params (optional but exposed for tuning)
    parser.add_argument("--id-lock-age", type=int, default=settings.players.ID_LOCK_AGE, help="Frames to lock ID after hit (anti-switch)")
    parser.add_argument("--switch-min-sim", type=float, default=settings.players.SWITCH_MIN_SIM, help="Min ReID sim to switch when locked")
    parser.add_argument("--reid-min-sim", type=float, default=settings.players.REID_MIN_SIM, help="Min similarity for appearance-only associations")
    parser.add_argument("--size-change-max", type=float, default=settings.players.SIZE_CHANGE_MAX_RATIO, help="Max size ratio allowed between frames")
    parser.add_argument("--reid-expand", type=float, default=settings.players.REID_EXPAND_RATIO, help="BBox expansion ratio for ReID crops")
    parser.add_argument("--reid-focus-top", type=float, default=settings.players.REID_FOCUS_TOP, help="Fraction of box height to keep for torso focus")
    parser.add_argument("--reid-min-crop-px", type=int, default=settings.players.REID_MIN_CROP_PX, help="Minimum crop height in pixels for ReID")
    parser.add_argument("--reid-profile-new", type=float, default=settings.players.REID_PROFILE_NEW_THRESH, help="Similarity threshold below which to spawn a new appearance profile")
    parser.add_argument("--reid-profile-merge", type=float, default=settings.players.REID_PROFILE_MERGE_THRESH, help="Similarity threshold to merge into existing profile")
    parser.add_argument("--reid-profile-beta", type=float, default=settings.players.REID_PROFILE_BETA, help="Blend factor for updating appearance profiles")
    parser.add_argument("--reid-profile-max", type=int, default=settings.players.REID_PROFILE_MAX, help="Maximum number of appearance profiles to keep")
    parser.add_argument("--reid-profile-ttl", type=int, default=settings.players.REID_PROFILE_TTL, help="Profile time-to-live in frames without updates")
    # Kalman removed for players tracking (no CLI args)
    args = parser.parse_args()

    vid_path = settings.common.VIDEO_PATH
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vid_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    dets = load_detections(args.players_jsonl)
    court_constraint = build_court_constraint(settings)

    ensure_dir(os.path.dirname(args.out) or ".")
    cfg = TrackerConfig(
        track_thresh=args.track_thresh,
        low_track_thresh=args.low_track_thresh,
        match_iou_thresh=args.match_iou,
        reid_weight=args.reid_weight,
        max_age=args.max_age,
        min_hits=args.min_hits,
    )
    # Optional advanced params
    cfg.id_lock_age = max(0, int(args.id_lock_age))
    cfg.switch_min_sim = float(args.switch_min_sim)
    cfg.reid_min_sim = float(args.reid_min_sim)
    cfg.size_change_max_ratio = max(1.0, float(args.size_change_max))
    cfg.reid_expand_ratio = max(0.0, float(args.reid_expand))
    cfg.reid_focus_top = float(args.reid_focus_top)
    cfg.reid_min_crop_px = max(4, int(args.reid_min_crop_px))
    cfg.reid_profile_new_thresh = float(args.reid_profile_new)
    cfg.reid_profile_merge_thresh = float(args.reid_profile_merge)
    cfg.reid_profile_beta = float(args.reid_profile_beta)
    cfg.reid_profile_max = max(1, int(args.reid_profile_max))
    cfg.reid_profile_ttl = max(0, int(args.reid_profile_ttl))

    # Adapt max_age by video FPS (~0.8s tolerance)
    adapt_age = max(cfg.min_hits, int(round(fps * 0.8)))
    if args.max_age == settings.players.MAX_AGE:
        cfg.max_age = max(cfg.max_age, adapt_age)
    # No Kalman-related config assignment
    # OCR removed: no OCR-related settings

    embedder = build_reid_embedder(settings)
    # No OCR integration
    tracker = ByteTrackReID(cfg, embedder=embedder)

    total_limit = total
    if int(args.max_frames) > 0:
        total_limit = min(total_limit, int(args.max_frames))
    with open(args.out, "w", encoding="utf-8") as out_f:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            preds = dets.get(i, [])
            if court_constraint is not None:
                preds = court_constraint.filter_predictions(preds)
            preds = refine_detections(preds, frame.shape[:2], settings)
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
    print(f"Players tracking saved: {args.out}")


if __name__ == "__main__":
    main()
