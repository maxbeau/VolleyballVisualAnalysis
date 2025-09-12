import os
import sys
import json
import cv2
import argparse
from typing import Dict, List, Tuple, Any

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from core.utils import ensure_dir
from court.utils import (
    order_corners,
    compute_homography,
    standard_court_model_size,
    apply_homography_points,
)
from visualization.mini_birdseye import MiniBirdseyeOverlay
from visualization.hud import draw_hud
from court.orientation import decide_orientation as decide_court_orientation


Point = Tuple[float, float]


def load_tracking(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            fr = int(rec.get("frame", -1))
            corners = rec.get("corners")
            if fr < 0 or not corners or len(corners) < 4:
                continue
            pts = order_corners([(float(corners[0][0]), float(corners[0][1])),
                                (float(corners[1][0]), float(corners[1][1])),
                                (float(corners[2][0]), float(corners[2][1])),
                                (float(corners[3][0]), float(corners[3][1]))])
            out[fr] = {"corners": pts, "info": rec.get("info", {})}
    return out


def draw_court(frame, corners: List[Point], color=(0, 255, 0), thickness=2):
    c = [(int(round(x)), int(round(y))) for x, y in corners]
    for i in range(4):
        p1 = c[i]
        p2 = c[(i + 1) % 4]
        cv2.line(frame, p1, p2, color, thickness)


## Use shared visualization helpers


def main():
    parser = argparse.ArgumentParser(description="Preview court tracking overlay")
    parser.add_argument("--tracking-jsonl", default=settings.COURT_TRACKING_JSONL)
    parser.add_argument("--out", default=os.path.join("outputs", "court_tracking_preview.mp4"))
    args = parser.parse_args()

    video_path = settings.VIDEO_PATH
    tracking_jsonl = args.tracking_jsonl
    out_path = args.out
    ensure_dir(os.path.dirname(out_path) or ".")

    ts = load_tracking(tracking_jsonl)
    if not ts:
        print(f"No tracking found at {tracking_jsonl}. Run scripts/run_court_processing.py first.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_w = width - (width % 2)
    out_h = height - (height % 2)
    fourcc = cv2.VideoWriter_fourcc(*settings.OVERLAY_CODEC)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        alt = os.path.splitext(out_path)[0] + ".avi"
        writer = cv2.VideoWriter(alt, fourcc, fps, (out_w, out_h))
        out_path = alt

    color = settings.COURT_COLOR
    thickness = settings.COURT_THICKNESS

    # Decide stable orientation (horizontal/vertical): prefer meta if available, else compute
    model_W, model_H = standard_court_model_size(100.0)
    orient = None
    # Try meta file first
    try:
        with open(settings.COURT_TRACKING_META, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
            orient = str(meta.get("orientation")) if isinstance(meta, dict) else None
    except Exception:
        orient = None
    if not orient or orient not in ("horizontal", "vertical"):
        orient = decide_court_orientation(cap, ts, (model_W, model_H), mode=settings.COURT_MINI_ORIENT_MODE)

    # Mini bird's-eye overlay helper
    tpl_colors = {"border": settings.COURT_COLOR, "center": settings.COURT_CENTER_COLOR, "attack": settings.COURT_ATTACK_COLOR}
    mini = MiniBirdseyeOverlay(
        colors=tpl_colors,
        thickness=thickness,
        placement=getattr(settings, "COURT_MINI_PLACEMENT", "top-right"),
        scale=getattr(settings, "COURT_MINI_SCALE", 0.24),
        margin=12,
        show_label=getattr(settings, "COURT_MINI_SHOW_LABEL", True),
        draw_poly=getattr(settings, "COURT_MINI_DRAW_POLY", True),
    )

    i = 0
    last = None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame.shape[1] != out_w) or (frame.shape[0] != out_h):
            frame = cv2.resize(frame, (out_w, out_h))

        rec = ts.get(i)
        if rec is None and last is not None:
            rec = last
        if rec is not None:
            draw_court(frame, rec["corners"], color=color, thickness=thickness)
            last = rec

        # HUD
        frame = draw_hud(frame, fps, i, total_frames)
        # Compose mini bird's-eye (toggle by .env)
        if not settings.COURT_MINI_ENABLE:
            writer.write(frame)
            i += 1
            if i >= total_frames:
                break
            continue
        # Render mini birdseye
        mini.render(frame, orient, rec["corners"] if rec is not None else None, (model_W, model_H))

        writer.write(frame)
        i += 1
        if i >= total_frames:
            break

    writer.release()
    cap.release()
    print(f"Court tracking preview saved: {out_path}")


if __name__ == "__main__":
    main()
