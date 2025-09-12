import os
import sys
import json
import argparse
from typing import Dict, List, Any

import cv2

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from core.utils import ensure_dir
from visualization.hud import draw_hud


def load_players(jsonl_path: str) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    if not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            fr = int(rec.get("frame", -1))
            preds = rec.get("predictions")
            if fr < 0 or not isinstance(preds, list):
                continue
            out[fr] = preds
    return out


def to_tlbr_from_xywh(x: float, y: float, w: float, h: float):
    x1 = int(round(x - w / 2))
    y1 = int(round(y - h / 2))
    x2 = int(round(x + w / 2))
    y2 = int(round(y + h / 2))
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(description="Preview players detections overlay")
    parser.add_argument("--players-jsonl", default=settings.PLAYERS_DETECTIONS_JSONL)
    parser.add_argument("--out", default=settings.PLAYERS_OVERLAY_FULL)
    parser.add_argument("--min-conf", type=float, default=settings.OVERLAY_MIN_CONF)
    args = parser.parse_args()

    video_path = settings.VIDEO_PATH
    players_jsonl = args.players_jsonl
    out_path = args.out
    ensure_dir(os.path.dirname(out_path) or ".")

    ts = load_players(players_jsonl)
    if not ts:
        print(f"No players detections found at {players_jsonl}. Run scripts/run_players_detect.py first.")

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

    color = (0, 165, 255)  # orange for players
    thickness = 2

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame.shape[1] != out_w) or (frame.shape[0] != out_h):
            frame = cv2.resize(frame, (out_w, out_h))

        preds = ts.get(i)
        if preds:
            for p in preds:
                try:
                    conf = float(p.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                if conf < args.min_conf:
                    continue
                x = float(p.get("x", 0.0))
                y = float(p.get("y", 0.0))
                w = float(p.get("width", 0.0))
                h = float(p.get("height", 0.0))
                x1, y1, x2, y2 = to_tlbr_from_xywh(x, y, w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                if settings.SHOW_BOX_LABELS:
                    label = f"{p.get('class','player')} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # HUD overlay
        frame = draw_hud(frame, fps, i, total_frames)

        writer.write(frame)
        i += 1
        if i >= total_frames:
            break

    writer.release()
    cap.release()
    print(f"Players preview saved: {out_path}")


if __name__ == "__main__":
    main()

