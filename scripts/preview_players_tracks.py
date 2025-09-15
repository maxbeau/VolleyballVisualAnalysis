import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple

import cv2

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import settings
except Exception:  # fallback for older path
    from core.config import settings
from core.utils import ensure_dir
from visualization.hud import draw_hud
import bisect


def load_tracks(jsonl_path: str) -> Dict[int, List[Dict[str, Any]]]:
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
            trs = rec.get("tracks")
            if fr < 0 or not isinstance(trs, list):
                continue
            out[fr] = trs
    return out


def to_tlbr_from_xywh(x: float, y: float, w: float, h: float):
    x1 = int(round(x - w / 2))
    y1 = int(round(y - h / 2))
    x2 = int(round(x + w / 2))
    y2 = int(round(y + h / 2))
    return x1, y1, x2, y2


def build_id_index(ts: Dict[int, List[Dict[str, Any]]]):
    id_frames: Dict[int, List[int]] = {}
    id_map: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for fr, lst in ts.items():
        for t in lst:
            try:
                tid = int(t.get("id"))
            except Exception:
                continue
            id_map.setdefault(tid, {})[fr] = t
    for tid, fmap in id_map.items():
        frames = sorted(fmap.keys())
        id_frames[tid] = frames
    return id_frames, id_map


def interp_box(a: Dict[str, Any], b: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    def _f(k):
        return float(a.get(k, 0.0)) * (1 - alpha) + float(b.get(k, 0.0)) * alpha
    return {
        "id": int(a.get("id", b.get("id", -1))),
        "x": _f("x"),
        "y": _f("y"),
        "width": _f("width"),
        "height": _f("height"),
        "confidence": float(a.get("confidence", 0.0)) * (1 - alpha) + float(b.get("confidence", 0.0)) * alpha,
        "_interp": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Preview players tracks overlay")
    parser.add_argument("--tracks-jsonl", default=settings.players.TRACKS_JSONL)
    parser.add_argument("--out", default=os.path.join("outputs", "players_tracks_preview.mp4"))
    parser.add_argument("--min-conf", type=float, default=settings.common.OVERLAY_MIN_CONF)
    args = parser.parse_args()

    video_path = settings.common.VIDEO_PATH
    tracks_jsonl = args.tracks_jsonl
    out_path = args.out
    ensure_dir(os.path.dirname(out_path) or ".")

    ts = load_tracks(tracks_jsonl)
    if not ts:
        print(f"No players tracks found at {tracks_jsonl}. Run scripts/run_players_track.py first.")
    # Build ID index for interpolation/hold
    id_frames, id_map = build_id_index(ts)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_w = width - (width % 2)
    out_h = height - (height % 2)
    fourcc = cv2.VideoWriter_fourcc(*settings.common.OVERLAY_CODEC)
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

        drawn_ids = set()
        trs = ts.get(i, [])
        # Draw current detections first
        for t in trs:
            try:
                conf = float(t.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            if conf < args.min_conf:
                continue
            _id = int(t.get("id", -1))
            drawn_ids.add(_id)
            x = float(t.get("x", 0.0))
            y = float(t.get("y", 0.0))
            w = float(t.get("width", 0.0))
            h = float(t.get("height", 0.0))
            x1, y1, x2, y2 = to_tlbr_from_xywh(x, y, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"P{_id}"
            if settings.common.SHOW_BOX_LABELS:
                label = f"P{_id} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # (Kalman debug drawing removed)

        # Interp/hold for missing IDs
        interp_enable = bool(getattr(settings, 'PLAYERS_INTERP_ENABLE', True))
        interp_max_gap = int(getattr(settings, 'PLAYERS_INTERP_MAX_GAP', 6))
        hold_ttl = int(getattr(settings, 'PLAYERS_HOLD_TTL_FRAMES', 8))
        for tid, frames_list in id_frames.items():
            if tid in drawn_ids:
                continue
            if not frames_list:
                continue
            pos = bisect.bisect_left(frames_list, i)
            prev_fr = frames_list[pos - 1] if pos > 0 else None
            next_fr = frames_list[pos] if pos < len(frames_list) else None
            drawn = False
            # Try interpolation if within gap
            if interp_enable and prev_fr is not None and next_fr is not None and prev_fr < i < next_fr:
                gap = next_fr - prev_fr
                if gap <= max(1, interp_max_gap):
                    a = id_map[tid][prev_fr]
                    b = id_map[tid][next_fr]
                    alpha = (i - prev_fr) / float(gap)
                    t = interp_box(a, b, alpha)
                    if float(t.get("confidence", 0.0)) >= args.min_conf:
                        x1, y1, x2, y2 = to_tlbr_from_xywh(t["x"], t["y"], t["width"], t["height"])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        label = f"P{tid}"
                        if settings.common.SHOW_BOX_LABELS:
                            label += " *"  # mark interpolated
                        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        drawn = True
            # Else try hold from prev within TTL
            if not drawn and prev_fr is not None and (i - prev_fr) <= max(1, hold_ttl):
                t = id_map[tid][prev_fr]
                if float(t.get("confidence", 0.0)) >= args.min_conf:
                    x1, y1, x2, y2 = to_tlbr_from_xywh(float(t.get("x", 0.0)), float(t.get("y", 0.0)), float(t.get("width", 0.0)), float(t.get("height", 0.0)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"P{tid}"
                    if settings.common.SHOW_BOX_LABELS:
                        label += " (hold)"
                    cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        frame = draw_hud(frame, fps, i, total_frames)

        writer.write(frame)
        i += 1
        if i >= total_frames:
            break

    writer.release()
    cap.release()
    print(f"Players tracks preview saved: {out_path}")


if __name__ == "__main__":
    main()
