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


def load_actions(jsonl_path: str) -> Dict[int, List[Dict[str, Any]]]:
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


def has_box(p: Dict[str, Any]) -> bool:
    return all(k in p for k in ("x", "y", "width", "height"))


def _color_from_class(cls: str) -> tuple:
    """Return a BGR color for a given action class.

    Known volleyball actions are assigned fixed colors for consistency.
    Unknown classes get a deterministic color derived from the class name.
    """
    cls_l = (cls or "").strip().lower()
    palette = {
        # BGR colors (OpenCV)
        "serve": (0, 165, 255),      # orange
        "pass": (0, 255, 255),       # yellow
        "bump": (0, 255, 255),       # yellow (alias of pass)
        "set": (255, 0, 255),        # magenta
        "spike": (0, 255, 0),        # green
        "attack": (0, 255, 0),       # green (alias)
        "block": (255, 0, 0),        # blue
        "dig": (255, 255, 0),        # cyan
        "celebrate": (180, 105, 255),# pink-ish
        "idle": (200, 200, 200),     # gray
        "standby": (200, 200, 200),  # gray (alias)
    }
    if cls_l in palette:
        return palette[cls_l]

    # Deterministic fallback: hash the class name to a hue and convert to BGR
    import hashlib
    h = int(hashlib.md5(cls_l.encode("utf-8")).hexdigest(), 16) % 360
    # Convert HSV -> BGR
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.8, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def main():
    parser = argparse.ArgumentParser(description="Preview actions detections overlay")
    parser.add_argument("--actions-jsonl", default=settings.ACTIONS_DETECTIONS_JSONL)
    parser.add_argument("--out", default=settings.ACTIONS_OVERLAY_FULL)
    parser.add_argument("--min-conf", type=float, default=settings.OVERLAY_MIN_CONF)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    video_path = settings.VIDEO_PATH
    actions_jsonl = args.actions_jsonl
    out_path = args.out
    ensure_dir(os.path.dirname(out_path) or ".")

    ts = load_actions(actions_jsonl)
    if not ts:
        print(f"No actions detections found at {actions_jsonl}. Run scripts/run_actions_detect.py first.")

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

    thickness = 2

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame.shape[1] != out_w) or (frame.shape[0] != out_h):
            frame = cv2.resize(frame, (out_w, out_h))

        preds = ts.get(i)
        drew_box = False
        if preds:
            # Draw detection boxes if present
            for p in preds:
                try:
                    conf = float(p.get("confidence", 0.0))
                except Exception:
                    conf = 0.0
                if conf < args.min_conf:
                    continue
                cls = str(p.get("class", "action"))
                color = _color_from_class(cls)
                if has_box(p):
                    x = float(p.get("x", 0.0))
                    y = float(p.get("y", 0.0))
                    w = float(p.get("width", 0.0))
                    h = float(p.get("height", 0.0))
                    x1, y1, x2, y2 = to_tlbr_from_xywh(x, y, w, h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    drew_box = True
                    if settings.SHOW_BOX_LABELS:
                        label = f"{cls} {conf:.2f}"
                        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

            # If no boxes, show top-k class labels in top-left
            if not drew_box:
                # sort by confidence desc
                try:
                    preds_sorted = sorted(preds, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
                except Exception:
                    preds_sorted = preds
                y0 = 30
                for j, p in enumerate(preds_sorted[: max(1, int(args.topk))]):
                    cls = str(p.get("class", "action"))
                    color = _color_from_class(cls)
                    try:
                        conf = float(p.get("confidence", 0.0))
                    except Exception:
                        conf = 0.0
                    txt = f"{cls}: {conf:.2f}"
                    cv2.putText(frame, txt, (15, y0 + j * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # HUD overlay
        frame = draw_hud(frame, fps, i, total_frames)

        writer.write(frame)
        i += 1
        if i >= total_frames:
            break

    writer.release()
    cap.release()
    print(f"Actions preview saved: {out_path}")


if __name__ == "__main__":
    main()
