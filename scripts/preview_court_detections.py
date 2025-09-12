import os
import sys
import json
import cv2
from typing import Dict, List, Tuple, Any

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from core.utils import ensure_dir
from court.utils import order_corners


Point = Tuple[float, float]


def load_court_detections(path: str) -> Dict[int, Dict[str, Any]]:
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
            out[fr] = {"corners": pts, "tpl_prec": rec.get("tpl_prec"), "confidence": (rec.get("pred") or {}).get("confidence")}
    return out


def draw_court(frame, corners: List[Point], color=(0, 255, 0), thickness=2):
    c = [(int(round(x)), int(round(y))) for x, y in corners]
    for i in range(4):
        p1 = c[i]
        p2 = c[(i + 1) % 4]
        cv2.line(frame, p1, p2, color, thickness)


def main():
    video_path = settings.VIDEO_PATH
    det_jsonl = settings.COURT_DETECTIONS_JSONL
    out_path = os.path.join("outputs", "court_detect_preview.mp4")
    ensure_dir(os.path.dirname(out_path) or ".")

    dets = load_court_detections(det_jsonl)
    if not dets:
        print(f"No detections found at {det_jsonl}. Run scripts/run_court_detect.py first or disable gating.")

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

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame.shape[1] != out_w) or (frame.shape[0] != out_h):
            frame = cv2.resize(frame, (out_w, out_h))

        rec = dets.get(i)
        if rec is not None:
            draw_court(frame, rec["corners"], color=color, thickness=thickness)
            # Put small text with stats
            txts = []
            if rec.get("confidence") is not None:
                try:
                    txts.append(f"conf={float(rec['confidence']):.2f}")
                except Exception:
                    pass
            if rec.get("tpl_prec") is not None:
                try:
                    txts.append(f"tpl={float(rec['tpl_prec']):.2f}")
                except Exception:
                    pass
            if txts:
                cv2.putText(frame, " / ".join(txts), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

        writer.write(frame)
        i += 1
        if i >= total_frames:
            break

    writer.release()
    cap.release()
    print(f"Court detection preview saved: {out_path}")


if __name__ == "__main__":
    main()

