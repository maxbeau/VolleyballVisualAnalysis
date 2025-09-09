import os
import json
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2

from utils import load_env_file, pick_video_path, ensure_dir
from court_utils import apply_homography_points


def load_traj_world(jsonl_path: str) -> Dict[int, Tuple[float, float]]:
    data: Dict[int, Tuple[float, float]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fi = int(rec.get("frame", -1))
            wp = rec.get("world_px") or rec.get("world")
            if fi >= 0 and wp and isinstance(wp, list) and len(wp) >= 2:
                data[fi] = (float(wp[0]), float(wp[1]))
    return data


def main():
    env = load_env_file()

    ap = argparse.ArgumentParser(description="Overlay two world trajectories back onto the original video")
    ap.add_argument("--traj-a", default=os.path.join("outputs", "trajectory_world_world.jsonl"), help="First world trajectory JSONL")
    ap.add_argument("--traj-b", default=os.path.join("outputs", "trajectory_world_two.jsonl"), help="Second world trajectory JSONL")
    ap.add_argument("--label-a", default="world", help="Label for trajectory A")
    ap.add_argument("--label-b", default="two_step", help="Label for trajectory B")
    ap.add_argument("--out", default=os.path.join("outputs", "trajectory_compare_overlay.mp4"), help="Output video path")
    ap.add_argument("--tail", type=int, default=int(env.get("TRAJ_TAIL_FRAMES", 90)), help="Tail length in frames for polyline")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--color-a", default=env.get("TRAJ_COLOR_A", "0,0,255"))
    ap.add_argument("--color-b", default=env.get("TRAJ_COLOR_B", "0,200,0"))
    ap.add_argument("--alpha", type=float, default=float(env.get("TRAJ_ALPHA", 1.0)))
    ap.add_argument("--detections-jsonl", default=os.path.join("outputs", "ball_detections.jsonl"), help="Optional detections JSONL to draw current box")
    ap.add_argument("--classes", default=(env.get("BALL_CLASSES", "ball,volleyball")), help="Comma-separated classes for picking best box")
    args = ap.parse_args()

    def parse_color(s: str, default=(0, 0, 255)):
        try:
            parts = [int(p.strip()) for p in s.split(",")]
            if len(parts) == 3:
                return (parts[0], parts[1], parts[2])
        except Exception:
            pass
        return default

    color_a = parse_color(args.color_a, (0, 0, 255))
    color_b = parse_color(args.color_b, (0, 200, 0))

    # Load trajectories
    traj_a = load_traj_world(args.traj_a)
    traj_b = load_traj_world(args.traj_b)

    # Optional detections for drawing boxes
    best_box: Dict[int, Tuple[float, float, float, float]] = {}
    if os.path.exists(args.detections_jsonl):
        allowed = [c.strip() for c in (args.classes or "ball,volleyball").split(",")]
        with open(args.detections_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fi = int(rec.get("frame", -1))
                preds = rec.get("predictions", []) or []
                cand = None
                for p in preds:
                    if p.get("class") not in allowed:
                        continue
                    if cand is None or float(p.get("confidence", 0.0)) > float(cand.get("confidence", 0.0)):
                        cand = p
                if cand is not None:
                    x = float(cand.get("x", 0.0))
                    y = float(cand.get("y", 0.0))
                    w = float(cand.get("width", 0.0))
                    h = float(cand.get("height", 0.0))
                    best_box[fi] = (x, y, w, h)

    # Load video
    video_path, _ = pick_video_path(env)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Load homography and invert
    H_npy = env.get("COURT_H_NPY", "outputs/court_homography.npy")
    if not os.path.exists(H_npy):
        raise FileNotFoundError(f"Homography file not found: {H_npy}")
    H = np.load(H_npy)
    H_inv = np.linalg.inv(H)

    # Prepare writer
    out_w = width - (width % 2)
    out_h = height - (height % 2)
    ensure_dir(os.path.dirname(args.out) or ".")
    fourcc = cv2.VideoWriter_fourcc(*env.get("OVERLAY_CODEC", "avc1"))
    writer = cv2.VideoWriter(args.out, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        alt = os.path.splitext(args.out)[0] + ".avi"
        writer = cv2.VideoWriter(alt, fourcc, fps, (out_w, out_h))
        args.out = alt

    tail = max(1, int(args.tail))
    thick = max(1, int(args.thickness))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Optional resize to match writer size
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Build tail for A and B in world plane
        def tail_points(traj: Dict[int, Tuple[float, float]], f: int) -> List[Tuple[float, float]]:
            pts: List[Tuple[float, float]] = []
            start = max(0, f - tail + 1)
            for j in range(start, f + 1):
                if j in traj:
                    pts.append(traj[j])
            return pts

        ta_w = tail_points(traj_a, i)
        tb_w = tail_points(traj_b, i)

        # Map world->image using inverse homography
        if len(ta_w) >= 2:
            pts_img = apply_homography_points(ta_w, H_inv)
            pa = [(int(round(x)), int(round(y))) for x, y in pts_img]
            cv2.polylines(frame, [np.array(pa, dtype=np.int32)], isClosed=False, color=color_a, thickness=thick)
            cv2.circle(frame, pa[-1], max(3, thick + 1), color_a, -1)
        if len(tb_w) >= 2:
            pts_img = apply_homography_points(tb_w, H_inv)
            pb = [(int(round(x)), int(round(y))) for x, y in pts_img]
            cv2.polylines(frame, [np.array(pb, dtype=np.int32)], isClosed=False, color=color_b, thickness=thick)
            cv2.circle(frame, pb[-1], max(3, thick + 1), color_b, -1)

        # Draw current best detection box for reference
        if i in best_box:
            x, y, w, h = best_box[i]
            x1 = int(round(x - w / 2))
            y1 = int(round(y - h / 2))
            x2 = int(round(x + w / 2))
            y2 = int(round(y + h / 2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Labels
        cv2.putText(frame, args.label_a, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_a, 2, cv2.LINE_AA)
        cv2.putText(frame, args.label_b, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_b, 2, cv2.LINE_AA)

        writer.write(frame)
        i += 1

    writer.release()
    cap.release()
    print(f"Saved trajectory overlay video: {args.out}")


if __name__ == "__main__":
    main()
