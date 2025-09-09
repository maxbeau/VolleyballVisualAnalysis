import os
import json
import argparse
from typing import List, Tuple

import numpy as np
import cv2


def load_world_xy(jsonl_path: str) -> List[Tuple[int, float, float]]:
    frames = []
    xs = []
    ys = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fidx = int(rec.get("frame", -1))
            wp = rec.get("world_px") or rec.get("world")
            if fidx < 0 or not wp or len(wp) < 2:
                continue
            frames.append(fidx)
            xs.append(float(wp[0]))
            ys.append(float(wp[1]))
    return list(zip(frames, xs, ys))


def draw_paths(bg, path_a: List[Tuple[int, float, float]], path_b: List[Tuple[int, float, float]]):
    if bg is None:
        # Determine canvas size from ranges
        allx = [x for _, x, _ in path_a] + [x for _, x, _ in path_b]
        ally = [y for _, _, y in path_a] + [y for _, _, y in path_b]
        W = int(max(100, (max(allx) - min(allx) + 50)))
        H = int(max(100, (max(ally) - min(ally) + 50)))
        bg = np.full((H, W, 3), 255, dtype=np.uint8)
    pa = [(int(round(x)), int(round(y))) for _, x, y in path_a]
    pb = [(int(round(x)), int(round(y))) for _, x, y in path_b]
    if len(pa) >= 2:
        cv2.polylines(bg, [np.array(pa, dtype=np.int32)], False, (0, 0, 255), 2)
        cv2.circle(bg, pa[0], 4, (0, 200, 0), -1)
        cv2.circle(bg, pa[-1], 4, (0, 0, 200), -1)
    if len(pb) >= 2:
        cv2.polylines(bg, [np.array(pb, dtype=np.int32)], False, (0, 180, 0), 2)
        cv2.circle(bg, pb[0], 4, (0, 200, 0), -1)
        cv2.circle(bg, pb[-1], 4, (0, 0, 200), -1)
    return bg


def main():
    ap = argparse.ArgumentParser(description="Overlay two world trajectories for visual comparison")
    ap.add_argument("jsonl_a", help="First trajectory JSONL (e.g., _world)")
    ap.add_argument("jsonl_b", help="Second trajectory JSONL (e.g., _two)")
    ap.add_argument("--birdseye", default="outputs/court_birdseye.jpg", help="Background bird's-eye image")
    ap.add_argument("--out", default="outputs/trajectory_compare.jpg", help="Output comparison image")
    args = ap.parse_args()

    path_a = load_world_xy(args.jsonl_a)
    path_b = load_world_xy(args.jsonl_b)
    bg = None
    if os.path.exists(args.birdseye):
        bg = cv2.imread(args.birdseye)
    img = draw_paths(bg, path_a, path_b)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"Saved comparison: {args.out}")


if __name__ == "__main__":
    main()

