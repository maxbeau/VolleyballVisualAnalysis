import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

from core.config import settings
from core.utils import ensure_dir
from analysis.smoothing import kalman_rts_smooth
from court.smoothing import smooth_xy_timeseries
from court.utils import apply_homography_points


def parse_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def load_best_ball_per_frame(jsonl_path: str, allowed_classes: List[str]) -> Dict[int, Dict[str, Any]]:
    best: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frame_idx = int(rec.get("frame", -1))
            if frame_idx < 0:
                continue
            preds = rec.get("predictions", []) or []
            cand = None
            for p in preds:
                cls = p.get("class")
                if cls not in allowed_classes:
                    continue
                if cand is None or float(p.get("confidence", 0.0)) > float(cand.get("confidence", 0.0)):
                    cand = p
            if cand is not None:
                best[frame_idx] = cand
    return best


def soft_weight_aspect_ratio(best: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    f_min_ar = settings.FILTER_MIN_ASPECT_RATIO
    f_max_ar = settings.FILTER_MAX_ASPECT_RATIO
    ar_alpha = settings.FILTER_AR_SOFT_ALPHA
    adjusted: Dict[int, Dict[str, Any]] = {}
    for k, p in best.items():
        q = p.copy()
        try:
            w = float(q.get("width", 0.0))
            h = float(q.get("height", 0.0))
            ar = (w / h) if h > 0 else 0.0
            conf = float(q.get("confidence", 0.0))
            weight = 1.0
            if ar <= 0.0:
                weight = 0.5
            elif ar < f_min_ar:
                t = (f_min_ar - ar) / max(f_min_ar, 1e-6)
                weight = float(np.exp(-ar_alpha * t))
            elif ar > f_max_ar:
                t = (ar - f_max_ar) / max(f_max_ar, 1e-6)
                weight = float(np.exp(-ar_alpha * t))
            if weight < 1.0:
                q["confidence"] = max(0.0, min(1.0, conf * weight))
                q["_ar_weight"] = round(weight, 3)
        except Exception:
            pass
        adjusted[k] = q
    return adjusted


def pred_with_kalman_or_hold(
    frames: List[int],
    best: Dict[int, Dict[str, Any]],
    smoothed: Dict[int, Dict[str, Any]],
    i: int,
    hold_mode: str,
    hold_ttl: int,
) -> Optional[Dict[str, Any]]:
    import bisect
    if i in smoothed:
        return smoothed[i].copy()
    if not frames:
        return None
    pos = bisect.bisect_left(frames, i)
    prev_idx = frames[pos - 1] if pos > 0 else None
    next_idx = frames[pos] if pos < len(frames) else None
    hold_mode = (hold_mode or "prev").lower()
    if hold_mode not in ("prev", "next", "both", "none"):
        hold_mode = "prev"
    if hold_mode in ("prev", "both") and prev_idx is not None:
        if (i - prev_idx) <= hold_ttl:
            p = best[prev_idx].copy()
            p["_hold"] = True
            return p
    if hold_mode in ("next", "both") and next_idx is not None:
        if (next_idx - i) <= hold_ttl:
            p = best[next_idx].copy()
            p["_hold"] = True
            return p
    return None


def analyze(strategy_cli: Optional[str] = None, suffix_cli: str = ""):
    # IO and config
    detections_jsonl = settings.BALL_DETECTIONS_JSONL
    H_npy = "outputs/court_homography.npy"
    H_meta_json = "outputs/court_homography.json"
    base_jsonl = "outputs/trajectory_world.jsonl"
    base_csv = "outputs/trajectory_world.csv"
    base_img = "outputs/trajectory_birdseye.jpg"
    # Apply optional suffix to avoid overwrite
    suffix_env = ""
    suffix = suffix_cli or suffix_env or ""
    def with_suffix(path: str) -> str:
        if not suffix:
            return path
        root, ext = os.path.splitext(path)
        return f"{root}{suffix}{ext}"
    out_jsonl = with_suffix(base_jsonl)
    out_csv = with_suffix(base_csv)
    out_path_img = with_suffix(base_img)

    ensure_dir(os.path.dirname(out_jsonl) or ".")
    ensure_dir(os.path.dirname(out_csv) or ".")
    ensure_dir(os.path.dirname(out_path_img) or ".")

    # Load video/fps
    video_path = settings.VIDEO_PATH
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    # Load homography and meta
    if not os.path.exists(H_npy) or not os.path.exists(H_meta_json):
        raise FileNotFoundError("Court homography not found. Run court_homography.py first.")
    H = np.load(H_npy)
    with open(H_meta_json, "r", encoding="utf-8") as f:
        H_meta = json.load(f)
    dst_size = (int(H_meta.get("dst_size", {}).get("w", 1800)), int(H_meta.get("dst_size", {}).get("h", 900)))
    px_per_m = float(H_meta.get("scale_px_per_meter", 100.0))

    # Load detections and choose per-frame best
    allowed = ["ball", "volleyball"]
    best = load_best_ball_per_frame(detections_jsonl, allowed)
    best = soft_weight_aspect_ratio(best)
    frames_with_pred = sorted(best.keys())

    # Two-step only (image-space smoothing -> map -> optional light world smoothing)
    strategy = "two_step"
    min_conf = settings.OVERLAY_MIN_CONF

    # 1) Build raw observations (no image-space smoothing)
    img_xy_meas: Dict[int, Tuple[float, float]] = {}
    confs: Dict[int, float] = {}
    meas_point = "center"
    bottom_alpha = 1.0
    for f in frames_with_pred:
        p = best[f]
        c = float(p.get("confidence", 0.0))
        if c < min_conf:
            continue
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        w = float(p.get("width", 0.0))
        h = float(p.get("height", 0.0))
        if meas_point == "bottom":
            y = y + 0.5 * h * bottom_alpha
        elif meas_point == "top":
            y = y - 0.5 * h * bottom_alpha
        # center as default
        img_xy_meas[f] = (x, y)
        confs[f] = c

    # 2) Map to court plane
    world_meas: Dict[int, Dict[str, Any]] = {}
    if img_xy_meas:
        frames_sorted = sorted(img_xy_meas.keys())
        mapped = apply_homography_points([img_xy_meas[f] for f in frames_sorted], H)
        for f, (wx, wy) in zip(frames_sorted, mapped):
            # Attach image-domain context for position-aware noise
            p = best.get(f, {})
            world_meas[f] = {
                "x": float(wx),
                "y": float(wy),
                "confidence": float(confs.get(f, 1.0)),
                "img_x": float(p.get("x", img_xy_meas[f][0])),
                "img_y": float(p.get("y", img_xy_meas[f][1])),
                "img_h": float(img_h),
                "bbox_h": float(p.get("height", 0.0)),
            }

    if strategy == "two_step":
        # First: image-space smoothing (with gating/gravity/hold)
        max_gap_frames = settings.MAX_INTERP_GAP_FRAMES
        gate_chisq = settings.OBS_GATE_CHISQ_THRESH
        gate_use_conf = settings.OBS_GATE_USE_CONF
        gravity_pps2 = settings.GRAVITY_PPS2
        gravity_per_frame = (gravity_pps2 / (fps * fps)) if fps and fps > 0 else 0.0
        smoothed_img = kalman_rts_smooth(
            best,
            max_gap_frames,
            gate_chisq,
            gate_use_conf,
            gravity_per_frame=gravity_per_frame,
        )
        hold_mode = "prev"
        hold_ttl = settings.HOLD_TTL_FRAMES
        # Fill per-frame predictions via smoothed+hold
        img_xy_full: Dict[int, Tuple[float, float]] = {}
        flags: Dict[int, Dict[str, bool]] = {}
        for i in range(total_frames):
            p = pred_with_kalman_or_hold(frames_with_pred, best, smoothed_img, i, hold_mode, hold_ttl)
            if p is None or float(p.get("confidence", 0.0)) < min_conf:
                continue
            img_xy_full[i] = (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
            flags[i] = {"interp": bool(p.get("_interp", False)), "hold": bool(p.get("_hold", False))}

        # Map to world
        world_xy_meas: Dict[int, Tuple[float, float]] = {}
        if img_xy_full:
            frames_sorted_2 = sorted(img_xy_full.keys())
            mapped_2 = apply_homography_points([img_xy_full[f] for f in frames_sorted_2], H)
            for f, (wx, wy) in zip(frames_sorted_2, mapped_2):
                world_xy_meas[f] = (wx, wy)
        # Optional light smoothing in world plane for de-noising
        world_q_var = 200.0
        world_r_var = 16.0
        world_hold_ttl = 0
        world_xy = smooth_xy_timeseries(world_xy_meas, total_frames, q_var=world_q_var, r_var=world_r_var, hold_ttl=world_hold_ttl)

    # Derive kinematics in meters
    def px_to_m(v: float) -> float:
        return float(v) / px_per_m if px_per_m and px_per_m > 0 else float(v)

    frames_sorted = sorted(world_xy.keys())
    vx_mps: Dict[int, float] = {}
    vy_mps: Dict[int, float] = {}
    speed_mps: Dict[int, float] = {}
    dist_cum_m: Dict[int, float] = {}
    cum = 0.0
    for idx, f in enumerate(frames_sorted):
        x, y = world_xy[f]
        xm, ym = px_to_m(x), px_to_m(y)
        if idx > 0:
            f_prev = frames_sorted[idx - 1]
            xp, yp = world_xy[f_prev]
            dx_px = x - xp
            dy_px = y - yp
            dt = (f - f_prev) / fps if fps and fps > 0 else 1.0
            vx = px_to_m(dx_px) / max(dt, 1e-6)
            vy = px_to_m(dy_px) / max(dt, 1e-6)
            v = float(np.hypot(vx, vy))
            vx_mps[f] = vx
            vy_mps[f] = vy
            speed_mps[f] = v
            seg = float(np.hypot(px_to_m(dx_px), px_to_m(dy_px)))
            cum += seg
        dist_cum_m[f] = cum

    # Emit JSONL
    with open(out_jsonl, "w", encoding="utf-8") as jf:
        for f in frames_sorted:
            x, y = world_xy[f]
            rec = {
                "frame": int(f),
                "time_sec": (f / fps) if fps and fps > 0 else None,
                "world_px": [float(x), float(y)],
                "world_m": [px_to_m(x), px_to_m(y)],
                "vx_mps": vx_mps.get(f),
                "vy_mps": vy_mps.get(f),
                "speed_mps": speed_mps.get(f),
                "distance_m_cum": dist_cum_m.get(f, 0.0),
                "flags": flags.get(f, {}),
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Emit CSV
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["frame", "time_sec", "world_x_px", "world_y_px", "world_x_m", "world_y_m", "vx_mps", "vy_mps", "speed_mps", "distance_m_cum"])
        for f in frames_sorted:
            x, y = world_xy[f]
            w.writerow([
                int(f),
                (f / fps) if fps and fps > 0 else "",
                float(x),
                float(y),
                px_to_m(x),
                px_to_m(y),
                vx_mps.get(f, ""),
                vy_mps.get(f, ""),
                speed_mps.get(f, ""),
                dist_cum_m.get(f, 0.0),
            ])

    # Draw a bird's-eye path image
    bg = None
    # Prefer previously generated bird's-eye frame if present
    bird_path = "outputs/court_birdseye.jpg"
    if os.path.exists(bird_path):
        bg = cv2.imread(bird_path)
    if bg is None:
        bg = np.full((dst_size[1], dst_size[0], 3), 255, dtype=np.uint8)
    pts = [(int(round(world_xy[f][0])), int(round(world_xy[f][1]))) for f in frames_sorted]
    if len(pts) >= 2:
        cv2.polylines(bg, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)
        # Mark start/end
        cv2.circle(bg, pts[0], 5, (0, 200, 0), -1)
        cv2.circle(bg, pts[-1], 5, (0, 0, 200), -1)
    cv2.imwrite(out_path_img, bg)

    print(f"Trajectory JSONL: {out_jsonl}")
    print(f"Trajectory CSV:   {out_csv}")
    print(f"Bird's-eye path:  {out_path_img}")


def main():
    parser = argparse.ArgumentParser(description="Map ball detections to court coordinates and analyze trajectory (two-step)")
    parser.add_argument("--suffix", default="", help="Suffix to append to outputs (e.g., _exp)")
    parser.add_argument("--meas-point", choices=["center", "bottom", "top"], default=None, help="Which image point to map: center/bottom/top of box")
    parser.add_argument("--bottom-alpha", type=float, default=None, help="Scale for half-box offset when using bottom/top (default 1.0)")
    args = parser.parse_args()
    if args.meas_point is not None:
        os.environ["WORLD_MEAS_POINT"] = args.meas_point
    if args.bottom_alpha is not None:
        os.environ["WORLD_BOTTOM_ALPHA"] = str(args.bottom_alpha)
    analyze(strategy_cli=None, suffix_cli=args.suffix)


if __name__ == "__main__":
    main()
