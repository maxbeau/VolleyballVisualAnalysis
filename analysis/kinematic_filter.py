import math
from typing import Dict, Any, List, Tuple, Optional, Set


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    x1, y1 = v1
    x2, y2 = v2
    n1 = math.hypot(x1, y1)
    n2 = math.hypot(x2, y2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cos = max(-1.0, min(1.0, (x1 * x2 + y1 * y2) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def filter_by_kinematics(
    best: Dict[int, Dict[str, Any]],
    fps: float,
    max_speed_px_per_s: float,
    max_accel_px_per_s2: float,
    max_dir_change_deg: float,
    max_size_change_frac_per_s: float,
    # Optional static-content filter
    static_filter_enable: bool = True,
    static_min_speed_px_per_s: float = 30.0,
    static_min_frames: int = 8,
    # Gate toggles
    enable_speed_gate: bool = True,
    enable_accel_gate: bool = True,
    enable_dir_gate: bool = True,
    enable_size_gate: bool = True,
    # Dynamic thresholding based on confidence
    dyn_enable: bool = True,
    dyn_min_mult: float = 0.7,
    dyn_max_mult: float = 1.5,
    # Warmup: disable size gate on certain frames (e.g., reseed frames)
    warmup_disable_size_frames: Optional[Set[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Lightweight kinematics gate on per-frame best detections in image space.
    Removes frames whose motion/size change is physically implausible for a volleyball.

    Args:
        best: frame_index -> prediction dict (expects x,y,width,height,confidence).
        fps: video frames per second.
        max_speed_px_per_s: hard cap for 2D pixel speed.
        max_accel_px_per_s2: hard cap for 2D pixel acceleration magnitude.
        max_dir_change_deg: maximum allowed direction change between consecutive velocity vectors.
        max_size_change_frac_per_s: maximum allowed fractional size change per second, applied to width and height separately.

    Returns:
        A new dict with obviously-invalid frames removed.
    """
    if not best or fps <= 0.0:
        return dict(best)

    frames = sorted(best.keys())
    if len(frames) < 2:
        return dict(best)

    # --- Static streak filter (based on consecutive raw frames) ---
    static_filtered = set()
    if static_filter_enable:
        from collections import deque
        dq = deque()
        prev_raw_f = frames[0]
        prev_raw = best[prev_raw_f]
        dq.append(prev_raw_f)
        for f in frames[1:]:
            cur = best[f]
            dt_raw = (f - prev_raw_f) / fps if fps > 0 else 0.0
            if dt_raw <= 0:
                dq.clear(); dq.append(f)
                prev_raw_f, prev_raw = f, cur
                continue
            dxr = float(cur.get("x", 0.0)) - float(prev_raw.get("x", 0.0))
            dyr = float(cur.get("y", 0.0)) - float(prev_raw.get("y", 0.0))
            v_raw = math.hypot(dxr, dyr) / dt_raw
            # Dynamic static threshold: higher when confidence is low, lower when high
            if dyn_enable:
                c_prev = max(0.0, min(1.0, float(prev_raw.get("confidence", 0.0))))
                c_cur = max(0.0, min(1.0, float(cur.get("confidence", 0.0))))
                c_use = 0.5 * (c_prev + c_cur)
                mult = dyn_min_mult + (dyn_max_mult - dyn_min_mult) * c_use
                static_thr = max(0.0, static_min_speed_px_per_s / max(1e-6, mult))
            else:
                static_thr = static_min_speed_px_per_s
            if v_raw < static_thr:
                dq.append(f)
                if len(dq) >= max(1, static_min_frames):
                    for t in list(dq):
                        static_filtered.add(t)
            else:
                dq.clear()
                dq.append(f)
            prev_raw_f, prev_raw = f, cur

    kept: Dict[int, Dict[str, Any]] = {}
    # Use last ACCEPTED (kept) frame as the reference. Rejects do not advance reference.
    last_keep_f = frames[0]
    last_keep = best[last_keep_f]
    kept[last_keep_f] = last_keep

    # Track last valid velocity for direction-change checks (based on kept frames)
    last_vx, last_vy = 0.0, 0.0
    have_last_v = False

    for f in frames[1:]:
        cur = best[f]
        # Static pre-filter
        if f in static_filtered:
            continue
        dt = (f - last_keep_f) / fps
        if dt <= 0:
            # pathological; skip
            last_keep_f, last_keep = f, cur
            kept[f] = cur
            continue

        x0, y0 = float(last_keep.get("x", 0.0)), float(last_keep.get("y", 0.0))
        x1, y1 = float(cur.get("x", 0.0)), float(cur.get("y", 0.0))

        # Compute dynamic multipliers from confidence
        if dyn_enable:
            c_prev = max(0.0, min(1.0, float(last_keep.get("confidence", 0.0))))
            c_cur = max(0.0, min(1.0, float(cur.get("confidence", 0.0))))
            c_use = 0.5 * (c_prev + c_cur)
            mult = dyn_min_mult + (dyn_max_mult - dyn_min_mult) * c_use
        else:
            mult = 1.0

        # Speed gate (pixels/sec)
        dx, dy = x1 - x0, y1 - y0
        v = math.hypot(dx, dy) / dt
        if enable_speed_gate and max_speed_px_per_s > 0 and v > (max_speed_px_per_s * mult) * 1.2:
            # Reject this frame (do not advance reference)
            continue

        # Direction change gate
        vx, vy = dx / dt, dy / dt
        if enable_dir_gate and have_last_v and max_dir_change_deg > 0:
            ang = _angle_between((last_vx, last_vy), (vx, vy))
            max_dir_eff = max(1.0, min(179.0, max_dir_change_deg * mult))
            if ang > max_dir_eff:
                continue

        # Size-change gate (fraction per second)
        w0, h0 = max(1e-6, float(last_keep.get("width", 1.0))), max(1e-6, float(last_keep.get("height", 1.0)))
        w1, h1 = max(1e-6, float(cur.get("width", 1.0))), max(1e-6, float(cur.get("height", 1.0)))
        frac_w = abs(w1 - w0) / (w0 * dt)
        frac_h = abs(h1 - h0) / (h0 * dt)
        max_size_eff = max_size_change_frac_per_s * mult if max_size_change_frac_per_s > 0 else 0.0
        if enable_size_gate and max_size_eff > 0:
            # Warmup exemption for size gate
            if not (warmup_disable_size_frames and f in warmup_disable_size_frames):
                if (frac_w > max_size_eff * 1.2) or (frac_h > max_size_eff * 1.2):
                    continue

        # Acceleration gate requires previous valid velocity
        if enable_accel_gate and have_last_v and max_accel_px_per_s2 > 0:
            ax = (vx - last_vx) / dt
            ay = (vy - last_vy) / dt
            a = math.hypot(ax, ay)
            if a > (max_accel_px_per_s2 * mult) * 1.2:
                continue

        # Accept
        kept[f] = cur
        last_keep_f, last_keep = f, cur
        last_vx, last_vy = vx, vy
        have_last_v = True

    return kept
