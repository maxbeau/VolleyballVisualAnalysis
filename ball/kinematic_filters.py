from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def filter_ball_tracks(
    tracks: Dict[int, Dict[str, Any]],
    *,
    min_confidence: float,
    max_speed_px_per_frame: float,
    max_accel_px_per_frame2: float,
    speed_reset_frame_gap: int,
    static_filter_enable: bool,
    static_window_frames: int,
    static_min_motion_px: float,
    continuity_filter_enable: bool,
    continuity_window_frames: int,
    continuity_max_error_px: float,
    continuity_error_growth_px: float,
) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """Apply image-plane kinematic filtering to per-frame ball detections."""
    frames = sorted(tracks.keys())
    if not frames:
        return {}, []

    speed_gate = float(max(0.0, max_speed_px_per_frame))
    accel_gate = float(max(0.0, max_accel_px_per_frame2))
    reset_gap = max(1, int(speed_reset_frame_gap or 1))
    static_enable = bool(static_filter_enable)
    static_window = max(1, int(static_window_frames or 1))
    static_motion = float(max(0.0, static_min_motion_px))

    continuity_enable = bool(continuity_filter_enable)
    continuity_window = max(2, int(continuity_window_frames or 2))
    continuity_err_px = float(max(0.0, continuity_max_error_px))
    continuity_err_growth_px = float(max(0.0, continuity_error_growth_px))
    continuity_history: deque[Tuple[int, float, float]] = deque(maxlen=continuity_window)
    continuity_gap_limit = max(reset_gap, continuity_window * 4)
    continuity_reset_gap = max(reset_gap + continuity_window, continuity_window * 2)

    filtered: Dict[int, Dict[str, Any]] = {}
    rejected_frames: List[int] = []

    last_frame: Optional[int] = None
    last_xy: Optional[Tuple[float, float]] = None
    last_speed: Optional[float] = None
    segment_frames: List[int] = []
    segment_positions: List[Tuple[float, float]] = []

    for frame in frames:
        prediction = tracks[frame]
        confidence = float(prediction.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        x_val = float(prediction.get("x", 0.0))
        y_val = float(prediction.get("y", 0.0))

        accept = True
        dt = None
        speed_val: Optional[float] = None

        gap = frame - last_frame if last_frame is not None else None
        if last_frame is not None and last_xy is not None:
            dt = frame - last_frame
            if dt <= 0:
                accept = False
            else:
                dist = math.hypot(x_val - last_xy[0], y_val - last_xy[1])
                speed_val = dist / dt if dt else 0.0

                if speed_gate > 0.0 and speed_val > speed_gate:
                    if dt >= reset_gap:
                        last_frame = None
                        last_xy = None
                        last_speed = None
                        segment_frames = []
                        segment_positions = []
                        dt = None
                        speed_val = None
                    else:
                        accept = False

                if accept and last_frame is not None and accel_gate > 0.0 and last_speed is not None and dt:
                    accel = abs(speed_val - last_speed) / dt if speed_val is not None else 0.0
                    if accel > accel_gate:
                        if dt >= reset_gap:
                            last_frame = None
                            last_xy = None
                            last_speed = None
                            segment_frames = []
                            segment_positions = []
                            dt = None
                            speed_val = None
                        else:
                            accept = False

        continuity_reject = False
        allowed_err = None
        dt_hist = None
        if (
            accept
            and continuity_enable
            and continuity_err_px > 0.0
            and len(continuity_history) >= 2
        ):
            last_hist_frame, last_hist_x, last_hist_y = continuity_history[-1]
            first_hist_frame, first_hist_x, first_hist_y = continuity_history[0]
            frames_span = last_hist_frame - first_hist_frame
            if frames_span > 0:
                vx = (last_hist_x - first_hist_x) / frames_span
                vy = (last_hist_y - first_hist_y) / frames_span
            else:
                vx = vy = 0.0
            dt_hist = frame - last_hist_frame if last_hist_frame is not None else None
            if dt_hist is not None and dt_hist >= continuity_reset_gap:
                continuity_history.clear()
            elif dt_hist is not None and dt_hist > 0 and dt_hist <= continuity_gap_limit:
                pred_x = last_hist_x + vx * dt_hist
                pred_y = last_hist_y + vy * dt_hist
                err = math.hypot(x_val - pred_x, y_val - pred_y)
                allowed_err = continuity_err_px + continuity_err_growth_px * max(0, dt_hist - 1)
                if err > allowed_err:
                    continuity_reject = True
            elif dt_hist is not None and dt_hist > continuity_gap_limit:
                continuity_history.clear()

        if continuity_reject and len(continuity_history) >= 3:
            frames_arr = np.array([hist[0] for hist in continuity_history], dtype=float)
            xs = np.array([hist[1] for hist in continuity_history], dtype=float)
            ys = np.array([hist[2] for hist in continuity_history], dtype=float)
            t0 = frames_arr[0]
            t_hist = frames_arr - t0
            t_pred = float(frame - t0)
            x_pred = last_hist_x
            y_pred = last_hist_y
            if np.ptp(t_hist) > 0.0:
                try:
                    coeff_x = np.polyfit(t_hist, xs, deg=1)
                    x_pred = np.polyval(coeff_x, t_pred)
                except Exception:
                    x_pred = last_hist_x
            if np.unique(t_hist).size >= 3:
                try:
                    coeff_y = np.polyfit(t_hist, ys, deg=2)
                    y_pred = np.polyval(coeff_y, t_pred)
                except Exception:
                    try:
                        coeff_y_lin = np.polyfit(t_hist, ys, deg=1)
                        y_pred = np.polyval(coeff_y_lin, t_pred)
                    except Exception:
                        y_pred = last_hist_y
            else:
                try:
                    coeff_y_lin = np.polyfit(t_hist, ys, deg=1)
                    y_pred = np.polyval(coeff_y_lin, t_pred)
                except Exception:
                    y_pred = last_hist_y

            err_poly = math.hypot(x_val - x_pred, y_val - y_pred)
            if allowed_err is None and dt_hist is not None and dt_hist > 0:
                allowed_err = continuity_err_px + continuity_err_growth_px * max(0, dt_hist - 1)
            if allowed_err is None:
                allowed_err = continuity_err_px
            if err_poly <= allowed_err * 1.1:
                continuity_reject = False

        if continuity_reject:
            accept = False

        if not accept:
            rejected_frames.append(frame)
            continue

        if gap is not None and gap >= reset_gap:
            segment_frames = []
            segment_positions = []
            continuity_history.clear()

        filtered[frame] = prediction

        if last_frame is not None and dt and speed_val is not None:
            last_speed = speed_val
        else:
            last_speed = None

        last_frame = frame
        last_xy = (x_val, y_val)

        segment_frames.append(frame)
        segment_positions.append((x_val, y_val))
        continuity_history.append((frame, x_val, y_val))

        if static_enable and len(segment_frames) >= static_window:
            sx, sy = segment_positions[0]
            disp = math.hypot(x_val - sx, y_val - sy)
            if disp <= static_motion:
                for rejected_frame in segment_frames:
                    filtered.pop(rejected_frame, None)
                    rejected_frames.append(rejected_frame)
                segment_frames = []
                segment_positions = []
                last_frame = None
                last_xy = None
                last_speed = None
                continuity_history.clear()

    if rejected_frames:
        print(
            "Kinematic filter removed frames:",
            ", ".join(str(idx) for idx in rejected_frames),
        )

    return filtered, rejected_frames
