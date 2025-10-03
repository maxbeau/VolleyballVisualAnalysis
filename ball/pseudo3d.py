from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from court.utils import apply_homography_points


class GroundSizeModel:
    """Predict expected image-plane ball size at ground contact."""

    def __init__(self, coeffs: Optional[np.ndarray], default_size: float) -> None:
        self.coeffs = coeffs
        self.default_size = float(default_size) if math.isfinite(default_size) else 24.0

    def predict(self, x_norm: float, y_norm: float) -> float:
        base = self.default_size
        if self.coeffs is not None and self.coeffs.size == 3:
            value = float(
                self.coeffs[0]
                + self.coeffs[1] * x_norm
                + self.coeffs[2] * y_norm
            )
            if math.isfinite(value):
                base = value
        return max(4.0, base)


def build_ground_size_model(
    samples: List[Tuple[float, float, float, float]],
    min_samples: int,
) -> GroundSizeModel:
    """Fit a weighted planar model mapping image position to ground-contact size."""
    if not samples:
        return GroundSizeModel(None, 24.0)

    arr_sizes = np.array([sample[2] for sample in samples], dtype=float)
    default_size = float(np.median(arr_sizes)) if arr_sizes.size else 24.0

    if len(samples) < max(3, int(min_samples)):
        return GroundSizeModel(None, default_size)

    matrix_a = np.array([[1.0, sample[0], sample[1]] for sample in samples], dtype=float)
    vector_b = np.array([sample[2] for sample in samples], dtype=float)
    weights = np.array([max(1e-3, sample[3]) for sample in samples], dtype=float)
    sqrt_weights = np.sqrt(weights)
    matrix_aw = matrix_a * sqrt_weights[:, None]
    vector_bw = vector_b * sqrt_weights

    try:
        coeffs, *_ = np.linalg.lstsq(matrix_aw, vector_bw, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None

    if coeffs is not None and not np.all(np.isfinite(coeffs)):
        coeffs = None

    return GroundSizeModel(coeffs, default_size)


@dataclass
class BallTrajectory2DResult:
    img_predictions: Dict[int, Dict[str, Any]]
    measurement_confidences: Dict[int, float]
    measurement_flags: Dict[int, bool]
    world_states_m: Dict[int, Tuple[float, float]]
    world_states_px: Dict[int, Tuple[float, float]]
    world_measurements_px: Dict[int, Tuple[float, float]]
    world_measurements_m: Dict[int, Tuple[float, float]]
    velocities_mps: Dict[int, Tuple[float, float]]
    speed_mps: Dict[int, float]
    dist_cum_m: Dict[int, float]
    world_height_m: Dict[int, float]
    world_height_px: Dict[int, float]
    vertical_speed_mps: Dict[int, float]


def run_planar_pipeline(
    frames_with_pred: List[int],
    detections: Dict[int, Dict[str, Any]],
    *,
    homography: np.ndarray,
    homography_by_frame: Optional[Dict[int, np.ndarray]] = None,
    px_per_m: float,
    img_w: int,
    img_h: int,
    min_confidence: float,
    fps: float,
    ball_diameter_m: float,
    size_model_min_samples: int,
    measurement_confidence_floor: float,
    max_interp_gap: int,
    height_max_m: float,
) -> BallTrajectory2DResult:
    """Estimate planar trajectory information and synthetic predictions."""
    img_predictions: Dict[int, Dict[str, Any]] = {}
    world_states_m: Dict[int, Tuple[float, float]] = {}
    world_states_px: Dict[int, Tuple[float, float]] = {}
    velocities_mps: Dict[int, Tuple[float, float]] = {}
    speed_mps: Dict[int, float] = {}
    dist_cum_m: Dict[int, float] = {}
    world_meas_px: Dict[int, Tuple[float, float]] = {}
    world_meas_m: Dict[int, Tuple[float, float]] = {}
    measurement_confs: Dict[int, float] = {}
    measurement_flags: Dict[int, bool] = {}
    world_height_m: Dict[int, float] = {}
    world_height_px: Dict[int, float] = {}
    vertical_speed_mps: Dict[int, float] = {}

    if not frames_with_pred:
        return BallTrajectory2DResult(
            img_predictions,
            measurement_confs,
            measurement_flags,
            world_states_m,
            world_states_px,
            world_meas_px,
            world_meas_m,
            velocities_mps,
            speed_mps,
            dist_cum_m,
            world_height_m,
            world_height_px,
            vertical_speed_mps,
        )

    default_ground_px = 24.0
    if px_per_m and px_per_m > 0 and ball_diameter_m > 0:
        default_ground_px = max(8.0, ball_diameter_m * px_per_m * 0.6)
    size_model = GroundSizeModel(None, default_ground_px)
    size_samples: List[Tuple[float, float, float, float]] = []
    size_px_by_frame: Dict[int, float] = {}
    ground_size_by_frame: Dict[int, float] = {}

    for frame in frames_with_pred:
        prediction = detections[frame]
        confidence = float(prediction.get("confidence", 0.0))
        if confidence < measurement_confidence_floor:
            continue
        x_val = float(prediction.get("x", 0.0))
        y_val = float(prediction.get("y", 0.0))
        w_px = max(1e-3, float(prediction.get("width", 0.0)))
        h_px = max(1e-3, float(prediction.get("height", 0.0)))
        size_px = math.sqrt(w_px * h_px)
        x_norm = x_val / max(1, img_w or 1)
        y_norm = y_val / max(1, img_h or 1)
        measurement_confs[frame] = confidence
        size_px_by_frame[frame] = size_px
        size_samples.append((x_norm, y_norm, size_px, confidence + 1e-3))

    if size_samples:
        size_model = build_ground_size_model(size_samples, size_model_min_samples)

    img_xy_meas: Dict[int, Tuple[float, float]] = {}
    for frame in frames_with_pred:
        if frame not in measurement_confs:
            continue
        prediction = detections[frame]
        x_val = float(prediction.get("x", 0.0))
        y_val = float(prediction.get("y", 0.0))
        img_xy_meas[frame] = (x_val, y_val)
        ground_size = size_model.predict(
            x_val / max(1, img_w or 1),
            y_val / max(1, img_h or 1),
        )
        detections[frame]["_ground_size_px"] = ground_size
        ground_size_by_frame[frame] = ground_size

    height_ratio_by_frame: Dict[int, float] = {}
    if ground_size_by_frame and size_px_by_frame:
        for frame, ground_size in ground_size_by_frame.items():
            size_px = size_px_by_frame.get(frame)
            if not size_px or size_px <= 1e-3:
                continue
            ratio = float(ground_size / size_px)
            if not math.isfinite(ratio):
                continue
            ratio = max(0.25, min(5.0, ratio))
            height_ratio_by_frame[frame] = ratio

    ratio_candidates = [ratio for ratio in height_ratio_by_frame.values() if ratio > 1.02]
    ratio_high = float(np.percentile(ratio_candidates, 90)) if ratio_candidates else 1.05
    ratio_high = max(ratio_high, 1.05)
    ratio_denom = max(1e-6, ratio_high - 1.0)

    height_max_m = max(0.1, float(height_max_m if height_max_m is not None else 3.0))
    height_proxy_m: Dict[int, float] = {}
    for frame, ratio in height_ratio_by_frame.items():
        if ratio <= 1.02:
            height_proxy_m[frame] = 0.0
            continue
        norm = (ratio - 1.0) / ratio_denom
        norm = max(0.0, min(1.2, norm))
        height_proxy_m[frame] = float(height_max_m * min(1.0, norm))

    if height_proxy_m:
        height_smoothed: Dict[int, float] = {}
        prev_frame: Optional[int] = None
        prev_val: Optional[float] = None
        smoothing_alpha = 0.65
        for frame in frames_with_pred:
            raw_val = float(height_proxy_m.get(frame, 0.0))
            if prev_val is None or prev_frame is None or (frame - prev_frame) > max(1, max_interp_gap):
                smoothed_val = raw_val
            else:
                smoothed_val = smoothing_alpha * raw_val + (1.0 - smoothing_alpha) * prev_val
            height_smoothed[frame] = smoothed_val
            prev_val = smoothed_val
            prev_frame = frame

        frames_idx = {frame: idx for idx, frame in enumerate(frames_with_pred)}
        for frame in frames_with_pred:
            idx = frames_idx[frame]
            window_vals = []
            for offset in (-1, 0, 1):
                neighbour_idx = idx + offset
                if 0 <= neighbour_idx < len(frames_with_pred):
                    neighbour_frame = frames_with_pred[neighbour_idx]
                    window_vals.append(height_smoothed.get(neighbour_frame, 0.0))
            if window_vals:
                height_smoothed[frame] = float(np.median(window_vals))

        height_proxy_m = height_smoothed

    if img_xy_meas:
        frames_img_sorted = sorted(img_xy_meas.keys())
        mapped_points: Dict[int, Tuple[float, float]] = {}

        if homography_by_frame:
            for frame in frames_img_sorted:
                H_frame = homography_by_frame.get(frame)
                if H_frame is None or np.shape(H_frame) != (3, 3):
                    H_frame = homography
                mapped_pt = apply_homography_points([img_xy_meas[frame]], H_frame)[0]
                mapped_points[frame] = (float(mapped_pt[0]), float(mapped_pt[1]))
        else:
            mapped = apply_homography_points([img_xy_meas[f] for f in frames_img_sorted], homography)
            mapped_points = {
                frame: (float(pt[0]), float(pt[1])) for frame, pt in zip(frames_img_sorted, mapped)
            }

        for frame in frames_img_sorted:
            wx, wy = mapped_points[frame]
            wx = float(wx)
            wy = float(wy)
            world_meas_px[frame] = (wx, wy)
            world_meas_m[frame] = (
                float(wx / px_per_m) if px_per_m else float(wx),
                float(wy / px_per_m) if px_per_m else float(wy),
            )
            height_val = float(height_proxy_m.get(frame, 0.0))
            world_height_m[frame] = height_val
            world_height_px[frame] = float(height_val * px_per_m) if px_per_m else height_val

    for frame in frames_with_pred:
        measurement_flags.setdefault(frame, frame in world_meas_m)

    if world_meas_m:
        frames_meas_sorted = sorted(world_meas_m.keys())
        cum_dist = 0.0
        prev_pos: Optional[Tuple[float, float]] = None
        prev_frame: Optional[int] = None
        prev_height: Optional[float] = None
        fps_safe = fps if fps and fps > 0 else None

        for frame in frames_meas_sorted:
            wx_m, wy_m = world_meas_m[frame]
            wx_px, wy_px = world_meas_px[frame]
            curr_height = world_height_m.get(frame, 0.0)

            world_states_m[frame] = (wx_m, wy_m)
            world_states_px[frame] = (wx_px, wy_px)
            measurement_flags[frame] = True

            vx_m = vy_m = 0.0
            has_velocity = False
            if prev_pos is not None and prev_frame is not None:
                dist = math.hypot(wx_m - prev_pos[0], wy_m - prev_pos[1])
                cum_dist += dist
                if fps_safe:
                    dt = (frame - prev_frame) / fps_safe if frame > prev_frame else 0.0
                    if dt > 1e-6:
                        vx_m = (wx_m - prev_pos[0]) / dt
                        vy_m = (wy_m - prev_pos[1]) / dt
                        has_velocity = True
                        if prev_height is not None:
                            vz = (curr_height - prev_height) / dt
                            if math.isfinite(vz):
                                vertical_speed_mps[frame] = float(vz)
            dist_cum_m[frame] = cum_dist

            if prev_pos is not None and prev_frame is not None and fps_safe and has_velocity:
                velocities_mps[frame] = (vx_m, vy_m)
                speed_mps[frame] = math.hypot(vx_m, vy_m)
            else:
                velocities_mps[frame] = (None, None)
                speed_mps[frame] = None

            prev_pos = (wx_m, wy_m)
            prev_frame = frame
            prev_height = curr_height
            vertical_speed_mps.setdefault(frame, 0.0)

    return BallTrajectory2DResult(
        img_predictions,
        measurement_confs,
        measurement_flags,
        world_states_m,
        world_states_px,
        world_meas_px,
        world_meas_m,
        velocities_mps,
        speed_mps,
        dist_cum_m,
        world_height_m,
        world_height_px,
        vertical_speed_mps,
    )


# Backwards compatibility exports
Pseudo3DResult = BallTrajectory2DResult
run_pseudo3d_pipeline = run_planar_pipeline
