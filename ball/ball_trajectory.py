from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.utils import ensure_dir
from visualization.trajectory_debug import render_segmentation_timeline

from .ball_selection import (
    load_all_ball_candidates,
    load_best_ball_per_frame,
    select_ball_track_viterbi,
    soft_weight_aspect_ratio,
)
from .pseudo3d import BallTrajectory2DResult, run_planar_pipeline
from .trajectory_segmentation import (
    TrajectoryChangeEvent,
    TrajectorySegment,
    TrajectorySegmentationConfig,
    detect_touch_events,
)

__all__ = (
    "run_trajectory_analysis",
)


@dataclass
class TrajectoryIOConfig:
    video_path: str
    detections_jsonl: str
    homography_npy: str
    homography_meta_json: str
    birdseye_jpg: str
    output_jsonl: str
    output_csv: str
    output_path_img: str
    output_segmentation_img: Optional[str]
    homography_timeseries_jsonl: Optional[str] = None


@dataclass
class TrajectoryAnalysisConfig:
    min_confidence: float
    ar_filter_min: float
    ar_filter_max: float
    ar_filter_alpha: float
    max_interp_gap: int
    obs_gate_chisq: float
    obs_gate_use_conf: bool
    hold_ttl: int
    max_speed_px_per_frame: float
    max_accel_px_per_frame2: float
    speed_reset_frame_gap: int
    static_filter_enable: bool
    static_window_frames: int
    static_min_motion_px: float
    continuity_filter_enable: bool
    continuity_window_frames: int
    continuity_max_error_px: float
    continuity_error_growth_px: float
    viterbi_cfg: Any
    ball_diameter_m: float
    size_model_min_samples: int
    measurement_confidence_floor: float
    height_max_m: float = 3.3
    segmentation: TrajectorySegmentationConfig = field(default_factory=TrajectorySegmentationConfig)


def pred_with_kalman_or_hold(
    frames: List[int],
    best: Dict[int, Dict[str, Any]],
    smoothed: Dict[int, Dict[str, Any]],
    frame_index: int,
    hold_mode: str,
    hold_ttl: int,
) -> Optional[Dict[str, Any]]:
    """Pick a Kalman-smoothed prediction or fall back to neighbouring frames."""
    if frame_index in smoothed:
        return smoothed[frame_index].copy()

    if not frames:
        return None

    import bisect

    position = bisect.bisect_left(frames, frame_index)
    prev_idx = frames[position - 1] if position > 0 else None
    next_idx = frames[position] if position < len(frames) else None

    mode = (hold_mode or "prev").lower()
    if mode not in {"prev", "next", "both", "none"}:
        mode = "prev"

    if mode in {"prev", "both"} and prev_idx is not None:
        if (frame_index - prev_idx) <= hold_ttl:
            prev_pred = best[prev_idx].copy()
            prev_pred["_hold"] = True
            return prev_pred

    if mode in {"next", "both"} and next_idx is not None:
        if (next_idx - frame_index) <= hold_ttl:
            next_pred = best[next_idx].copy()
            next_pred["_hold"] = True
            return next_pred

    return None


def _load_and_select_track(
    detections_jsonl: str, viterbi_cfg: Any, img_wh: Tuple[int, int], ar_filter_min: float, ar_filter_max: float, ar_filter_alpha: float
) -> Dict[int, Dict[str, Any]]:
    """Loads all ball candidates and selects the most likely track."""
    allowed = ["ball", "volleyball"]
    preds_by_frame = load_all_ball_candidates(detections_jsonl, allowed)
    best = select_ball_track_viterbi(preds_by_frame, viterbi_cfg, img_wh=img_wh)
    if not best:
        best = load_best_ball_per_frame(detections_jsonl, allowed)
    return soft_weight_aspect_ratio(best, ar_filter_min, ar_filter_max, ar_filter_alpha)


def _apply_observation_gate(
    best: Dict[int, Dict[str, Any]],
    img_preds_full: Dict[int, Dict[str, Any]],
    obs_gate_chisq: float,
    obs_gate_use_conf: bool,
) -> Dict[int, Dict[str, Any]]:
    """Filters detections using Mahalanobis distance from Kalman predictions."""
    if not (obs_gate_chisq > 0.0 and best):
        return best

    def mahalanobis(px: float, py: float, mx: float, my: float, cov: np.ndarray) -> float:
        diff = np.array([[px - mx], [py - my]])
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return float("inf")
        value = float(diff.T @ inv_cov @ diff)
        return value if np.isfinite(value) else float("inf")

    base_cov = np.diag([16.0, 16.0])
    gated_best: Dict[int, Dict[str, Any]] = {}
    for frame, prediction in best.items():
        px, py = float(prediction.get("x", 0.0)), float(prediction.get("y", 0.0))
        if frame in img_preds_full:
            est = img_preds_full[frame]
            mx, my = float(est.get("x", px)), float(est.get("y", py))
            cov_matrix = base_cov
            if obs_gate_use_conf:
                conf = float(prediction.get("confidence", 0.0))
                scale = max(0.25, 1.0 - conf)
                cov_matrix = np.diag([16.0 * scale, 16.0 * scale])
            distance = mahalanobis(px, py, mx, my, cov_matrix)
            if distance <= obs_gate_chisq:
                gated_best[frame] = prediction
            else:
                gated_best[frame] = {**prediction, "_rejected": True, "_gate_distance": float(distance)}
        else:
            gated_best[frame] = prediction
    return gated_best


def _format_and_export_results(
    io_cfg: TrajectoryIOConfig,
    analysis_cfg: TrajectoryAnalysisConfig,
    best: Dict[int, Dict[str, Any]],
    trajectory_result: BallTrajectory2DResult,
    fps: float,
    px_per_m: float,
    touch_events: List[TrajectoryChangeEvent],
    segments: List[TrajectorySegment],
    frame_to_segment: Dict[int, int],
) -> None:
    """Formats the trajectory data and writes it to JSONL and CSV files."""
    img_preds_full = trajectory_result.img_predictions
    frames_img_sorted = sorted(img_preds_full.keys())
    smoothed_preds = {frame: img_preds_full[frame] for frame in frames_img_sorted}
    output_frames = sorted(set(best.keys()) | set(frames_img_sorted))

    export_rows: Dict[int, Dict[str, Any]] = {}
    csv_rows: Dict[int, Dict[str, Any]] = {}

    for frame in output_frames:
        img_pred = img_preds_full.get(frame)
        base_pred = best.get(frame)
        hold_pred = pred_with_kalman_or_hold(
            frames_img_sorted, best, smoothed_preds, frame, hold_mode="prev", hold_ttl=analysis_cfg.hold_ttl
        )

        img_p = img_pred or hold_pred or base_pred
        if img_p is None:
            continue

        vx_m, vy_m = trajectory_result.velocities_mps.get(frame, (None, None))
        speed_val = trajectory_result.speed_mps.get(frame)
        height_val_m = trajectory_result.world_height_m.get(frame, 0.0)
        height_val_px = trajectory_result.world_height_px.get(frame, height_val_m * px_per_m)
        vz_m = trajectory_result.vertical_speed_mps.get(frame)
        if vz_m is not None:
            try:
                vz_m = float(vz_m)
            except (TypeError, ValueError):
                vz_m = None
            else:
                if not math.isfinite(vz_m):
                    vz_m = None

        world_state = trajectory_result.world_states_m.get(frame)
        if world_state:
            wx_m, wy_m = world_state
            default_px = (wx_m * px_per_m, wy_m * px_per_m)
            wx_px, wy_px = trajectory_result.world_states_px.get(frame, default_px)
            wz_m = float(height_val_m)
            wz_px = float(height_val_px)
            dist_cum = trajectory_result.dist_cum_m.get(frame, 0.0)
        elif frame in trajectory_result.world_measurements_px:
            wx_px, wy_px = trajectory_result.world_measurements_px[frame]
            wx_m, wy_m = trajectory_result.world_measurements_m.get(
                frame,
                (float(wx_px / px_per_m), float(wy_px / px_per_m)),
            )
            wz_m = float(height_val_m)
            wz_px = float(height_val_px)
            dist_cum = trajectory_result.dist_cum_m.get(frame, 0.0)
        else:
            continue

        source = "kalman" if img_pred else ("hold" if hold_pred else "raw")
        export_rows[frame] = {
            "frame": int(frame), "time_sec": (frame / fps) if fps > 0 else None,
            "img_x": img_p.get("x"), "img_y": img_p.get("y"), "w": img_p.get("width"), "h": img_p.get("height"),
            "confidence": img_p.get("confidence"), "world_px": [float(wx_px), float(wy_px)],
            "world_m": [float(wx_m), float(wy_m)], "world_z_m": float(wz_m), "world_z_px": float(wz_px),
            "world_3d_m": [float(wx_m), float(wy_m), float(wz_m)], "vx_mps": vx_m, "vy_mps": vy_m, "vz_mps": vz_m,
            "speed_mps": speed_val, "distance_m_cum": dist_cum, "height_est_m": float(wz_m),
            "flags": {"interp": bool(img_p.get("_interp", False)), "hold": bool(img_p.get("_hold", False)), "source": source},
        }
        height_components = getattr(trajectory_result, "height_components", {})
        height_meta = height_components.get(frame) if isinstance(height_components, dict) else None
        if height_meta:
            debug_payload: Dict[str, Any] = {}
            for key, value in height_meta.items():
                if isinstance(value, (int, float)):
                    debug_payload[key] = float(value)
                else:
                    debug_payload[key] = value
            export_rows[frame]["height_debug"] = debug_payload
        csv_rows[frame] = {
            "frame": int(frame), "time_sec": (frame / fps) if fps > 0 else "",
            "world_x_px": float(wx_px), "world_y_px": float(wy_px), "world_z_px": float(wz_px),
            "world_x_m": float(wx_m), "world_y_m": float(wy_m), "world_z_m": float(wz_m),
            "vx_mps": vx_m or "", "vy_mps": vy_m or "", "vz_mps": vz_m or "",
            "speed_mps": speed_val or "", "distance_m_cum": dist_cum,
        }

    events_by_frame = {event.frame: event for event in touch_events}
    segments_by_id = {segment.segment_id: segment for segment in segments}

    for frame in export_rows:
        segment_id = frame_to_segment.get(frame)
        if segment_id is not None:
            segment = segments_by_id.get(segment_id)
            export_rows[frame]["segment_id"] = segment_id
            if segment:
                export_rows[frame]["segment_valid"] = segment.valid
                export_rows[frame]["segment_duration_sec"] = segment.duration_sec
                if frame == segment.start_frame:
                    export_rows[frame]["segment_is_start"] = True
                if frame == segment.end_frame:
                    export_rows[frame]["segment_is_end"] = True
                if segment.change_frame is not None and frame == segment.end_frame:
                    export_rows[frame]["segment_change_reason"] = segment.change_reason
                    export_rows[frame]["segment_change_score"] = segment.change_score
        event = events_by_frame.get(frame)
        if event:
            if segment_id is not None:
                export_rows[frame].setdefault("segment_id", segment_id)
            export_rows[frame]["touch_event"] = {
                "reason": event.reason,
                "score": round(event.score, 3),
                "delta_speed": event.delta_speed,
                "delta_heading_deg": event.delta_heading_deg,
                "delta_height_m": event.delta_height_m,
                "prev_speed": event.prev_speed,
                "next_speed": event.next_speed,
                "prev_heading_deg": event.prev_heading_deg,
                "next_heading_deg": event.next_heading_deg,
                "prev_height_m": event.prev_height_m,
                "next_height_m": event.next_height_m,
                "details": dict(event.details),
                "segment_id": segment_id,
            }

    with open(io_cfg.output_jsonl, "w", encoding="utf-8") as jf:
        for frame in sorted(export_rows.keys()):
            jf.write(json.dumps(export_rows[frame], ensure_ascii=False) + "\n")

    with open(io_cfg.output_csv, "w", newline="", encoding="utf-8") as cf:
        header = [
            "frame", "time_sec", "world_x_px", "world_y_px", "world_z_px", "world_x_m", "world_y_m", "world_z_m",
            "vx_mps", "vy_mps", "vz_mps", "speed_mps", "distance_m_cum",
        ]
        writer = csv.writer(cf)
        writer.writerow(header)
        for frame in sorted(csv_rows.keys()):
            writer.writerow([csv_rows[frame][k] for k in header])


def _draw_path_on_birdseye(
    output_path_img: str,
    birdseye_jpg: str,
    dst_size: Tuple[int, int],
    trajectory_result: BallTrajectory2DResult,
) -> None:
    """Draws the ball's trajectory on a bird's-eye view of the court."""
    bg = cv2.imread(birdseye_jpg) if os.path.exists(birdseye_jpg) else np.full((dst_size[1], dst_size[0], 3), 255, dtype=np.uint8)

    path_pts: List[Tuple[int, int]] = []
    if trajectory_result.world_states_px:
        path_frames = sorted(trajectory_result.world_states_px.keys())
        path_pts = [
            (int(round(px)), int(round(py)))
            for frame in path_frames
            for px, py in [trajectory_result.world_states_px[frame]]
        ]
    elif trajectory_result.world_measurements_px:
        path_frames = sorted(trajectory_result.world_measurements_px.keys())
        path_pts = [
            (int(round(px)), int(round(py)))
            for frame in path_frames
            for px, py in [trajectory_result.world_measurements_px[frame]]
        ]

    if len(path_pts) >= 2:
        cv2.polylines(bg, [np.array(path_pts, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)
        cv2.circle(bg, path_pts[0], 5, (0, 200, 0), -1)
        cv2.circle(bg, path_pts[-1], 5, (0, 0, 200), -1)
    cv2.imwrite(output_path_img, bg)


def run_trajectory_analysis(
    io_cfg: TrajectoryIOConfig,
    analysis_cfg: TrajectoryAnalysisConfig,
) -> None:
    """Maps ball detections to court coordinates and analyzes trajectory."""
    ensure_dir(os.path.dirname(io_cfg.output_jsonl) or ".")
    ensure_dir(os.path.dirname(io_cfg.output_csv) or ".")
    ensure_dir(os.path.dirname(io_cfg.output_path_img) or ".")

    cap = cv2.VideoCapture(io_cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {io_cfg.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_w, img_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if not os.path.exists(io_cfg.homography_npy) or not os.path.exists(io_cfg.homography_meta_json):
        raise FileNotFoundError(f"Court homography not found for: {io_cfg.homography_npy}")
    H = np.load(io_cfg.homography_npy)
    with open(io_cfg.homography_meta_json, "r", encoding="utf-8") as f:
        H_meta = json.load(f)
    dst_size = (int(H_meta.get("dst_size", {}).get("w", 1800)), int(H_meta.get("dst_size", {}).get("h", 900)))
    px_per_m = float(H_meta.get("scale_px_per_meter", 100.0))

    homography_by_frame: Dict[int, np.ndarray] = {}
    timeseries_path = io_cfg.homography_timeseries_jsonl
    if timeseries_path and os.path.exists(timeseries_path):
        with open(timeseries_path, "r", encoding="utf-8") as ts_f:
            for line in ts_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                frame_val = rec.get("frame")
                H_flat = rec.get("homography")
                if frame_val is None or not H_flat or len(H_flat) != 9:
                    continue
                try:
                    H_frame = np.array(H_flat, dtype=float).reshape(3, 3)
                except (ValueError, TypeError):
                    continue
                homography_by_frame[int(frame_val)] = H_frame

    best = _load_and_select_track(
        io_cfg.detections_jsonl, analysis_cfg.viterbi_cfg, (img_w, img_h),
        analysis_cfg.ar_filter_min, analysis_cfg.ar_filter_max, analysis_cfg.ar_filter_alpha
    )

    frames_with_pred = sorted(best.keys())

    planar_args = {
        "min_confidence": analysis_cfg.min_confidence,
        "ball_diameter_m": analysis_cfg.ball_diameter_m,
        "size_model_min_samples": analysis_cfg.size_model_min_samples,
        "measurement_confidence_floor": analysis_cfg.measurement_confidence_floor,
        "max_interp_gap": analysis_cfg.max_interp_gap,
        "height_max_m": analysis_cfg.height_max_m,
    }
    trajectory_result = run_planar_pipeline(
        frames_with_pred,
        best,
        homography=H,
        homography_by_frame=homography_by_frame or None,
        px_per_m=px_per_m,
        img_w=img_w,
        img_h=img_h,
        fps=fps,
        **planar_args,
    )

    touch_events: List[TrajectoryChangeEvent] = []
    segments: List[TrajectorySegment] = []
    frame_to_segment: Dict[int, int] = {}
    if analysis_cfg.segmentation.enable:
        touch_events, segments, frame_to_segment = detect_touch_events(
            trajectory_result,
            fps=fps,
            cfg=analysis_cfg.segmentation,
        )

    _format_and_export_results(
        io_cfg,
        analysis_cfg,
        best,
        trajectory_result,
        fps,
        px_per_m,
        touch_events,
        segments,
        frame_to_segment,
    )
    _draw_path_on_birdseye(io_cfg.output_path_img, io_cfg.birdseye_jpg, dst_size, trajectory_result)

    timeline_rendered = False
    if io_cfg.output_segmentation_img:
        timeline_rendered = render_segmentation_timeline(
            io_cfg.output_segmentation_img,
            trajectory_result,
            fps,
            touch_events,
            segments,
            analysis_cfg.segmentation,
        )

    if touch_events:
        print("Detected trajectory change events:")
        for event in touch_events:
            reason = event.reason
            print(
                f"  frame {event.frame}: {reason}"
                f" (score={event.score:.2f})"
            )

    print(f"Trajectory JSONL: {io_cfg.output_jsonl}")
    print(f"Trajectory CSV:   {io_cfg.output_csv}")
    print(f"Bird's-eye path:  {io_cfg.output_path_img}")
    if timeline_rendered:
        print(f"Segments timeline: {io_cfg.output_segmentation_img}")
