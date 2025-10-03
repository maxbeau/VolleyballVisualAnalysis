from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.utils import ensure_dir
from ball.pseudo3d import BallTrajectory2DResult
from ball.trajectory_segmentation import (
    TrajectoryChangeEvent,
    TrajectorySegment,
    TrajectorySegmentationConfig,
)


_PALETTE: Sequence[Tuple[int, int, int]] = (
    (244, 67, 54),
    (0, 188, 212),
    (76, 175, 80),
    (255, 235, 59),
    (156, 39, 176),
    (33, 150, 243),
    (255, 152, 0),
    (121, 85, 72),
    (205, 220, 57),
    (63, 81, 181),
)


def render_segmentation_timeline(
    output_path: str,
    trajectory: BallTrajectory2DResult,
    fps: float,
    events: Sequence[TrajectoryChangeEvent],
    segments: Sequence[TrajectorySegment],
    cfg: TrajectorySegmentationConfig,
) -> bool:
    """Render a diagnostic timeline with segments, speed trace, and change events."""
    frames = _ordered_frames(trajectory)
    if not frames:
        return False

    speeds = _gather_speeds(frames, trajectory)
    if not any(math.isfinite(val) and val > 0 for val in speeds):
        return False

    height = 520
    width = max(900, min(2400, 120 + len(frames) * 4))
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 90
    base = np.full((height, width, 3), 20, dtype=np.uint8)

    seg_band_top = margin_top
    seg_band_bottom = seg_band_top + 130
    speed_band_top = seg_band_bottom + 40
    speed_band_bottom = height - margin_bottom

    frame_min = frames[0]
    frame_max = frames[-1]
    frame_span = max(1, frame_max - frame_min)

    def frame_to_x(frame: int) -> int:
        norm = (frame - frame_min) / frame_span
        return int(round(margin_left + norm * (width - margin_left - margin_right)))

    for idx, segment in enumerate(segments):
        color = _PALETTE[idx % len(_PALETTE)]
        start_x = frame_to_x(segment.start_frame)
        end_x = frame_to_x(segment.end_frame)
        cv2.rectangle(
            base,
            (start_x, seg_band_top),
            (end_x, seg_band_bottom),
            color,
            thickness=-1,
        )
        label = f"Segment {segment.segment_id}"
        if not segment.valid:
            label += " (short)"
        cv2.putText(
            base,
            label,
            (start_x + 6, seg_band_top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            lineType=cv2.LINE_AA,
        )
        details = []
        duration = segment.duration_sec
        if duration:
            details.append(f"{duration:.2f}s")
        if segment.change_reason:
            details.append(segment.change_reason)
        if details:
            cv2.putText(
                base,
                " / ".join(details),
                (start_x + 6, seg_band_top + 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )

    _draw_speed_trace(
        base,
        frames,
        speeds,
        frame_to_x,
        (speed_band_top, speed_band_bottom),
    )

    _draw_axes(
        base,
        frame_min,
        frame_max,
        fps,
        frame_to_x,
        margin_left,
        width - margin_right,
        speed_band_top,
        speed_band_bottom,
    )

    _draw_events(base, events, frame_to_x, seg_band_bottom, speed_band_top)

    meta_lines = _build_summary_lines(events, cfg)
    for idx, text in enumerate(meta_lines[:8]):
        cv2.putText(
            base,
            text,
            (margin_left, height - margin_bottom + 30 + idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )

    ensure_dir(os.path.dirname(output_path) or ".")
    cv2.imwrite(output_path, base)
    return True


def _ordered_frames(trajectory: BallTrajectory2DResult) -> List[int]:
    frames = sorted(set(trajectory.world_states_m.keys()) | set(trajectory.world_measurements_m.keys()))
    return frames


def _gather_speeds(frames: Sequence[int], trajectory: BallTrajectory2DResult) -> List[float]:
    speeds: List[float] = []
    for frame in frames:
        speed = trajectory.speed_mps.get(frame)
        if speed is None:
            vx, vy = trajectory.velocities_mps.get(frame, (None, None))
            if vx is not None and vy is not None:
                speed = math.hypot(vx, vy)
        speeds.append(float(speed) if speed is not None and math.isfinite(speed) else 0.0)
    return speeds


def _draw_speed_trace(
    img: np.ndarray,
    frames: Sequence[int],
    speeds: Sequence[float],
    frame_to_x,
    speed_band: Tuple[int, int],
) -> None:
    top, bottom = speed_band
    speed_max = max(0.1, max(speeds) * 1.1)
    points: List[Tuple[int, int]] = []
    for frame, speed in zip(frames, speeds):
        x = frame_to_x(frame)
        norm = min(1.0, max(0.0, speed / speed_max))
        y = bottom - int(round(norm * (bottom - top)))
        points.append((x, y))
    if len(points) >= 2:
        cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (250, 250, 250), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        img,
        "Speed (m/s)",
        (10, top - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"max ~ {speed_max:.1f}",
        (10, top + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (170, 170, 170),
        1,
        lineType=cv2.LINE_AA,
    )


def _draw_axes(
    img: np.ndarray,
    frame_min: int,
    frame_max: int,
    fps: float,
    frame_to_x,
    axis_start: int,
    axis_end: int,
    speed_top: int,
    speed_bottom: int,
) -> None:
    cv2.line(img, (axis_start, speed_bottom), (axis_end, speed_bottom), (120, 120, 120), 1, lineType=cv2.LINE_AA)
    frame_span = max(1, frame_max - frame_min)
    tick_count = 8
    for idx in range(tick_count + 1):
        ratio = idx / tick_count
        frame_val = frame_min + int(round(ratio * frame_span))
        x = frame_to_x(frame_val)
        cv2.line(img, (x, speed_bottom), (x, speed_bottom + 6), (180, 180, 180), 1)
        time_val = (frame_val - frame_min) / fps if fps else 0.0
        label = f"{time_val:.1f}s"
        cv2.putText(
            img,
            label,
            (x - 20, speed_bottom + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.putText(
        img,
        "Timeline",
        (axis_start, speed_bottom + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        lineType=cv2.LINE_AA,
    )


def _draw_events(
    img: np.ndarray,
    events: Sequence[TrajectoryChangeEvent],
    frame_to_x,
    top_band_bottom: int,
    speed_band_top: int,
) -> None:
    if not events:
        return
    text_offset = 0
    for event in events:
        x = frame_to_x(event.frame)
        color = (255, 255, 255)
        if event.reason == "track_gap":
            color = (80, 80, 80)
        cv2.line(img, (x, top_band_bottom), (x, speed_band_top), color, 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, top_band_bottom + 6), 5, color, -1, lineType=cv2.LINE_AA)
        label = event.reason.replace("+", "/")
        cv2.putText(
            img,
            label,
            (x + 4, speed_band_top - 8 - (text_offset % 3) * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
        text_offset += 1


def _build_summary_lines(
    events: Sequence[TrajectoryChangeEvent],
    cfg: TrajectorySegmentationConfig,
) -> List[str]:
    lines: List[str] = []
    lines.append(
        f"events: {len(events)} | speed ≥ {cfg.speed_jump_abs_mps:.2f} m/s | Δh ≥ {cfg.height_change_abs_m:.2f} m"
    )
    lines.append(
        f"direction ≥ {cfg.heading_change_abs_deg:.1f}° | drop ratio ≤ {cfg.speed_drop_ratio:.2f} | min height {cfg.min_height_for_event_m:.2f} m"
    )
    for event in events[:6]:
        parts = [f"f{event.frame}", event.reason, f"score {event.score:.2f}"]
        if event.delta_speed is not None:
            parts.append(f"Δv {event.delta_speed:.2f}")
        if event.delta_heading_deg is not None:
            parts.append(f"Δθ {event.delta_heading_deg:.1f}°")
        if event.delta_height_m is not None:
            parts.append(f"Δh {event.delta_height_m:.2f} m")
        lines.append(" | ".join(parts))
    if len(events) > 6:
        lines.append(f"(+{len(events) - 6} more)")
    return lines
