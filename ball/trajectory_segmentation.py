from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pseudo3d import BallTrajectory2DResult


@dataclass
class TrajectorySegmentationConfig:
    """Runtime parameters controlling trajectory change detection."""

    enable: bool = True
    gap_frame_threshold: int = 8
    smoothing_window_frames: int = 3
    min_segment_frames: int = 6
    min_speed_mps: float = 1.0
    min_heading_speed_mps: float = 2.5
    speed_jump_abs_mps: float = 3.0
    speed_jump_mad_multiplier: float = 2.8
    speed_jump_ratio_min: float = 3.0
    heading_change_abs_deg: float = 33.0
    heading_change_mad_multiplier: float = 2.5
    height_change_abs_m: float = 0.35
    height_change_mad_multiplier: float = 2.4
    min_height_for_event_m: float = 0.35
    require_speed_or_height_for_flip: bool = True
    gap_guard_frames: int = 4
    vertical_zero_cross_enable: bool = True
    min_vertical_speed_mps: float = 2.0
    speed_jump_noheight_multiplier: float = 1.8
    max_event_height_m: float = 0.55
    combined_score_threshold: float = 1.2
    merge_event_window_frames: int = 3
    speed_drop_ratio: float = 0.55
    annotate_segments: bool = True


@dataclass
class TrajectoryChangeEvent:
    frame: int
    next_frame: Optional[int]
    score: float
    reason: str
    delta_speed: Optional[float]
    delta_heading_deg: Optional[float]
    delta_height_m: Optional[float]
    prev_speed: Optional[float]
    next_speed: Optional[float]
    prev_heading_deg: Optional[float]
    next_heading_deg: Optional[float]
    prev_height_m: Optional[float]
    next_height_m: Optional[float]
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrajectorySegment:
    segment_id: int
    start_frame: int
    end_frame: int
    change_frame: Optional[int]
    change_reason: Optional[str]
    change_score: Optional[float]
    duration_sec: Optional[float]
    valid: bool


def detect_touch_events(
    trajectory: BallTrajectory2DResult,
    fps: float,
    cfg: TrajectorySegmentationConfig,
) -> Tuple[List[TrajectoryChangeEvent], List[TrajectorySegment], Dict[int, int]]:
    """Identify significant velocity/direction changes along the trajectory."""
    frames = sorted(trajectory.world_states_m.keys())
    if not frames and trajectory.world_measurements_m:
        frames = sorted(trajectory.world_measurements_m.keys())

    if not frames or len(frames) < 2 or not cfg.enable:
        return [], [], {}

    speeds: List[float] = []
    headings: List[float] = []
    heights: List[float] = []
    velocities = trajectory.velocities_mps
    heights_map = getattr(trajectory, "world_height_m", {}) or {}

    for frame in frames:
        vx, vy = velocities.get(frame, (None, None))
        speed_val = trajectory.speed_mps.get(frame)
        if speed_val is None and vx is not None and vy is not None:
            speed_val = math.hypot(float(vx), float(vy))
        speeds.append(_finite_float(speed_val))
        if vx is None or vy is None:
            headings.append(math.nan)
        else:
            headings.append(math.degrees(math.atan2(float(vy), float(vx))))
        heights.append(_finite_float(heights_map.get(frame)))

    smooth_window = max(1, int(cfg.smoothing_window_frames))
    speeds_smooth = _smooth_scalar_series(speeds, smooth_window)
    headings_smooth = _smooth_angle_series(headings, smooth_window)
    heights_smooth = _smooth_scalar_series(heights, smooth_window)

    delta_speeds: List[float] = []
    delta_headings: List[float] = []
    delta_heights: List[float] = []
    for idx in range(1, len(frames)):
        gap = frames[idx] - frames[idx - 1]
        if gap > cfg.gap_frame_threshold:
            continue
        prev_speed = speeds_smooth[idx - 1]
        curr_speed = speeds_smooth[idx]
        if math.isfinite(prev_speed) and math.isfinite(curr_speed):
            delta_speeds.append(abs(curr_speed - prev_speed))
        prev_heading = headings_smooth[idx - 1]
        curr_heading = headings_smooth[idx]
        if math.isfinite(prev_heading) and math.isfinite(curr_heading):
            delta_headings.append(_angular_difference_deg(curr_heading, prev_heading))
        prev_height = heights_smooth[idx - 1]
        curr_height = heights_smooth[idx]
        if math.isfinite(prev_height) and math.isfinite(curr_height):
            delta_heights.append(abs(curr_height - prev_height))

    speed_scale = cfg.speed_jump_abs_mps
    median_speed_delta, mad_speed_delta = _median_abs_deviation(delta_speeds)
    if mad_speed_delta > 1e-6:
        candidate = median_speed_delta + cfg.speed_jump_mad_multiplier * mad_speed_delta
        if math.isfinite(candidate):
            speed_scale = max(speed_scale, candidate)

    heading_scale = cfg.heading_change_abs_deg
    median_heading_delta, mad_heading_delta = _median_abs_deviation(delta_headings)
    if mad_heading_delta > 1e-6:
        candidate = median_heading_delta + cfg.heading_change_mad_multiplier * mad_heading_delta
        if math.isfinite(candidate):
            heading_scale = max(heading_scale, candidate)

    height_scale = cfg.height_change_abs_m
    median_height_delta, mad_height_delta = _median_abs_deviation(delta_heights)
    if mad_height_delta > 1e-6:
        candidate = median_height_delta + cfg.height_change_mad_multiplier * mad_height_delta
        if math.isfinite(candidate):
            height_scale = max(height_scale, candidate)

    events: List[TrajectoryChangeEvent] = []
    gap_cooldown = 0
    vertical_speeds = getattr(trajectory, "vertical_speed_mps", {}) or {}

    for idx in range(1, len(frames)):
        frame = frames[idx]
        prev_frame = frames[idx - 1]
        gap = frame - prev_frame
        prev_speed = speeds[idx - 1]
        curr_speed = speeds[idx]
        prev_speed_smooth = speeds_smooth[idx - 1]
        curr_speed_smooth = speeds_smooth[idx]
        prev_heading = headings_smooth[idx - 1]
        curr_heading = headings_smooth[idx]
        prev_height = heights_smooth[idx - 1]
        curr_height = heights_smooth[idx]
        prev_vel = velocities.get(prev_frame, (None, None))
        curr_vel = velocities.get(frame, (None, None))
        prev_heading_raw = None
        curr_heading_raw = None
        if prev_vel and prev_vel[0] is not None and prev_vel[1] is not None:
            prev_heading_raw = math.degrees(math.atan2(float(prev_vel[1]), float(prev_vel[0])))
        if curr_vel and curr_vel[0] is not None and curr_vel[1] is not None:
            curr_heading_raw = math.degrees(math.atan2(float(curr_vel[1]), float(curr_vel[0])))

        if gap > cfg.gap_frame_threshold:
            events.append(
                TrajectoryChangeEvent(
                    frame=frame,
                    next_frame=frames[idx + 1] if (idx + 1) < len(frames) else None,
                    score=1.0,
                    reason="track_gap",
                    delta_speed=None,
                    delta_heading_deg=None,
                    delta_height_m=None,
                    prev_speed=_finite_optional(prev_speed),
                    next_speed=None,
                    prev_heading_deg=_finite_optional(prev_heading),
                    next_heading_deg=None,
                    prev_height_m=_finite_optional(prev_height),
                    next_height_m=None,
                    details={"gap_frames": float(gap)},
                )
            )
            gap_cooldown = cfg.gap_guard_frames
            continue

        cooldown_active = gap_cooldown > 0
        if gap_cooldown > 0:
            gap_cooldown -= 1

        if gap > cfg.gap_guard_frames:
            gap_cooldown = max(gap_cooldown, cfg.gap_guard_frames)

        if not (math.isfinite(prev_speed) and math.isfinite(curr_speed)):
            continue

        active_speed = max(prev_speed, curr_speed)
        if active_speed < cfg.min_speed_mps:
            continue

        delta_speed = abs(curr_speed - prev_speed)
        speed_trigger = delta_speed >= speed_scale
        speed_score = delta_speed / speed_scale if speed_scale > 1e-6 else 0.0

        ratio_trigger = False
        ratio_score = 0.0
        if cfg.speed_jump_ratio_min > 1.0 and math.isfinite(prev_speed) and math.isfinite(curr_speed):
            slower = min(abs(prev_speed), abs(curr_speed))
            faster = max(abs(prev_speed), abs(curr_speed))
            if slower > 1e-3:
                ratio = faster / slower
                ratio_trigger = ratio >= cfg.speed_jump_ratio_min
                if ratio_trigger:
                    denom = max(cfg.speed_jump_ratio_min - 1.0, 1e-6)
                    ratio_score = (ratio - 1.0) / denom
        if ratio_trigger and curr_speed > prev_speed:
            speed_trigger = True

        delta_heading = None
        heading_trigger = False
        heading_score = 0.0
        delta_heading_candidates: List[float] = []
        if math.isfinite(prev_heading) and math.isfinite(curr_heading):
            delta_heading_candidates.append(_angular_difference_deg(curr_heading, prev_heading))
        if prev_heading_raw is not None and curr_heading_raw is not None and math.isfinite(prev_heading_raw) and math.isfinite(curr_heading_raw):
            delta_heading_candidates.append(_angular_difference_deg(curr_heading_raw, prev_heading_raw))
        if delta_heading_candidates:
            delta_heading = max(delta_heading_candidates)
            heading_trigger = delta_heading >= heading_scale
            heading_score = delta_heading / heading_scale if heading_scale > 1e-6 else 0.0

        drop_trigger = (
            prev_speed >= cfg.min_speed_mps
            and curr_speed <= prev_speed * cfg.speed_drop_ratio
        )
        drop_score = 0.0
        if drop_trigger:
            drop_score = (prev_speed - curr_speed) / speed_scale if speed_scale > 1e-6 else 1.0

        delta_height = None
        height_trigger = False
        height_score = 0.0
        if math.isfinite(prev_height) and math.isfinite(curr_height):
            delta_height = abs(curr_height - prev_height)
            height_trigger = delta_height >= height_scale
            height_score = delta_height / height_scale if height_scale > 1e-6 else 0.0

        event_height = 0.0
        prev_height_val = _finite_optional(prev_height)
        next_height_val = _finite_optional(curr_height)
        if prev_height_val is not None:
            event_height = max(event_height, prev_height_val)
        if next_height_val is not None:
            event_height = max(event_height, next_height_val)

        prev_vz = _finite_float(vertical_speeds.get(prev_frame))
        curr_vz = _finite_float(vertical_speeds.get(frame))
        zero_cross_trigger = False
        vz_score = 0.0
        if cfg.vertical_zero_cross_enable and math.isfinite(prev_vz) and math.isfinite(curr_vz):
            if prev_vz * curr_vz < 0:
                max_vz = max(abs(prev_vz), abs(curr_vz))
                if max_vz >= cfg.min_vertical_speed_mps:
                    zero_cross_trigger = True
                    vz_score = max_vz / max(cfg.min_vertical_speed_mps, 1e-6)

        if zero_cross_trigger and not (speed_trigger or height_trigger or drop_trigger or heading_trigger):
            zero_cross_trigger = False
            vz_score = 0.0

        if active_speed < cfg.min_heading_speed_mps:
            heading_trigger = False

        if heading_trigger and cfg.require_speed_or_height_for_flip:
            allow_heading = height_trigger or drop_trigger or speed_trigger or zero_cross_trigger
            if not allow_heading and event_height >= cfg.min_height_for_event_m:
                allow_heading = True
            if not allow_heading and delta_heading is not None:
                if delta_heading >= heading_scale * 1.5:
                    allow_heading = True
            heading_trigger = allow_heading

        if speed_trigger and event_height < cfg.min_height_for_event_m and not drop_trigger and not height_trigger:
            allow_speed = zero_cross_trigger or ratio_trigger
            if delta_speed is not None and speed_scale > 1e-6:
                if delta_speed >= speed_scale * cfg.speed_jump_noheight_multiplier:
                    allow_speed = True
            if not allow_speed:
                speed_trigger = False

        if drop_trigger and event_height < cfg.min_height_for_event_m and not zero_cross_trigger:
            if ratio_trigger and prev_speed > curr_speed:
                pass
            else:
                drop_trigger = False
                drop_score = 0.0

        combined_score = max(speed_score, heading_score, drop_score, height_score, vz_score, ratio_score)

        reasons: List[str] = []
        if speed_trigger:
            reasons.append("speed_jump")
        if heading_trigger:
            reasons.append("direction_flip")
        if drop_trigger:
            reasons.append("speed_drop")
        if height_trigger:
            reasons.append("height_spike")
        if zero_cross_trigger:
            reasons.append("vz_cross")

        reason_tokens = set(reasons)

        if cooldown_active and not (ratio_trigger or height_trigger or zero_cross_trigger or heading_trigger):
            continue

        vz_mag_ok = (
            (math.isfinite(prev_vz) and abs(prev_vz) >= cfg.min_vertical_speed_mps)
            or (math.isfinite(curr_vz) and abs(curr_vz) >= cfg.min_vertical_speed_mps)
        )

        near_ground = (
            event_height is not None
            and math.isfinite(event_height)
            and event_height <= cfg.max_event_height_m
        )
        very_low_height = (
            event_height is not None
            and math.isfinite(event_height)
            and event_height <= cfg.min_height_for_event_m
        )
        prev_speed_val = prev_speed if math.isfinite(prev_speed) else None
        curr_speed_val = curr_speed if math.isfinite(curr_speed) else None
        curr_speed_val = curr_speed if math.isfinite(curr_speed) else None

        contact_trigger = False
        if 'track_gap' in reason_tokens:
            contact_trigger = True
        else:
            vz_or_speed_strong = vz_mag_ok or (
                prev_speed_val is not None and abs(prev_speed_val) >= cfg.min_heading_speed_mps
            ) or (
                curr_speed_val is not None and abs(curr_speed_val) >= cfg.min_heading_speed_mps
            )

            if zero_cross_trigger and vz_mag_ok and near_ground:
                contact_trigger = True
            elif height_trigger and near_ground:
                contact_trigger = True
            elif (
                speed_trigger
                and near_ground
                and (
                    zero_cross_trigger
                    or height_trigger
                    or drop_trigger
                    or ratio_trigger
                    or vz_or_speed_strong
                )
            ):
                contact_trigger = True
            elif (
                heading_trigger
                and near_ground
                and (
                    zero_cross_trigger
                    or height_trigger
                    or ratio_trigger
                    or vz_or_speed_strong
                )
            ):
                contact_trigger = True

        if not contact_trigger:
            continue

        if 'track_gap' in reason_tokens and combined_score < cfg.combined_score_threshold:
            combined_score = cfg.combined_score_threshold

        if combined_score < cfg.combined_score_threshold:
            continue

        events.append(
            TrajectoryChangeEvent(
                frame=frame,
                next_frame=frames[idx + 1] if (idx + 1) < len(frames) else None,
                score=float(combined_score),
                reason="+".join(sorted(set(reasons))),
                delta_speed=_finite_optional(delta_speed),
                delta_heading_deg=_finite_optional(delta_heading),
                delta_height_m=_finite_optional(delta_height),
                prev_speed=_finite_optional(prev_speed),
                next_speed=_finite_optional(curr_speed),
                prev_heading_deg=_finite_optional(prev_heading),
                next_heading_deg=_finite_optional(curr_heading),
                prev_height_m=prev_height_val,
                next_height_m=next_height_val,
                details={
                    "speed_score": float(speed_score),
                    "heading_score": float(heading_score),
                    "drop_score": float(drop_score),
                    "height_score": float(height_score),
                    "vz_score": float(vz_score),
                    "ratio_score": float(ratio_score),
                },
            )
        )

    events = _merge_close_events(events, cfg.merge_event_window_frames)

    segments: List[TrajectorySegment] = []
    frame_to_segment: Dict[int, int] = {}

    if cfg.annotate_segments:
        frame_index = {frame: idx for idx, frame in enumerate(frames)}
        segment_start_idx = 0

        for event in events:
            reason_tokens = set(event.reason.split('+')) if event.reason else set()
            if 'track_gap' in reason_tokens:
                continue
            if reason_tokens == {'speed_drop'}:
                continue
            event_idx = frame_index.get(event.frame)
            if event_idx is None or event_idx <= segment_start_idx:
                continue
            segment_frames = frames[segment_start_idx:event_idx]
            if not segment_frames:
                segment_start_idx = event_idx
                continue
            segment_id = len(segments)
            start_frame = segment_frames[0]
            end_frame = segment_frames[-1]
            duration_sec = _duration_sec(start_frame, end_frame, fps)
            valid = len(segment_frames) >= max(1, cfg.min_segment_frames)
            segments.append(
                TrajectorySegment(
                    segment_id=segment_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    change_frame=event.frame,
                    change_reason=event.reason,
                    change_score=event.score,
                    duration_sec=duration_sec,
                    valid=valid,
                )
            )
            for fr in segment_frames:
                frame_to_segment[fr] = segment_id
            segment_start_idx = event_idx

        remaining_frames = frames[segment_start_idx:]
        if remaining_frames:
            segment_id = len(segments)
            start_frame = remaining_frames[0]
            end_frame = remaining_frames[-1]
            duration_sec = _duration_sec(start_frame, end_frame, fps)
            valid = len(remaining_frames) >= max(1, cfg.min_segment_frames)
            segments.append(
                TrajectorySegment(
                    segment_id=segment_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    change_frame=None,
                    change_reason=None,
                    change_score=None,
                    duration_sec=duration_sec,
                    valid=valid,
                )
            )
            for fr in remaining_frames:
                frame_to_segment[fr] = segment_id

    return events, segments, frame_to_segment


def _finite_float(value: Optional[float]) -> float:
    if value is None:
        return math.nan
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value_f if math.isfinite(value_f) else math.nan


def _finite_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def _smooth_scalar_series(series: List[float], window: int) -> List[float]:
    if window <= 1:
        return [float(val) if math.isfinite(val) else math.nan for val in series]
    half = window // 2
    length = len(series)
    smoothed: List[float] = []
    for idx in range(length):
        start = max(0, idx - half)
        end = min(length, idx + half + 1)
        vals = [series[j] for j in range(start, end) if math.isfinite(series[j])]
        if vals:
            smoothed.append(float(sum(vals) / len(vals)))
        else:
            smoothed.append(float(series[idx]) if math.isfinite(series[idx]) else math.nan)
    return smoothed


def _smooth_angle_series(series_deg: List[float], window: int) -> List[float]:
    if window <= 1:
        return [float(val) if math.isfinite(val) else math.nan for val in series_deg]
    half = window // 2
    length = len(series_deg)
    smoothed: List[float] = []
    for idx in range(length):
        start = max(0, idx - half)
        end = min(length, idx + half + 1)
        sin_sum = 0.0
        cos_sum = 0.0
        count = 0
        for j in range(start, end):
            val = series_deg[j]
            if math.isfinite(val):
                rad = math.radians(val)
                sin_sum += math.sin(rad)
                cos_sum += math.cos(rad)
                count += 1
        if count:
            smoothed.append(math.degrees(math.atan2(sin_sum / count, cos_sum / count)))
        else:
            orig = series_deg[idx]
            smoothed.append(float(orig) if math.isfinite(orig) else math.nan)
    return smoothed


def _angular_difference_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def _median_abs_deviation(values: List[float]) -> Tuple[float, float]:
    filtered = [float(v) for v in values if math.isfinite(v)]
    if not filtered:
        return 0.0, 0.0
    filtered.sort()
    mid = len(filtered) // 2
    if len(filtered) % 2:
        median = filtered[mid]
    else:
        median = 0.5 * (filtered[mid - 1] + filtered[mid])
    deviations = [abs(v - median) for v in filtered]
    deviations.sort()
    mid_dev = len(deviations) // 2
    if len(deviations) % 2:
        mad = deviations[mid_dev]
    else:
        mad = 0.5 * (deviations[mid_dev - 1] + deviations[mid_dev])
    return float(median), float(mad)


def _merge_close_events(events: List[TrajectoryChangeEvent], merge_window: int) -> List[TrajectoryChangeEvent]:
    if merge_window <= 0 or len(events) <= 1:
        return events
    merged: List[TrajectoryChangeEvent] = []
    for event in events:
        if not merged:
            merged.append(event)
            continue
        prev_event = merged[-1]
        if event.reason == "track_gap" or prev_event.reason == "track_gap":
            merged.append(event)
            continue
        if event.frame - prev_event.frame <= merge_window:
            if event.score >= prev_event.score:
                merged[-1] = event
        else:
            merged.append(event)
    return merged


def _duration_sec(start_frame: int, end_frame: int, fps: float) -> Optional[float]:
    if fps and fps > 0 and end_frame >= start_frame:
        return (end_frame - start_frame + 1) / fps
    return None
