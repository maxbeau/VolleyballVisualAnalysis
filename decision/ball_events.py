from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from court.utils import apply_homography_points, compute_homography

from decision.rally_types import BallEvent, BallSample


class CourtMapper:
    """Lightweight helper that maps image coordinates into a canonical court plane."""

    def __init__(self, timeseries: Dict[int, Sequence[Tuple[float, float]]], width: int, height: int) -> None:
        self._timeseries = timeseries or {}
        self._width = int(width)
        self._height = int(height)

    def _homography_for(self, frame: int) -> Optional[List[List[float]]]:
        pts = self._timeseries.get(frame)
        if pts is None:
            prev = [k for k in self._timeseries.keys() if k <= frame]
            if prev:
                pts = self._timeseries.get(max(prev))
        if pts is None or len(pts) < 4:
            return None
        try:
            H, _ = compute_homography(pts, dst_size=(self._width, self._height))
            return H
        except Exception:
            return None

    def image_to_model(self, frame: int, x: float, y: float) -> Optional[Tuple[float, float]]:
        H = self._homography_for(frame)
        if H is None:
            return None
        try:
            (tx, ty) = apply_homography_points([(x, y)], H)[0]
            return float(tx), float(ty)
        except Exception:
            return None

    def side_for_point(self, frame: int, x: float, y: float) -> Optional[str]:
        mapped = self.image_to_model(frame, x, y)
        if mapped is None:
            return None
        tx, _ = mapped
        if tx < (self._width * 0.5):
            return "left"
        return "right"

    def point_in_bounds(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        w = float(self._width)
        h = float(self._height)
        m = max(0.0, float(margin))
        return (m <= x <= (w - m)) and (m <= y <= (h - m))

    @property
    def size(self) -> Tuple[float, float]:
        return float(self._width), float(self._height)


def build_ball_samples(
    ball_tracks: Dict[int, Dict[str, float]],
    fps: float,
    mapper: CourtMapper,
    side_to_team: Optional[Dict[str, str]] = None,
) -> Dict[int, BallSample]:
    samples: Dict[int, BallSample] = {}
    for frame, det in ball_tracks.items():
        try:
            x = float(det.get("x", 0.0))
            y = float(det.get("y", 0.0))
        except Exception:
            continue
        conf = float(det.get("confidence", 0.0)) if det.get("confidence") is not None else 0.0
        mapped = mapper.image_to_model(frame, x, y) if mapper else None
        court_side = mapper.side_for_point(frame, x, y) if mapper else None
        team_name = side_to_team.get(court_side) if (side_to_team and court_side) else None
        if mapped is not None:
            cx, cy = mapped
            mapped_flag = True
        else:
            cx, cy = x, y
            mapped_flag = False
        sample = BallSample(
            frame=int(frame),
            x=float(cx),
            y=float(cy),
            confidence=conf,
            quality=det.get("quality"),
            court_side=court_side,
            team_name=team_name,
            is_mapped=mapped_flag,
        )
        samples[int(frame)] = sample
    return samples


def detect_ball_events(
    samples: Dict[int, BallSample],
    *,
    gap_frames: int = 4,
    mapper: Optional[CourtMapper] = None,
    in_bounds_margin: float = 24.0,
    ground_band_ratio: float = 0.12,
) -> List[BallEvent]:
    """Derive coarse ball events (ground / out) from sparse trajectory samples.

    Heuristic:
      - Whenever there is a temporal gap between samples, treat the last sample
        before the gap as an event.
      - If the sample is mapped to the court plane and lies inside the canonical
        court bounds (with a small margin), label it as ``ground``.
      - Otherwise, label it as ``out``.
      - Fallback to ``contact`` when mapping information is unavailable.
    """
    if not samples:
        return []

    frames = sorted(samples.keys())
    events: List[BallEvent] = []
    if not frames:
        return events

    prev = frames[0]
    for f in frames[1:]:
        if (f - prev) > max(1, gap_frames):
            sample = samples.get(prev)
            if sample is not None:
                events.append(_event_from_sample(sample, mapper, in_bounds_margin, ground_band_ratio))
        prev = f

    last = samples.get(frames[-1])
    if last is not None:
        events.append(_event_from_sample(last, mapper, in_bounds_margin, ground_band_ratio))
    return events


def _event_from_sample(
    sample: BallSample,
    mapper: Optional[CourtMapper],
    margin: float,
    ground_band_ratio: float,
) -> BallEvent:
    kind = "contact"
    if mapper is not None and sample.is_mapped:
        if mapper.point_in_bounds(sample.x, sample.y, margin=margin):
            kind = "ground"
        else:
            _, court_h = mapper.size
            band = max(0.0, min(0.5, float(ground_band_ratio)))
            ground_thresh = court_h * (1.0 - band)
            if sample.y >= ground_thresh:
                kind = "out"
            else:
                kind = "contact"
    confidence = float(sample.quality or sample.confidence or 0.0)
    return BallEvent(
        frame=sample.frame,
        kind=kind,
        confidence=confidence,
        by_team=sample.team_name,
        court_side=sample.court_side,
    )


__all__ = ["CourtMapper", "build_ball_samples", "detect_ball_events"]
