from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from decision.rally_types import ActionEvent


def _clip_confidence(clip: Dict[str, object]) -> float:
    for key in ("confidence", "mean_conf", "peak_conf"):
        val = clip.get(key)
        if val is not None:
            try:
                return float(val)
            except Exception:
                continue
    return 0.0


def action_events_from_clips(clips: Sequence[Dict[str, object]]) -> List[ActionEvent]:
    events: List[ActionEvent] = []
    for clip in clips:
        cls = str(clip.get("class", "")).strip().lower()
        if not cls:
            continue
        start = int(clip.get("start", -1))
        end = int(clip.get("end", start))
        if start < 0:
            continue
        if end < start:
            end = start
        event_id = str(clip.get("id") or f"{cls}_{start}_{end}")
        conf = _clip_confidence(clip)
        event = ActionEvent(
            id=event_id,
            action=cls,
            start=start,
            end=end,
            confidence=conf,
            team_name=clip.get("team_name") if clip.get("team_name") else None,
            team_side=clip.get("actor_side") if clip.get("actor_side") else None,
            actor_id=str(clip.get("actor_id")) if clip.get("actor_id") is not None else None,
        )
        events.append(event)
    events.sort(key=lambda ev: (ev.start, ev.end))
    return events


def index_actions_by_frame(events: Iterable[ActionEvent]) -> Dict[int, List[ActionEvent]]:
    index: Dict[int, List[ActionEvent]] = {}
    for ev in events:
        start = int(ev.start)
        end = int(ev.end)
        if end < start:
            end = start
        for fi in range(start, end + 1):
            index.setdefault(fi, []).append(ev)
    return index


__all__ = ["action_events_from_clips", "index_actions_by_frame"]
