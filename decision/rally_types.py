from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ActionEvent:
    id: str
    action: str
    start: int
    end: int
    confidence: float
    team_name: Optional[str] = None
    team_side: Optional[str] = None
    actor_id: Optional[str] = None
    result: Optional[str] = None
    result_confidence: float = 0.0
    ball_touch_frame: Optional[int] = None

    def duration(self) -> int:
        return max(0, int(self.end) - int(self.start) + 1)

    def set_result(self, result: Optional[str], confidence: float = 0.0) -> None:
        self.result = result
        self.result_confidence = float(confidence or 0.0)


@dataclass
class BallSample:
    frame: int
    x: float
    y: float
    z: Optional[float] = None
    confidence: float = 1.0
    quality: Optional[float] = None
    court_side: Optional[str] = None
    team_name: Optional[str] = None
    is_mapped: bool = False


@dataclass
class BallEvent:
    frame: int
    kind: str
    confidence: float = 0.0
    by_team: Optional[str] = None
    court_side: Optional[str] = None


@dataclass
class Rally:
    id: str
    start_frame: int
    serve: Optional[ActionEvent] = None
    receiving_team: Optional[str] = None
    actions: List[ActionEvent] = field(default_factory=list)
    ball_events: List[BallEvent] = field(default_factory=list)
    end_frame: Optional[int] = None
    winner_team: Optional[str] = None
    end_reason: Optional[str] = None
    serve_result: Optional[str] = None
    decisive_frame: Optional[int] = None
    confidence: float = 0.0
    next_serve_team: Optional[str] = None

    def append_action(self, action: ActionEvent) -> None:
        self.actions.append(action)
        if self.serve is None and action.action == "serve":
            self.serve = action
        if self.end_frame is None or action.end > self.end_frame:
            self.end_frame = action.end

    def append_ball_event(self, event: BallEvent) -> None:
        self.ball_events.append(event)
        if self.end_frame is None or event.frame > self.end_frame:
            self.end_frame = event.frame

    def last_action(self) -> Optional[ActionEvent]:
        return self.actions[-1] if self.actions else None

    def finalize(
        self,
        *,
        winner_team: Optional[str],
        end_reason: Optional[str],
        confidence: float,
    ) -> None:
        self.winner_team = winner_team
        self.end_reason = end_reason
        self.confidence = float(confidence or 0.0)


@dataclass
class FrameContext:
    frame_index: int
    fps: float
    active_actions: Dict[str, ActionEvent]
    rally: Optional[Rally]
    ball_samples: Tuple[BallSample, ...] = tuple()
    ball_events: Tuple[BallEvent, ...] = tuple()
    players: Tuple[Dict[str, object], ...] = tuple()


__all__ = [
    "ActionEvent",
    "BallEvent",
    "BallSample",
    "FrameContext",
    "Rally",
]
