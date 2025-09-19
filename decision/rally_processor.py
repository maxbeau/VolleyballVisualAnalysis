from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from decision.ball_events import CourtMapper, build_ball_samples, detect_ball_events
from decision.rally_events import action_events_from_clips, index_actions_by_frame
from decision.rally_types import ActionEvent, BallEvent, BallSample, FrameContext, Rally


@dataclass(slots=True)
class RallyInferenceConfig:
    rally_gap_frames: int = 240
    result_window_frames: int = 150
    ball_gap_frames: int = 4
    min_receiver_action_frames: int = 12


class RallyProcessor:
    """Fuse action clips, ball trajectory, and court context into rallies."""

    def __init__(
        self,
        *,
        clips: Sequence[Dict[str, object]],
        ball_tracks: Optional[Dict[int, Dict[str, float]]],
        court_timeseries: Optional[Dict[int, Sequence[Tuple[float, float]]]],
        fps: float,
        teamA: str,
        teamB: str,
        side_to_team: Optional[Dict[str, str]] = None,
        players_by_frame: Optional[Dict[int, Sequence[Dict[str, object]]]] = None,
        mapper_dims: Tuple[int, int] = (1800, 900),
        config: Optional[RallyInferenceConfig] = None,
    ) -> None:
        self.teamA = teamA
        self.teamB = teamB
        self.fps = float(fps) if fps else 30.0
        self.side_to_team = side_to_team or {}
        self.players_by_frame = players_by_frame or {}
        self.config = config or RallyInferenceConfig()

        self._mapper = CourtMapper(court_timeseries or {}, mapper_dims[0], mapper_dims[1])
        self._ball_samples: Dict[int, BallSample] = build_ball_samples(ball_tracks or {}, self.fps, self._mapper, self.side_to_team)
        self._ball_events: List[BallEvent] = detect_ball_events(
            self._ball_samples,
            gap_frames=self.config.ball_gap_frames,
            mapper=self._mapper,
        )
        self._ball_events.sort(key=lambda e: (e.frame, e.kind))

        self._action_events: List[ActionEvent] = action_events_from_clips(clips)
        self._action_index: Dict[int, List[ActionEvent]] = index_actions_by_frame(self._action_events)

        self._rallies: List[Rally] = self._build_rallies()
        self._rally_by_frame: Dict[int, Rally] = {}
        self._ball_events_by_frame: Dict[int, List[BallEvent]] = {}
        for ev in self._ball_events:
            self._ball_events_by_frame.setdefault(ev.frame, []).append(ev)
        self._populate_frame_mappings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rallies(self) -> List[Rally]:
        return self._rallies

    def rally_for_frame(self, frame: int) -> Optional[Rally]:
        return self._rally_by_frame.get(int(frame))

    def context_for_frame(self, frame: int) -> FrameContext:
        fi = int(frame)
        actions_raw = self._action_index.get(fi, [])
        active: Dict[str, ActionEvent] = {}
        for ev in actions_raw:
            cur = active.get(ev.action)
            if cur is None or ev.confidence > cur.confidence:
                active[ev.action] = ev
        rally = self._rally_by_frame.get(fi)
        ball_sample = self._ball_samples.get(fi)
        ball_samples = (ball_sample,) if ball_sample is not None else ()
        ball_events = tuple(self._ball_events_by_frame.get(fi, ()))
        players = tuple(self.players_by_frame.get(fi, ()))
        return FrameContext(
            frame_index=fi,
            fps=self.fps,
            active_actions=active,
            rally=rally,
            ball_samples=ball_samples,
            ball_events=ball_events,
            players=players,
        )

    # ------------------------------------------------------------------
    # Rally construction
    # ------------------------------------------------------------------
    def _build_rallies(self) -> List[Rally]:
        rallies: List[Rally] = []
        current: Optional[Rally] = None
        idx = 0
        for act in self._action_events:
            if act.action == "serve":
                if current is not None:
                    self._finalize_rally(current)
                    current = None
                idx += 1
                rid = f"rally_{idx:04d}"
                current = Rally(id=rid, start_frame=act.start, serve=act, receiving_team=self._opponent_team(act.team_name))
                current.append_action(act)
                rallies.append(current)
                continue
            if current is None:
                # Ignore non-serve actions before the first serve
                continue
            # If gap is too large, start a new rally implicitly
            if current.end_frame is not None and (act.start - current.end_frame) > self.config.rally_gap_frames:
                self._finalize_rally(current)
                idx += 1
                rid = f"rally_{idx:04d}"
                current = Rally(id=rid, start_frame=act.start)
                current.append_action(act)
                rallies.append(current)
                continue
            current.append_action(act)
        if current is not None:
            self._finalize_rally(current)
        for i, rally in enumerate(rallies[:-1]):
            next_rally = rallies[i + 1]
            nxt_serve = next_rally.serve
            if nxt_serve is not None and nxt_serve.team_name:
                rally.next_serve_team = nxt_serve.team_name
        self._attach_ball_events(rallies)
        for rally in rallies:
            self._annotate_rally(rally)
        return rallies

    def _finalize_rally(self, rally: Rally) -> None:
        if rally.end_frame is None and rally.actions:
            rally.end_frame = rally.actions[-1].end

    def _attach_ball_events(self, rallies: List[Rally]) -> None:
        if not rallies or not self._ball_events:
            return
        events = sorted(self._ball_events, key=lambda e: e.frame)
        ei = 0
        n = len(events)
        for idx, rally in enumerate(rallies):
            next_start = rallies[idx + 1].start_frame if (idx + 1) < len(rallies) else None
            while ei < n and events[ei].frame < rally.start_frame:
                ei += 1
            while ei < n and (next_start is None or events[ei].frame < next_start):
                rally.append_ball_event(events[ei])
                ei += 1
            if rally.ball_events:
                rally.decisive_frame = rally.ball_events[-1].frame
                rally.end_frame = max(rally.end_frame or rally.start_frame, rally.ball_events[-1].frame)

    # ------------------------------------------------------------------
    # Rally annotation / inference
    # ------------------------------------------------------------------
    def _annotate_rally(self, rally: Rally) -> None:
        for act in rally.actions:
            self._assign_ball_touch(act)
        if rally.serve is not None:
            self._infer_serve_result(rally)
        for act in rally.actions:
            if act.action == "serve":
                continue
            if act.action == "spike":
                self._infer_spike_result(rally, act)
            elif act.action == "block":
                self._infer_block_result(rally, act)
            elif act.action == "set":
                self._infer_set_result(rally, act)
        self._infer_rally_winner(rally)

    def _assign_ball_touch(self, act: ActionEvent, window: int = 6) -> None:
        if act.ball_touch_frame is not None:
            return
        best: Optional[BallSample] = None
        for frame in range(act.start - window, act.end + window + 1):
            sample = self._ball_samples.get(frame)
            if sample is None:
                continue
            if best is None or (sample.quality or 0.0) > (best.quality or 0.0):
                best = sample
        if best is not None:
            act.ball_touch_frame = best.frame

    def _infer_serve_result(self, rally: Rally) -> None:
        serve = rally.serve
        if serve is None:
            return
        receiver = rally.receiving_team or self._opponent_team(serve.team_name)
        first_ball = self._first_ball_event_after(rally, serve.end)
        first_receiver_action = self._first_action_by_team(rally, receiver, after=serve.start)
        result: Optional[str] = None
        conf = 0.0
        if first_ball is None and first_receiver_action is None:
            result = None
        elif first_receiver_action is not None and (first_ball is None or first_receiver_action.start <= first_ball.frame):
            result = "received"
            conf = 0.6
        elif first_ball is not None:
            if first_ball.kind == "out":
                result = "out"
                conf = max(conf, first_ball.confidence)
            elif first_ball.kind == "ground":
                if first_ball.by_team and first_ball.by_team == receiver:
                    result = "ace"
                elif first_ball.by_team and first_ball.by_team == serve.team_name:
                    result = "fault"
                else:
                    result = "ground"
                conf = max(conf, first_ball.confidence)
        if result is None and first_ball is None:
            # fallback on next serve by opponent
            next_serve = self._next_action(rally, serve, "serve")
            if next_serve is not None and next_serve.team_name and serve.team_name:
                if next_serve.team_name != serve.team_name:
                    result = "lost"
                    conf = 0.4
        serve.set_result(result, conf)
        rally.serve_result = result

    def _infer_spike_result(self, rally: Rally, act: ActionEvent) -> None:
        ev = self._first_ball_event_after(rally, act.end)
        result = None
        conf = 0.0
        if ev is not None:
            if ev.kind == "ground":
                if ev.by_team and ev.by_team != act.team_name:
                    result = "kill"
                elif ev.by_team and ev.by_team == act.team_name:
                    result = "error"
                conf = max(conf, ev.confidence)
            elif ev.kind == "out":
                result = "out"
                conf = max(conf, ev.confidence)
        block = self._next_action(rally, act, "block")
        if result is None and block is not None and block.team_name and block.team_name != act.team_name:
            result = "blocked"
            conf = max(conf, 0.5)
        act.set_result(result, conf)

    def _infer_block_result(self, rally: Rally, act: ActionEvent) -> None:
        ev = self._first_ball_event_after(rally, act.end)
        result = None
        conf = 0.0
        if ev is not None and ev.kind == "ground":
            if ev.by_team and act.team_name and ev.by_team != act.team_name:
                result = "stuff"
            elif ev.by_team and ev.by_team == act.team_name:
                result = "blocked_out"
            conf = max(conf, ev.confidence)
        act.set_result(result, conf)

    def _infer_set_result(self, rally: Rally, act: ActionEvent) -> None:
        next_same = self._next_action_same_team(rally, act)
        result = None
        conf = 0.0
        if next_same is not None:
            if next_same.action == "spike":
                result = "assist"
                conf = 0.6
        ev = self._first_ball_event_after(rally, act.end)
        if ev is not None and ev.kind == "ground" and ev.by_team == act.team_name:
            result = "error"
            conf = max(conf, ev.confidence)
        act.set_result(result, conf)

    def _infer_rally_winner(self, rally: Rally) -> None:
        final_event = rally.ball_events[-1] if rally.ball_events else None
        if final_event is not None and final_event.kind in ("ground", "out"):
            losing_team = final_event.by_team or self._team_from_side(final_event.court_side)
            winner = self._opponent_team(losing_team)
            rally.finalize(winner_team=winner, end_reason=final_event.kind, confidence=max(0.3, final_event.confidence))
            rally.decisive_frame = final_event.frame
            return
        last_action = rally.last_action()
        if last_action is not None and last_action.team_name is not None:
            if last_action.result in ("kill", "ace"):
                rally.finalize(winner_team=last_action.team_name, end_reason=last_action.result, confidence=max(0.5, last_action.result_confidence))
                return
            elif last_action.result in ("error", "fault", "out", "blocked_out"):
                rally.finalize(winner_team=self._opponent_team(last_action.team_name), end_reason=last_action.result, confidence=max(0.5, last_action.result_confidence))
                return
        if rally.winner_team is None and rally.next_serve_team:
            rally.finalize(winner_team=rally.next_serve_team, end_reason=rally.serve_result or "next_serve", confidence=0.3)

    # ------------------------------------------------------------------
    # Helper lookups
    # ------------------------------------------------------------------
    def _populate_frame_mappings(self) -> None:
        for rally in self._rallies:
            start = rally.start_frame
            end = rally.end_frame or rally.start_frame
            next_start = None
            # find next rally start to avoid overlap
            for r in self._rallies:
                if r.start_frame > start:
                    if next_start is None or r.start_frame < next_start:
                        next_start = r.start_frame
            limit = min(end, next_start - 1) if next_start is not None else end
            if limit < start:
                limit = start
            for f in range(start, limit + 1):
                self._rally_by_frame[f] = rally

    def _first_ball_event_after(self, rally: Rally, frame: int) -> Optional[BallEvent]:
        for ev in rally.ball_events:
            if ev.frame >= frame:
                return ev
        return None

    def _first_action_by_team(self, rally: Rally, team: Optional[str], after: int) -> Optional[ActionEvent]:
        if team is None:
            return None
        for act in rally.actions:
            if act.start < after:
                continue
            if act.team_name == team:
                return act
        return None

    def _next_action(self, rally: Rally, current: ActionEvent, action: str) -> Optional[ActionEvent]:
        seen = False
        for act in rally.actions:
            if not seen:
                if act is current:
                    seen = True
                continue
            if act.action == action:
                return act
        return None

    def _next_action_same_team(self, rally: Rally, current: ActionEvent) -> Optional[ActionEvent]:
        seen = False
        for act in rally.actions:
            if not seen:
                if act is current:
                    seen = True
                continue
            if act.team_name == current.team_name:
                return act
        return None

    def _team_from_side(self, side: Optional[str]) -> Optional[str]:
        if side is None:
            return None
        return self.side_to_team.get(side)

    def _opponent_team(self, team: Optional[str]) -> Optional[str]:
        if team is None:
            return None
        if team == self.teamA:
            return self.teamB
        if team == self.teamB:
            return self.teamA
        # fallback via side map
        for side, name in self.side_to_team.items():
            if name == team:
                other_side = "left" if side == "right" else "right"
                return self.side_to_team.get(other_side)
        return None


__all__ = ["RallyProcessor", "RallyInferenceConfig"]
