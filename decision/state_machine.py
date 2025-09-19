from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from decision.rally_types import ActionEvent, FrameContext


class MatchStateMachine:
    """Stateful HUD compositor that leverages rally context for messaging."""

    ACTION_ORDER = ["serve", "set", "spike", "block"]

    def __init__(
        self,
        *,
        teamA: str,
        teamB: str,
        score_flash_frames: int = 24,
        min_action_conf: float = 0.25,
        serve_cooldown_frames: int = 20,
    ) -> None:
        self.teamA = teamA
        self.teamB = teamB
        self.score: Dict[str, int] = {teamA: 0, teamB: 0}
        self.score_flash_frames = int(max(1, score_flash_frames))
        self.min_action_conf = float(min_action_conf)
        self.serve_cooldown_frames = int(max(0, serve_cooldown_frames))

        self._processed_rallies: Dict[str, int] = {}
        self._score_messages: Dict[str, Tuple[str, int]] = {}
        self._last_frame: int = -1
        self.last_serve_team: Optional[str] = None

    # ------------------------------------------------------------------
    def process(self, ctx: FrameContext) -> List[str]:
        fi = ctx.frame_index
        self._expire_score_messages(fi)
        self._update_score(ctx)
        self._update_last_serve(ctx)

        lines: List[str] = [self._score_line()]
        lines.extend(self._score_message_lines(fi))
        lines.extend(self._format_rally_lines(ctx))
        self._last_frame = fi
        return lines

    # ------------------------------------------------------------------
    def _score_line(self) -> str:
        return f"{self.teamA} {self.score.get(self.teamA, 0)} - {self.score.get(self.teamB, 0)} {self.teamB}"

    def _expire_score_messages(self, frame: int) -> None:
        expired = [key for key, (_, exp) in self._score_messages.items() if frame > exp]
        for key in expired:
            self._score_messages.pop(key, None)

    def _update_score(self, ctx: FrameContext) -> None:
        rally = ctx.rally
        if rally is None or rally.winner_team is None:
            return
        if rally.id in self._processed_rallies:
            return
        decisive_frame = rally.decisive_frame or rally.end_frame or ctx.frame_index
        if ctx.frame_index < decisive_frame:
            return

        winner = rally.winner_team
        if winner not in self.score:
            self.score[winner] = 0
        self.score[winner] += 1
        self._processed_rallies[rally.id] = ctx.frame_index

        reason = rally.end_reason or rally.serve_result or "point"
        msg = f"{winner} +1 ({reason})"
        expire = ctx.frame_index + self.score_flash_frames
        self._score_messages[f"score:{rally.id}"] = (msg, expire)

    def _update_last_serve(self, ctx: FrameContext) -> None:
        rally = ctx.rally
        if rally is None:
            return
        serve = rally.serve
        if serve is not None and serve.team_name:
            if self._last_frame < serve.start or (ctx.frame_index - serve.start) <= self.serve_cooldown_frames:
                self.last_serve_team = serve.team_name

    def _score_message_lines(self, frame: int) -> List[str]:
        msgs = [(exp, msg) for msg, exp in self._score_messages.values() if frame <= exp]
        msgs.sort(key=lambda item: item[0])
        return [msg for _, msg in msgs]

    def _format_rally_lines(self, ctx: FrameContext) -> List[str]:
        rally = ctx.rally
        if rally is None:
            return []
        fi = ctx.frame_index
        items: List[Tuple[int, int, str]] = []

        for act in rally.actions:
            if act.start > fi:
                continue
            if act.confidence < self.min_action_conf and not act.result:
                continue
            status = None
            if act.end <= fi:
                status = act.result
            elif act.start <= fi <= act.end:
                status = "in play"
            team = act.team_name or "Unknown"
            text = f"{team} {act.action}"
            if status:
                text += f" ({status})"
            items.append((act.end, 0, text))

        for bev in rally.ball_events:
            if bev.frame > fi:
                continue
            text = f"Ball {bev.kind}"
            if bev.by_team:
                text += f" [{bev.by_team}]"
            elif bev.court_side:
                text += f" [{bev.court_side}]"
            items.append((bev.frame, 1, text))

        items.sort(key=lambda item: (item[0], item[1]))

        lines: List[str] = []
        for idx, (_, _, text) in enumerate(items, start=1):
            lines.append(f"{idx}. {text}")

        return lines

    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame_index: int,
        fps: float,
        active_clips: Optional[Dict[str, Optional[Dict[str, object]]]] = None,
        events: Optional[List[Dict[str, object]]] = None,
    ) -> List[str]:
        """Compatibility shim used by legacy overlay code paths.

        Builds a minimal FrameContext so existing HUD logic can operate
        even when full rally inference is not yet wired in. The context is
        intentionally sparse; score updates will only occur once a proper
        rally is attached (handled in newer pipelines).
        """

        ctx = FrameContext(
            frame_index=int(frame_index),
            fps=float(fps) if fps else 0.0,
            active_actions={},
            rally=None,
            ball_samples=(),
            ball_events=(),
            players=(),
        )

        return self.process(ctx)


__all__ = ["MatchStateMachine"]
