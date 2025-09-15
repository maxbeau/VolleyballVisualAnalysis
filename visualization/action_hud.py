from typing import Dict, Any, List, Optional, Tuple


class ActionHud:
    """
    Clip-driven HUD: trusts action clips (with team_name) and renders concise messages.
    Responsibilities:
      - Award point at the start of each serve clip
      - Render short-lived messages (2s) for serve/set/spike/block
    """

    def __init__(
        self,
        settings,
        Wm: int,
        Hm: int,
        court_timeseries: Optional[Dict[int, List[Tuple[float, float]]]] = None,
        court_corners: Optional[List[Tuple[float, float]]] = None,
        action_clips: Optional[List[Dict[str, Any]]] = None,
    ):
        self.settings = settings
        self.Wm = Wm
        self.Hm = Hm
        self.court_timeseries = court_timeseries
        self.court_corners = court_corners
        # Index clips by class
        self.clips_by_class: Dict[str, List[Dict[str, Any]]] = {}
        if action_clips:
            for c in action_clips:
                cls = str(c.get("class", ""))
                if not cls:
                    continue
                self.clips_by_class.setdefault(cls, []).append(c)
            for k in list(self.clips_by_class.keys()):
                self.clips_by_class[k].sort(key=lambda d: (int(d.get("start", 0)), int(d.get("end", 0))))

        self.teamA = getattr(settings.teams, "TEAM_A_NAME", "TeamA")
        self.teamB = getattr(settings.teams, "TEAM_B_NAME", "TeamB")
        self.serve_cooldown = int(getattr(settings.teams, "SERVE_COOLDOWN_FRAMES", 20))

        self.score: Dict[str, int] = {self.teamA: 0, self.teamB: 0}
        self.last_serve_team: Optional[str] = None
        self.last_serve_frame: int = -10**9
        self.last_serve_start: int = -10**9
        self.recent_msgs: Dict[str, Tuple[str, int]] = {}

    def _fmt(self, team: Optional[str], action: str, scored: bool = False) -> str:
        name = team if team else "Unknown"
        return f"{name} {action}{' (+1)' if scored else ''}"

    def _active_clip(self, fi: int, cls_name: str) -> Optional[Dict[str, Any]]:
        cl = self.clips_by_class.get(cls_name) or []
        for c in cl:
            if int(c.get("start", -1)) <= fi <= int(c.get("end", -1)):
                return c
        return None

    def render_lines(self, fi: int, fps: float) -> List[str]:
        lines: List[str] = []
        try:
            serve_clip = self._active_clip(fi, "serve")
            set_clip = self._active_clip(fi, "set")
            spike_clip = self._active_clip(fi, "spike")
            block_clip = self._active_clip(fi, "block")

            serve_team: Optional[str] = None
            if serve_clip is not None:
                serve_team = str(serve_clip.get("team_name")) if serve_clip.get("team_name") else None
                s_start = int(serve_clip.get("start", -10**9))
                if serve_team and serve_team in (self.teamA, self.teamB) and fi == s_start and (fi - self.last_serve_start) >= self.serve_cooldown:
                    if self.last_serve_team is not None:
                        self.score[serve_team] = self.score.get(serve_team, 0) + 1
                        ttl = int(max(1, round(2.0 * fps)))
                        # Attach (+1) to closest prior decisive clip of same team
                        decisive = None
                        for cname in ("spike", "block", "set"):
                            c = self._active_clip(s_start - 1, cname)
                            if c is not None:
                                team = str(c.get("team_name")) if c.get("team_name") else None
                                if team == serve_team:
                                    decisive = (cname, c)
                                    break
                        if decisive is not None:
                            cname, c = decisive
                            self.recent_msgs[c.get("id", f"{cname}")] = (self._fmt(serve_team, cname, scored=True), fi + ttl)
                        else:
                            self.recent_msgs[serve_clip.get("id", "serve")] = (self._fmt(serve_team, "serve", scored=True), fi + ttl)
                    self.last_serve_team = serve_team
                    self.last_serve_frame = fi
                    self.last_serve_start = s_start
                ttl = int(max(1, round(2.0 * fps)))
                self.recent_msgs.setdefault(serve_clip.get("id", "serve"), (self._fmt(serve_team, "serve"), fi + ttl))

            if set_clip is not None:
                a_team = str(set_clip.get("team_name")) if set_clip.get("team_name") else None
                ttl = int(max(1, round(2.0 * fps)))
                self.recent_msgs.setdefault(set_clip.get("id", "set"), (self._fmt(a_team, "set"), fi + ttl))

            if spike_clip is not None:
                a_team = str(spike_clip.get("team_name")) if spike_clip.get("team_name") else None
                ttl = int(max(1, round(2.0 * fps)))
                self.recent_msgs.setdefault(spike_clip.get("id", "spike"), (self._fmt(a_team, "spike"), fi + ttl))

            if block_clip is not None:
                a_team = str(block_clip.get("team_name")) if block_clip.get("team_name") else None
                ttl = int(max(1, round(2.0 * fps)))
                self.recent_msgs.setdefault(block_clip.get("id", "block"), (self._fmt(a_team, "block"), fi + ttl))

            # Score line first
            lines.append(f"{self.teamA} {self.score.get(self.teamA,0)} - {self.score.get(self.teamB,0)} {self.teamB}")
            # Emit messages in fixed class order
            for cls in ("serve", "set", "spike", "block"):
                for key, (msg, exp) in list(self.recent_msgs.items()):
                    if fi <= exp and msg and (msg.endswith(cls) or f" {cls}" in msg):
                        lines.append(msg)
        except Exception:
            pass
        return lines

__all__ = ["ActionHud"]

