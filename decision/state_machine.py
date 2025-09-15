from typing import Dict, List, Any, Optional, Tuple


class MatchStateMachine:
    """
    Match state machine with explicit phases and context-aware filtering.
    Phases:
      - SERVING(team): just served, rally about to start
      - ATTACKING(team): current attacking side inferred from events
      - NEUTRAL: no strong context yet
    Uses CrossValidator events to transition and to gate message plausibility.
    """

    def __init__(self, teamA: str, teamB: str, serve_cooldown_frames: int = 20):
        self.teamA = teamA
        self.teamB = teamB
        self.cooldown = int(max(0, serve_cooldown_frames))
        self.score: Dict[str, int] = {teamA: 0, teamB: 0}
        self.last_serve_team: Optional[str] = None
        self.last_serve_start: int = -10**9
        self.recent_msgs: Dict[str, Tuple[str, int]] = {}
        # State
        self.phase: str = "NEUTRAL"  # SERVING | ATTACKING | NEUTRAL
        self.phase_team: Optional[str] = None

    def _fmt(self, team: Optional[str], action: str, scored: bool = False) -> str:
        name = team if team else "Unknown"
        return f"{name} {action}{' (+1)' if scored else ''}"

    def process_frame(self, fi: int, fps: float, active_clips: Dict[str, Optional[Dict[str, Any]]], events: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        # Serve: award point at serve start
        serve_clip = active_clips.get("serve")
        if serve_clip is not None:
            team = serve_clip.get("team_name")
            s_start = int(serve_clip.get("start", -10**9))
            if team in (self.teamA, self.teamB) and fi == s_start and (fi - self.last_serve_start) >= self.cooldown:
                if self.last_serve_team is not None:
                    self.score[team] = self.score.get(team, 0) + 1
                    ttl = int(max(1, round(2.0 * fps)))
                    # attach (+1) to decisive prior clip
                    for cname in ("spike", "block", "set"):
                        c = active_clips.get(cname)
                        if c is not None and int(c.get("end", -1)) == (s_start - 1) and str(c.get("team_name")) == team:
                            self.recent_msgs[c.get("id", cname)] = (self._fmt(team, cname, True), fi + ttl)
                            break
                    else:
                        self.recent_msgs[serve_clip.get("id", "serve")] = (self._fmt(team, "serve", True), fi + ttl)
                self.last_serve_team = team
                self.last_serve_start = s_start
                # Enter SERVING phase for serving team
                self.phase = "SERVING"
                self.phase_team = team
            # Baseline message for serve: only add at start frame and replace older serve messages
            if fi == s_start:
                ttl = int(max(1, round(2.0 * fps)))
                # purge any prior serve messages to avoid multiple concurrent serves
                for key in list(self.recent_msgs.keys()):
                    msg, exp = self.recent_msgs.get(key, ("", -1))
                    if msg.endswith("serve") or " serve" in msg:
                        self.recent_msgs.pop(key, None)
                self.recent_msgs[serve_clip.get("id", "serve")] = (self._fmt(team, "serve"), fi + ttl)

        # Determine primary event (highest confidence) for transition
        primary: Optional[Dict[str, Any]] = None
        if events:
            primary = max(events, key=lambda e: float(e.get("confidence", 0.0)))
            cls = str(primary.get("class", ""))
            team = primary.get("team_name")
            if cls in ("set", "spike") and team in (self.teamA, self.teamB):
                self.phase = "ATTACKING"
                self.phase_team = team

        # Filter/gate action messages by state plausibility
        expected_attack_team: Optional[str] = None
        if self.phase == "SERVING":
            # After serve, receiver likely to set/spike
            if self.phase_team in (self.teamA, self.teamB):
                expected_attack_team = self.teamA if self.phase_team == self.teamB else self.teamB
        elif self.phase == "ATTACKING":
            expected_attack_team = self.phase_team

        # Compose messages from active clips, adjusting for plausibility
        for cname in ("set", "spike", "block"):
            c = active_clips.get(cname)
            if c is None:
                continue
            team = c.get("team_name")
            msg = self._fmt(team, cname)
            # Plausibility: attack actions expected from expected_attack_team; block from opposite
            plaus = 1.0
            if cname in ("set", "spike"):
                if expected_attack_team and team not in (expected_attack_team, None):
                    plaus = 0.6  # down-weight unexpected attack
            elif cname == "block":
                if expected_attack_team and team == expected_attack_team:
                    plaus = 0.6  # block from attacker side is unlikely
            # Only show if plausibility reasonable
            if plaus >= 0.6:
                ttl = int(max(1, round(2.0 * fps)))
                self.recent_msgs.setdefault(c.get("id", cname), (msg, fi + ttl))

        # Score line
        lines.append(f"{self.teamA} {self.score.get(self.teamA,0)} - {self.score.get(self.teamB,0)} {self.teamB}")
        # Emit messages in fixed order
        for cname in ("serve", "set", "spike", "block"):
            for key, (msg, exp) in list(self.recent_msgs.items()):
                if fi <= exp and msg and (msg.endswith(cname) or f" {cname}" in msg):
                    lines.append(msg)
        return lines

__all__ = ["MatchStateMachine"]
