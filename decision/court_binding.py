from typing import Dict, List, Any, Optional, Tuple


def compute_side_to_team(
    *,
    clips: List[Dict[str, Any]],
    teamA: str,
    teamB: str,
    fixed_side: str = "auto",
    max_serve_considered: int = 12,
    use_weighted_vote: bool = True,
) -> Dict[str, str]:
    """Derive side_to_team mapping.
    Strategy:
      - If fixed_side is left/right, map directly.
      - Else perform majority vote over early serve clips with valid actor_side.
        Optionally weight votes by clip['side_conf'] (default True).
      - Tie-breaker: earliest serve clip.
    """
    side_to_team: Dict[str, str] = {}
    fs = (fixed_side or "auto").lower()
    if fs in ("left", "right"):
        side_to_team[fs] = teamA
        side_to_team["right" if fs == "left" else "left"] = teamB
        return side_to_team

    # Collect serve clips with actor_side
    serves = [c for c in clips if c.get("class") == "serve" and str(c.get("actor_side")) in ("left", "right")]
    if not serves:
        return side_to_team
    serves.sort(key=lambda c: int(c.get("start", 10**9)))
    if max_serve_considered > 0:
        serves = serves[:max_serve_considered]

    left_w = 0.0
    right_w = 0.0
    for c in serves:
        s = str(c.get("actor_side"))
        w = float(c.get("side_conf", 1.0)) if use_weighted_vote else 1.0
        if s == "left":
            left_w += w
        else:
            right_w += w

    if abs(left_w - right_w) < 1e-6:
        # tie -> use earliest
        first = serves[0]
        s = str(first.get("actor_side"))
    else:
        s = "left" if left_w > right_w else "right"

    side_to_team[s] = teamA
    side_to_team["right" if s == "left" else "left"] = teamB
    return side_to_team


def attach_team_name_to_clips(
    *,
    clips: List[Dict[str, Any]],
    side_to_team: Dict[str, str],
) -> None:
    """Populate clip['team_name'] based on actor_side and side_to_team mapping, if missing."""
    for c in clips:
        if c.get("team_name"):
            continue
        s = c.get("actor_side")
        if isinstance(s, str) and s in side_to_team:
            c["team_name"] = side_to_team[s]


def bind_teams_for_clips(
    *,
    clips: List[Dict[str, Any]],
    teamA: str,
    teamB: str,
    strategy: str = "earliest",
    fixed_side: str = "auto",
    window_frames: int = 240,
    max_serve_considered: int = 12,
    use_weighted_vote: bool = True,
    bind_block_to_serve: bool = False,
    bind_set_to_receive: bool = False,
    bind_block_oppose_spike: bool = False,
    block_spike_window_frames: int = 24,
    bind_block_oppose_set: bool = False,
    block_set_window_frames: int = 48,
    rally_max_gap_frames: int = 600,
) -> Dict[str, str]:
    """Assign clip['team_name'] according to chosen strategy.
    Returns the global side_to_team used (best-effort) for reference.
    """
    fs = (fixed_side or "auto").lower()
    if fs in ("left", "right"):
        side_to_team = {fs: teamA, ("right" if fs == "left" else "left"): teamB}
        attach_team_name_to_clips(clips=clips, side_to_team=side_to_team)
        return side_to_team

    strat = (strategy or "earliest").lower()
    serves = [c for c in clips if c.get("class") == "serve" and str(c.get("actor_side")) in ("left", "right")]
    serves.sort(key=lambda c: int(c.get("start", 10**9)))

    if strat in ("majority", "weighted_majority"):
        side_to_team = compute_side_to_team(
            clips=clips,
            teamA=teamA,
            teamB=teamB,
            fixed_side="auto",
            max_serve_considered=max_serve_considered,
            use_weighted_vote=(strat == "weighted_majority"),
        )
        if side_to_team:
            attach_team_name_to_clips(clips=clips, side_to_team=side_to_team)
        return side_to_team

    if strat == "windowed" and serves:
        # Per-clip local voting
        for clip in clips:
            s = clip.get("actor_side")
            if not isinstance(s, str):
                continue
            t0 = int(clip.get("start", 0))
            local = [sv for sv in serves if abs(int(sv.get("start", 0)) - t0) <= int(window_frames)]
            if not local:
                continue
            left_w = right_w = 0.0
            for sv in local:
                side = str(sv.get("actor_side"))
                w = float(sv.get("side_conf", 1.0)) if use_weighted_vote else 1.0
                if side == "left":
                    left_w += w
                else:
                    right_w += w
            if abs(left_w - right_w) < 1e-6:
                # tie -> use closest serve
                nearest = min(local, key=lambda sv: abs(int(sv.get("start", 0)) - t0))
                ref = str(nearest.get("actor_side"))
            else:
                ref = "left" if left_w > right_w else "right"
            side_to_team_local = {ref: teamA, ("right" if ref == "left" else "left"): teamB}
            clip["team_name"] = side_to_team_local.get(s)
        # Provide a best-effort global map for reference using earliest as fallback
        if serves:
            ref = str(serves[0].get("actor_side"))
            return {ref: teamA, ("right" if ref == "left" else "left"): teamB}
        return {}

    # Default: earliest
    if serves:
        ref = str(serves[0].get("actor_side"))
        side_to_team = {ref: teamA, ("right" if ref == "left" else "left"): teamB}
        attach_team_name_to_clips(clips=clips, side_to_team=side_to_team)
    else:
        side_to_team = {}

    # Optional rally anchoring
    serves_sorted = sorted([c for c in clips if c.get("class") == "serve" and c.get("team_name")], key=lambda c: int(c.get("start", 10**9)))
    spikes_sorted = sorted([c for c in clips if c.get("class") == "spike" and c.get("team_name")], key=lambda c: int(c.get("start", 10**9)))
    sets_sorted = sorted([c for c in clips if c.get("class") == "set" and c.get("team_name")], key=lambda c: int(c.get("start", 10**9)))
    if bind_block_to_serve or bind_set_to_receive or bind_block_oppose_spike or bind_block_oppose_set:
        for clip in clips:
            cls = str(clip.get("class", ""))
            if cls not in ("block", "set"):
                continue
            t0 = int(clip.get("start", 0))
            # Find nearest previous serve
            prev = None
            for sv in serves_sorted:
                if int(sv.get("start", 0)) <= t0:
                    prev = sv
                else:
                    break
            if prev is None:
                continue
            if (t0 - int(prev.get("start", 0))) > int(rally_max_gap_frames):
                continue
            if cls == "block":
                # 1) Oppose nearest spike within small window
                if bind_block_oppose_spike:
                    near_spike = None
                    for sp in spikes_sorted:
                        ds = abs(int(sp.get("start", 0)) - t0)
                        if ds <= int(block_spike_window_frames):
                            near_spike = sp
                            break
                    if near_spike and near_spike.get("team_name"):
                        clip["team_name"] = (teamA if near_spike.get("team_name") == teamB else teamB)
                        continue
                # 2) Oppose nearest set within larger window
                if bind_block_oppose_set:
                    near_set = None
                    for st in sets_sorted:
                        ds = abs(int(st.get("start", 0)) - t0)
                        if ds <= int(block_set_window_frames):
                            near_set = st
                            break
                    if near_set and near_set.get("team_name"):
                        clip["team_name"] = (teamA if near_set.get("team_name") == teamB else teamB)
                        continue
                # 3) Fallback: bind to serving team in this rally
                if bind_block_to_serve:
                    clip["team_name"] = prev.get("team_name")
            elif cls == "set" and bind_set_to_receive:
                # Opponent of serving team is receiver
                st = prev.get("team_name")
                if st:
                    clip["team_name"] = (teamA if st == teamB else teamB)

    return side_to_team


__all__ = [
    "compute_side_to_team",
    "attach_team_name_to_clips",
    "bind_teams_for_clips",
]
