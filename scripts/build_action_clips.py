import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

from config import settings
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from decision.court_binding import bind_teams_for_clips
from court.utils import compute_homography, standard_court_model_size, apply_homography_points
from actions.io import load_actions_by_frame
from players.io import load_players_tracks_by_frame
from actions.clips import build_action_clips


def main():
    actions_jsonl = settings.actions.DETECTIONS_JSONL
    out_jsonl = settings.actions.CLIPS_JSONL

    if not os.path.exists(actions_jsonl):
        raise FileNotFoundError(f"Actions detections JSONL not found: {actions_jsonl}")

    actions_by_frame = load_actions_by_frame(actions_jsonl)
    players_path = getattr(settings.players, "TRACKS_JSONL", "outputs/players_tracks.jsonl")
    players_by_frame = load_players_tracks_by_frame(players_path) if os.path.exists(players_path) else None
    classes = ["serve", "set", "spike", "block"]
    clips = build_action_clips(
        actions_by_frame,
        classes,
        min_conf=float(getattr(settings.teams, "ACTION_MIN_CONF", 0.25)),
        max_merge_gap=8,
        pad_start=6,
        pad_end=6,
        min_len=3,
        players_by_frame=players_by_frame,
        min_player_conf=float(getattr(settings.common, "OVERLAY_MIN_CONF", 0.1)),
        max_match_dist_px=120.0,
    )

    # Optional: enrich with actor_side and side confidence using court tracking
    court_tracking = settings.court.TRACKING_JSONL
    court_ts: Dict[int, List[Tuple[float, float]]] = {}
    if os.path.exists(court_tracking):
        try:
            with open(court_tracking, "r", encoding="utf-8") as cf:
                for line in cf:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    fi = int(rec.get("frame", -1))
                    cs = rec.get("corners")
                    if fi >= 0 and cs and isinstance(cs, list) and len(cs) >= 4:
                        pts = [(float(cs[0][0]), float(cs[0][1])), (float(cs[1][0]), float(cs[1][1])), (float(cs[2][0]), float(cs[2][1])), (float(cs[3][0]), float(cs[3][1]))]
                        court_ts[fi] = pts
        except Exception:
            court_ts = {}

    def _H_for(fi: int) -> Optional[Any]:
        if not court_ts:
            return None
        pts = court_ts.get(fi)
        if pts is None:
            # fallback: nearest previous
            ks = [k for k in court_ts.keys() if k <= fi]
            if ks:
                pts = court_ts.get(max(ks))
        if pts is None or len(pts) != 4:
            return None
        try:
            Wm, Hm = standard_court_model_size(scale_px_per_meter=100.0)
            H, _ = compute_homography(pts, dst_size=(Wm, Hm))
            return H
        except Exception:
            return None

    def _best_action_xy(frame: int, cls_name: str) -> Optional[Tuple[float, float]]:
        preds = actions_by_frame.get(frame)
        if not preds:
            return None
        best = None
        best_c = -1.0
        for p in preds:
            if str(p.get("class", "")).lower() != cls_name:
                continue
            try:
                c = float(p.get("confidence", 0.0))
            except Exception:
                c = 0.0
            if c > best_c:
                best_c = c
                best = p
        if best is None:
            return None
        x = float(best.get("x", 0.0))
        y = float(best.get("y", 0.0)) + 0.5 * float(best.get("height", 0.0))
        return x, y

    # Determine side for each clip
    if court_ts:
        Wm, Hm = standard_court_model_size(scale_px_per_meter=100.0)
        for clip in clips:
            st = int(clip["start"])
            ed = int(clip["end"])
            cls = str(clip.get("class", ""))
            vote_left = 0
            vote_right = 0
            total = 0
            actor_id = clip.get("actor_id")
            med_x = clip.get("med_btm_x")
            med_y = clip.get("med_btm_y")
            for f in range(st, ed + 1):
                H = _H_for(f)
                if H is None:
                    continue
                # Detection bottom center vote
                det_xy = _best_action_xy(f, cls)
                if det_xy is not None:
                    try:
                        tx, ty = apply_homography_points([det_xy], H)[0]
                        if tx < (Wm * 0.5):
                            vote_left += 1
                        else:
                            vote_right += 1
                        total += 1
                    except Exception:
                        pass
                # Nearest player vote (weighted)
                if players_by_frame and f in players_by_frame:
                    # pick nearest player to detection or med point
                    ax, ay = (det_xy if det_xy is not None else ((float(med_x), float(med_y)) if (med_x is not None and med_y is not None) else (None, None)))
                    if ax is not None:
                        best_d2 = None
                        best_pt = None
                        for t in players_by_frame.get(f, []):
                            try:
                                pc = float(t.get("confidence", 0.0))
                            except Exception:
                                pc = 0.0
                            if pc < float(getattr(settings.common, "OVERLAY_MIN_CONF", 0.1)):
                                continue
                            cx = float(t.get("x", 0.0))
                            cy = float(t.get("y", 0.0))
                            h = float(t.get("height", 0.0))
                            bx, by = cx, cy + 0.5 * h
                            dx, dy = bx - ax, by - ay
                            d2 = dx*dx + dy*dy
                            if best_d2 is None or d2 < best_d2:
                                best_d2 = d2
                                best_pt = (bx, by)
                        if best_pt is not None and (best_d2 is None or best_d2 <= (120.0*120.0)):
                            try:
                                tx, ty = apply_homography_points([best_pt], H)[0]
                                if tx < (Wm * 0.5):
                                    vote_left += 2  # weighted
                                else:
                                    vote_right += 2
                                total += 2
                            except Exception:
                                pass
            if total > 0:
                actor_side = "left" if vote_left >= vote_right else "right"
                side_conf = float(max(vote_left, vote_right) / total)
                clip["actor_side"] = actor_side
                clip["side_conf"] = side_conf

    # Map side to team name and attach to clips
    bind_teams_for_clips(
        clips=clips,
        teamA=getattr(settings.teams, "TEAM_A_NAME", "TeamA"),
        teamB=getattr(settings.teams, "TEAM_B_NAME", "TeamB"),
        strategy=str(getattr(settings.teams, "BIND_STRATEGY", "earliest") or "earliest"),
        fixed_side=str(getattr(settings.teams, "TEAM_A_SIDE", "auto") or "auto"),
        window_frames=int(getattr(settings.teams, "BIND_WINDOW_FRAMES", 240)),
        max_serve_considered=12,
        use_weighted_vote=True,
        bind_block_to_serve=bool(getattr(settings.teams, "BIND_BLOCK_TO_SERVE", True)),
        bind_set_to_receive=bool(getattr(settings.teams, "BIND_SET_TO_RECEIVE", True)),
        bind_block_oppose_spike=bool(getattr(settings.teams, "BIND_BLOCK_OPPOSE_SPIKE", True)),
        block_spike_window_frames=int(getattr(settings.teams, "BIND_BLOCK_OPPOSE_SPIKE_WINDOW_FRAMES", 24)),
        bind_block_oppose_set=bool(getattr(settings.teams, "BIND_BLOCK_OPPOSE_SET", True)),
        block_set_window_frames=int(getattr(settings.teams, "BIND_BLOCK_OPPOSE_SET_WINDOW_FRAMES", 48)),
        rally_max_gap_frames=int(getattr(settings.teams, "BIND_RALLY_MAX_GAP_FRAMES", 600)),
    )

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Action clips saved: {out_jsonl} | total clips: {len(clips)}")


if __name__ == "__main__":
    main()
