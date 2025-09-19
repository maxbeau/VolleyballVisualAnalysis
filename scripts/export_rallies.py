import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from config import settings
from actions.io import load_action_clips, load_actions_by_frame
from actions.clips import build_action_clips
from players.io import load_players_tracks_by_frame
from decision.court_binding import bind_teams_for_clips
from decision.rally_processor import RallyProcessor, RallyInferenceConfig
from ball.pipeline import build_ball_tracks, parse_frame_spec


def _read_court_timeseries(tracking_jsonl: str) -> Dict[int, List[Tuple[float, float]]]:
    timeseries: Dict[int, List[Tuple[float, float]]] = {}
    if not tracking_jsonl or not os.path.exists(tracking_jsonl):
        return timeseries
    try:
        with open(tracking_jsonl, "r", encoding="utf-8") as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                fi = int(rec.get("frame", -1))
                cs = rec.get("corners")
                if fi >= 0 and cs and isinstance(cs, Sequence) and len(cs) >= 4:
                    pts = [
                        (float(cs[0][0]), float(cs[0][1])),
                        (float(cs[1][0]), float(cs[1][1])),
                        (float(cs[2][0]), float(cs[2][1])),
                        (float(cs[3][0]), float(cs[3][1])),
                    ]
                    timeseries[fi] = pts
    except Exception:
        return {}
    return timeseries


def _load_clips(actions_clips_path: str, actions_detections_path: str, players_tracks_path: Optional[str]) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []

    if actions_clips_path and os.path.exists(actions_clips_path):
        clips = load_action_clips(actions_clips_path)
        return clips

    if not actions_detections_path or not os.path.exists(actions_detections_path):
        return []

    actions_by_frame = load_actions_by_frame(actions_detections_path)
    players_by_frame = load_players_tracks_by_frame(players_tracks_path) if (players_tracks_path and os.path.exists(players_tracks_path)) else None

    clips = build_action_clips(
        actions_by_frame,
        classes=["serve", "set", "spike", "block"],
        min_conf=float(getattr(settings.teams, "ACTION_MIN_CONF", 0.25)),
        max_merge_gap=8,
        pad_start=6,
        pad_end=6,
        min_len=3,
        players_by_frame=players_by_frame,
        min_player_conf=float(getattr(settings.common, "OVERLAY_MIN_CONF", 0.1)),
        max_match_dist_px=120.0,
    )

    return clips


def _scoreboard_for_rallies(rallies: Sequence[Any], team_a: str, team_b: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    score = {team_a: 0, team_b: 0}
    timeline: List[Dict[str, Any]] = []
    for rally in rallies:
        winner = rally.winner_team
        if winner not in score:
            score[winner] = 0
        if winner is not None:
            score[winner] += 1
        timeline.append({
            "rally_id": rally.id,
            "winner": winner,
            "score": {team: score.get(team, 0) for team in (team_a, team_b)},
            "decisive_frame": rally.decisive_frame,
            "end_reason": rally.end_reason,
        })
    return score, timeline


def export_rallies(output_path: Optional[str] = None) -> str:
    video_path = settings.common.VIDEO_PATH
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    allowed = ["ball", "volleyball"]
    ball_jsonl = settings.ball.DETECTIONS_JSONL
    ball_tracks: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(ball_jsonl):
        best, _dbg = build_ball_tracks(
            jsonl_path=ball_jsonl,
            allowed_classes=allowed,
            fps=fps,
            settings=settings,
            img_wh=(width, height),
        )
        excludes = set(parse_frame_spec(settings.ball.EXCLUDE_FRAMES))
        if excludes:
            ball_tracks = {f: det for f, det in best.items() if f not in excludes}
        else:
            ball_tracks = best
    else:
        print(f"Ball detections missing: {ball_jsonl}")

    court_timeseries = _read_court_timeseries(settings.court.TRACKING_JSONL)

    players_tracks_path = getattr(settings.players, "TRACKS_JSONL", "")
    players_by_frame = load_players_tracks_by_frame(players_tracks_path) if (players_tracks_path and os.path.exists(players_tracks_path)) else None

    clips = _load_clips(settings.actions.CLIPS_JSONL, settings.actions.DETECTIONS_JSONL, players_tracks_path)
    clips = clips or []

    # Ensure clips carry team names using the same binding logic as overlay
    side_to_team_map = bind_teams_for_clips(
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

    rally_proc = RallyProcessor(
        clips=clips,
        ball_tracks=ball_tracks,
        court_timeseries=court_timeseries,
        fps=fps,
        teamA=getattr(settings.teams, "TEAM_A_NAME", "TeamA"),
        teamB=getattr(settings.teams, "TEAM_B_NAME", "TeamB"),
        side_to_team=side_to_team_map,
        players_by_frame=players_by_frame,
        mapper_dims=(1800, 900),
        config=RallyInferenceConfig(),
    )

    rallies = rally_proc.rallies()

    team_a = getattr(settings.teams, "TEAM_A_NAME", "TeamA")
    team_b = getattr(settings.teams, "TEAM_B_NAME", "TeamB")
    score_final, score_timeline = _scoreboard_for_rallies(rallies, team_a, team_b)

    rallies_payload: List[Dict[str, Any]] = []
    for rally in rallies:
        serve = rally.serve
        actions_payload = [
            {
                "id": act.id,
                "action": act.action,
                "team": act.team_name,
                "team_side": act.team_side,
                "actor_id": act.actor_id,
                "start": act.start,
                "end": act.end,
                "confidence": act.confidence,
                "result": act.result,
                "result_confidence": act.result_confidence,
                "ball_touch_frame": act.ball_touch_frame,
            }
            for act in rally.actions
        ]

        events_payload = [
            {
                "frame": evt.frame,
                "kind": evt.kind,
                "confidence": evt.confidence,
                "by_team": evt.by_team,
                "court_side": evt.court_side,
            }
            for evt in rally.ball_events
        ]

        rallies_payload.append(
            {
                "id": rally.id,
                "start_frame": rally.start_frame,
                "end_frame": rally.end_frame,
                "serve_team": serve.team_name if serve else None,
                "serve_result": rally.serve_result,
                "receiving_team": rally.receiving_team,
                "winner_team": rally.winner_team,
                "end_reason": rally.end_reason,
                "decisive_frame": rally.decisive_frame,
                "confidence": rally.confidence,
                "actions": actions_payload,
                "ball_events": events_payload,
            }
        )

    payload = {
        "video": {
            "path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "frame_size": [width, height],
        },
        "teams": {
            "teamA": team_a,
            "teamB": team_b,
        },
        "score_final": score_final,
        "score_timeline": score_timeline,
        "rallies": rallies_payload,
    }

    out_dir = settings.common.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = output_path or os.path.join(out_dir, "rallies_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Rallies summary saved to {out_path} | total rallies: {len(rallies_payload)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rally summaries inferred from detections and clips")
    parser.add_argument("--output", help="Optional output path (defaults to outputs/<video>/rallies_summary.json)")
    args = parser.parse_args()
    export_rallies(args.output)


if __name__ == "__main__":
    main()
