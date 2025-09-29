import os
import json
import bisect
import cv2
import math
from typing import Dict, Any, List, Optional, Tuple
from types import SimpleNamespace
import numpy as np
from court.utils import standard_court_model_size, compute_homography, apply_homography_points, corners_from_prediction
from visualization.mini_birdseye import MiniBirdseyeOverlay
from visualization.hud import draw_boxed_text
from visualization.overlay_utils import to_tlbr_from_xywh
from visualization.overlay_utils import has_box as _has_box
from visualization.overlay_utils import action_color as _action_color
from court.orientation import decide_orientation as decide_court_orientation
# from config import settings # Refactored: No longer using global settings
from core.utils import ensure_dir
from analysis.smoothing import kalman_rts_smooth
from analysis.kinematic_filter import filter_by_kinematics
from analysis.trajectory import load_best_ball_per_frame
from actions.io import load_actions_by_frame, load_action_clips
from visualization.action_hud import ActionHud
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from decision.cross_validate import CrossValidator
from decision.state_machine import MatchStateMachine
from decision.court_binding import bind_teams_for_clips
from decision.rally_processor import RallyProcessor
from players.io import load_players_tracks_by_frame


def parse_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def parse_color(s: str, default=(0, 255, 0)) -> Tuple[int, int, int]:
    try:
        parts = [int(p.strip()) for p in s.split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    except Exception:
        pass
    return default


    # legacy helpers removed; using ball.pipeline instead




def _read_court_timeseries(tracking_jsonl: str) -> Tuple[Dict[int, List[Tuple[float, float]]], Dict[int, Dict[str, Any]]]:
    """Read court tracking JSONL into (timeseries, infos)."""
    ts: Dict[int, List[Tuple[float, float]]] = {}
    infos: Dict[int, Dict[str, Any]] = {}
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
                if fi >= 0 and cs and isinstance(cs, list) and len(cs) >= 4:
                    pts = [
                        (float(cs[0][0]), float(cs[0][1])),
                        (float(cs[1][0]), float(cs[1][1])),
                        (float(cs[2][0]), float(cs[2][1])),
                        (float(cs[3][0]), float(cs[3][1])),
                    ]
                    ts[fi] = pts
                    if isinstance(rec.get("info"), dict):
                        infos[fi] = rec["info"]
    except Exception:
        return {}, {}
    return ts, infos


def _simple_ball_tracks(
    jsonl_path: str,
    min_confidence: float,
    frame_size: Tuple[int, int],
    allowed_classes: Optional[List[str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Build a lightweight per-frame ball track map from detections."""
    allowed = allowed_classes or ["ball", "volleyball"]
    best = load_best_ball_per_frame(jsonl_path, allowed)
    width, height = frame_size
    tracks: Dict[int, Dict[str, Any]] = {}
    for frame_idx, pred in best.items():
        try:
            conf = float(pred.get("confidence", 0.0))
            if conf < min_confidence:
                continue
            cx = float(pred.get("x", 0.0))
            cy = float(pred.get("y", 0.0))
            w = float(pred.get("width", 0.0))
            h = float(pred.get("height", 0.0))
            if w <= 0.0 or h <= 0.0:
                continue
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            # Clamp to frame bounds to avoid drawing outside frame
            x1 = max(0.0, min(x1, max(0.0, width - w)))
            y1 = max(0.0, min(y1, max(0.0, height - h)))
            tracks[int(frame_idx)] = {
                "box": (x1, y1, w, h),
                "confidence": conf,
                "raw": pred,
            }
        except Exception:
            continue
    return tracks


class MiniCourtOverlay:
    """Helper to render the miniature court overlay with configuration defaults."""

    def __init__(self, cfg: Dict[str, Any], meta_path: str) -> None:
        court_cfg = cfg.get("court", {}) or {}
        colors = {
            "border": tuple(court_cfg.get("color", (255, 255, 255))),
            "center": tuple(court_cfg.get("center_color", (200, 200, 200))),
            "attack": tuple(court_cfg.get("attack_color", (200, 200, 200))),
        }
        thickness = int(court_cfg.get("thickness", 2))
        placement = court_cfg.get("mini_placement", "top-right")
        scale = float(court_cfg.get("mini_scale", 0.24))
        margin = int(court_cfg.get("mini_margin", 12))
        show_label = bool(court_cfg.get("mini_show_label", True))
        draw_poly = bool(court_cfg.get("mini_draw_poly", True))

        self.renderer = MiniBirdseyeOverlay(
            colors=colors,
            thickness=thickness,
            placement=placement,
            scale=scale,
            margin=margin,
            show_label=show_label,
            draw_poly=draw_poly,
        )

        self.meta = self._load_meta(meta_path)
        self.show_teams = bool(court_cfg.get("mini_show_teams", True))
        teams_cfg = cfg.get("teams", {}) or {}
        self.team_labels = (
            {
                "left": teams_cfg.get("team_a_name"),
                "right": teams_cfg.get("team_b_name"),
            }
            if self.show_teams
            else None
        )
        players_cfg = cfg.get("players", {}) or {}
        players_color = players_cfg.get("color", (0, 165, 255))
        if isinstance(players_color, list):
            players_color = tuple(players_color)
        self.players_color = tuple(players_color)

    @staticmethod
    def _load_meta(meta_path: str) -> Dict[str, Any]:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def draw(
        self,
        frame: np.ndarray,
        frame_idx: int,
        court_ts: Dict[int, List[Tuple[float, float]]],
        players_by_frame: Dict[int, List[Dict[str, Any]]],
        ball_tracks: Dict[int, Dict[str, Any]],
    ) -> None:
        orientation = self.meta.get("orientation", "horizontal")
        model_info = self.meta.get("model_size", {}) or {}
        model_size = (
            int(model_info.get("w", 1800)),
            int(model_info.get("h", 900)),
        )
        corners = court_ts.get(frame_idx)

        players_xy: Optional[List[Tuple[float, float]]] = None
        if frame_idx in players_by_frame:
            try:
                players_xy = []
                for player in players_by_frame[frame_idx]:
                    cx = float(player.get("x", 0.0))
                    cy = float(player.get("y", 0.0))
                    h = float(player.get("height", 0.0))
                    # Use the bottom-center of the player box for more accurate court projection
                    players_xy.append((cx, cy + h / 2.0))
            except Exception:
                players_xy = None

        self.renderer.render(
            frame,
            orientation=orientation,
            corners=corners,
            model_size=model_size,
            players_xy=players_xy,
            players_color=self.players_color,
            team_labels=self.team_labels,
        )


class ActionHudOverlay:
    """Simple renderer wrapper around ActionHud to draw text lines."""

    def __init__(self, hud: ActionHud, position: Tuple[int, int] = (12, 36), line_gap: int = 24) -> None:
        self.hud = hud
        self.position = position
        self.line_gap = line_gap

    def draw(self, frame: np.ndarray, frame_idx: int, fps: float) -> None:
        try:
            lines = self.hud.render_lines(frame_idx, fps)
            x, y = self.position
            for line in lines:
                draw_boxed_text(frame, line, (x, y))
                y += self.line_gap
        except Exception:
            pass


def pred_with_kalman_or_hold(
    frames: List[int],
    best: Dict[int, Dict[str, Any]],
    smoothed: Dict[int, Dict[str, Any]],
    i: int,
    hold_mode: str,
    hold_ttl: int,
) -> Optional[Dict[str, Any]]:
    # Prefer Kalman+RTS smoothed output if available
    if i in smoothed:
        return smoothed[i].copy()

    # Otherwise, apply hold fallback based on configuration using raw observations
    if not frames:
        return None
    pos = bisect.bisect_left(frames, i)
    prev_idx = frames[pos - 1] if pos > 0 else None
    next_idx = frames[pos] if pos < len(frames) else None

    hold_mode = (hold_mode or "prev").lower()
    if hold_mode not in ("prev", "next", "both", "none"):
        hold_mode = "prev"

    # Prefer prev hold
    if hold_mode in ("prev", "both") and prev_idx is not None:
        if (i - prev_idx) <= hold_ttl:
            p = best[prev_idx].copy()
            p["_hold"] = True
            return p
    # Optionally allow next hold (forward)
    if hold_mode in ("next", "both") and next_idx is not None:
        if (next_idx - i) <= hold_ttl:
            p = best[next_idx].copy()
            p["_hold"] = True
            return p
    return None




def run_overlay(
    *,
    cfg: Dict[str, Any],
    video_path: str,
    ball_detections_jsonl: str,
    court_tracking_jsonl: str,
    players_tracks_jsonl: str,
    actions_detections_jsonl: str,
    actions_clips_jsonl: str,
    court_tracking_meta_json: str,
) -> None:
    """Main entrypoint for generating the full overlay video."""
    output_dir = os.path.dirname(actions_clips_jsonl)
    ensure_dir(output_dir)
    
    # --- Load all data ---
    court_ts, court_infos = _read_court_timeseries(court_tracking_jsonl)
    players_by_frame = load_players_tracks_by_frame(players_tracks_jsonl)
    actions_by_frame = load_actions_by_frame(actions_detections_jsonl)
    
    # --- Initialize video streams ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    min_ball_conf = float(cfg.get("ball", {}).get("min_confidence", 0.2))
    ball_tracks_raw = _simple_ball_tracks(ball_detections_jsonl, min_ball_conf, (width, height))
    
    # --- Apply kinematic filtering to ball detections ---
    ball_cfg = cfg.get("ball", {}) or {}
    kinematic_cfg = ball_cfg.get("kinematic_filter", {}) or {}
    filter_cfg = ball_cfg.get("filter", {}) or {}

    if filter_cfg.get("kinematic_filter_enable", True):
        print("Applying kinematic filter to ball detections...")
        ball_tracks = filter_by_kinematics(
            ball_tracks_raw,
            fps=fps,
            max_speed_px_per_s=kinematic_cfg.get("max_speed_px_per_s", 3000.0),
            max_accel_px_per_s2=kinematic_cfg.get("max_accel_px_per_s2", 6000.0),
            max_dir_change_deg=kinematic_cfg.get("max_dir_change_deg", 45.0),
            max_size_change_frac_per_s=kinematic_cfg.get("max_size_change_frac_per_s", 10.0),
            static_filter_enable=kinematic_cfg.get("static_filter_enable", True),
            static_min_speed_px_per_s=kinematic_cfg.get("static_min_speed_px_per_s", 20.0),
            static_min_frames=kinematic_cfg.get("static_min_frames", 4),
            enable_speed_gate=kinematic_cfg.get("enable_speed_gate", True),
            enable_accel_gate=kinematic_cfg.get("enable_accel_gate", True),
            enable_dir_gate=kinematic_cfg.get("enable_dir_gate", True),
            enable_size_gate=kinematic_cfg.get("enable_size_gate", True),
            dyn_enable=kinematic_cfg.get("dyn_enable", True),
            dyn_min_mult=kinematic_cfg.get("dyn_min_mult", 0.5),
            dyn_max_mult=kinematic_cfg.get("dyn_max_mult", 2.0),
        )
        print(f"Kinematic filter complete. Kept {len(ball_tracks)} of {len(ball_tracks_raw)} detections.")
    else:
        ball_tracks = ball_tracks_raw

    output_video_path = os.path.join(output_dir, cfg.get("output_video_path", "full_overlay.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*cfg.get("codec", "mp4v"))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Initialize overlays ---
    mini_court = MiniCourtOverlay(cfg, court_tracking_meta_json)

    model_info = mini_court.meta.get("model_size", {}) if hasattr(mini_court, "meta") else {}
    Wm = int(model_info.get("w", 1800))
    Hm = int(model_info.get("h", 900))
    teams_cfg = cfg.get("teams", {}) or {}
    hud_settings = SimpleNamespace(
        teams=SimpleNamespace(
            TEAM_A_NAME=teams_cfg.get("team_a_name", "TeamA"),
            TEAM_B_NAME=teams_cfg.get("team_b_name", "TeamB"),
            SERVE_COOLDOWN_FRAMES=teams_cfg.get("serve_cooldown_frames", 20),
        )
    )
    try:
        action_clips = load_action_clips(actions_clips_jsonl)
    except Exception:
        action_clips = None
    action_hud_core = ActionHud(
        hud_settings,
        Wm,
        Hm,
        court_timeseries=court_ts,
        action_clips=action_clips,
    )
    action_hud = ActionHudOverlay(action_hud_core)

    # --- Main processing loop ---
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Draw court
        if frame_idx in court_ts:
            corners = np.array(court_ts[frame_idx], dtype=np.int32)
            cv2.polylines(frame, [corners], isClosed=True, color=(255, 255, 255), thickness=2)

        # Draw ball
        if frame_idx in ball_tracks:
            track = ball_tracks[frame_idx]
            x, y, w, h = track['box']
            tl = (int(x), int(y))
            br = (int(x + w), int(y + h))
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)

        # Draw players
        if frame_idx in players_by_frame:
            for player in players_by_frame[frame_idx]:
                cx = float(player.get('x', 0.0))
                cy = float(player.get('y', 0.0))
                w = float(player.get('width', 0.0))
                h = float(player.get('height', 0.0))
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                track_id = player.get('id') or player.get('track_id')
                if track_id is not None:
                    draw_boxed_text(frame, f"P{int(track_id)}", (int(x1), int(y1)))

        # Draw actions
        if frame_idx in actions_by_frame:
            for action in actions_by_frame[frame_idx]:
                cx = float(action.get('x', 0.0))
                cy = float(action.get('y', 0.0))
                w = float(action.get('width', 0.0))
                h = float(action.get('height', 0.0))
                x1 = int(cx - w / 2.0)
                y1 = int(cy - h / 2.0)
                x2 = int(x1 + w)
                y2 = int(y1 + h)
                label = action.get('class') or action.get('label', '')
                if label and label.strip().lower() in {"ball", "volleyball"}:
                    continue
                color = _action_color(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_boxed_text(frame, label, (x1, y1), color=color)

        # Draw mini court
        mini_court.draw(frame, frame_idx, court_ts, players_by_frame, ball_tracks)
        
        # Draw action HUD
        action_hud.draw(frame, frame_idx, fps)

        # Draw main HUD (frame count, etc.)
        from .hud import draw_hud
        draw_hud(frame, fps, frame_idx, total_frames)

        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Processing overlay frame {frame_idx}/{total_frames}")
        
        frame_idx += 1

    # --- Cleanup ---
    cap.release()
    out.release()
    print(f"Overlay video saved to {output_video_path}")

__all__ = ["run_overlay"]
