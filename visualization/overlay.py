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


def _load_ball_trajectory(
    jsonl_path: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads the final, smoothed ball trajectory data and segmentation metadata."""
    tracks: Dict[int, Dict[str, Any]] = {}
    segments_meta: Dict[int, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                try:
                    frame_idx = int(record["frame"])
                    cx = float(record.get("img_x", 0.0))
                    cy = float(record.get("img_y", 0.0))
                    w = float(record.get("w", 0.0) or 0.0)
                    h = float(record.get("h", 0.0) or 0.0)
                    box = (cx - w / 2.0, cy - h / 2.0, w, h)
                    segment_id = record.get("segment_id")
                    touch_event = record.get("touch_event")
                    track_rec = {
                        "box": box,
                        "confidence": float(record.get("confidence", 0.0) or 0.0),
                        "raw": record,
                        "segment_id": int(segment_id) if segment_id is not None else None,
                        "touch_event": touch_event,
                        "center": (cx, cy),
                    }
                    tracks[frame_idx] = track_rec

                    if track_rec["segment_id"] is not None:
                        seg_id = track_rec["segment_id"]
                        meta = segments_meta.setdefault(
                            seg_id,
                            {
                                "start_frame": None,
                                "end_frame": None,
                                "duration_sec": None,
                                "change_reason": None,
                                "change_score": None,
                            },
                        )
                        if record.get("segment_is_start"):
                            meta["start_frame"] = frame_idx
                        if record.get("segment_is_end"):
                            meta["end_frame"] = frame_idx
                            if record.get("segment_change_reason"):
                                meta["change_reason"] = record.get("segment_change_reason")
                            if record.get("segment_change_score") is not None:
                                meta["change_score"] = record.get("segment_change_score")
                        if record.get("segment_duration_sec") is not None:
                            meta["duration_sec"] = record.get("segment_duration_sec")

                    if isinstance(touch_event, dict):
                        evt = dict(touch_event)
                        evt["frame"] = frame_idx
                        events.append(evt)
                except (KeyError, TypeError, ValueError):
                    continue
    except FileNotFoundError:
        print(f"Warning: Ball trajectory file not found at {jsonl_path}")
        return {}, {}, []
    return tracks, segments_meta, events


def _draw_net_overlay(
    frame: np.ndarray,
    net_info: Optional[Dict[str, Any]],
    *,
    color: Tuple[int, int, int],
    thickness: int,
    fill_alpha: float,
    post_radius: int,
) -> None:
    if not net_info:
        return

    polygon = net_info.get("polygon")
    alpha = float(max(0.0, min(1.0, fill_alpha)))
    try:
        if polygon and len(polygon) >= 4:
            pts = np.array([[float(p[0]), float(p[1])] for p in polygon[:4]], dtype=np.float32)
            pts_i = np.round(pts).astype(np.int32)
            if alpha > 0.0:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts_i], color)
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
            cv2.polylines(frame, [pts_i], isClosed=True, color=color, thickness=max(1, int(thickness)), lineType=cv2.LINE_AA)
    except Exception:
        pass

    if post_radius > 0:
        base_pts = net_info.get("base") or []
        for pt in base_pts[:2]:
            try:
                bx, by = int(round(float(pt[0]))), int(round(float(pt[1])))
                cv2.circle(frame, (bx, by), int(max(1, post_radius)), color, -1, lineType=cv2.LINE_AA)
            except Exception:
                continue


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






def run_overlay(
    *,
    cfg: Dict[str, Any],
    video_path: str,
    ball_trajectory_jsonl: str,
    court_tracking_jsonl: str,
    players_tracks_jsonl: Optional[str] = None,
    actions_detections_jsonl: Optional[str] = None,
    actions_clips_jsonl: str,
    court_tracking_meta_json: str,
) -> None:
    """Main entrypoint for generating the full overlay video."""
    output_dir = os.path.dirname(actions_clips_jsonl)
    ensure_dir(output_dir)
    
    # --- Load all data ---
    court_ts, court_infos = _read_court_timeseries(court_tracking_jsonl)
    players_cfg = cfg.get("players", {}) or {}
    players_enabled = bool(players_cfg.get("enable", True))
    players_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    if players_enabled and players_tracks_jsonl:
        try:
            players_by_frame = load_players_tracks_by_frame(players_tracks_jsonl)
        except FileNotFoundError:
            players_by_frame = {}
        except Exception:
            players_by_frame = {}
    else:
        players_enabled = False

    actions_cfg = cfg.get("actions", {}) or {}
    actions_enabled = bool(actions_cfg.get("enable", True))
    actions_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    if actions_enabled and actions_detections_jsonl:
        try:
            actions_by_frame = load_actions_by_frame(actions_detections_jsonl)
        except FileNotFoundError:
            actions_by_frame = {}
        except Exception:
            actions_by_frame = {}
    else:
        actions_enabled = False
    
    # --- Initialize video streams ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ball_tracks, segments_meta, _ = _load_ball_trajectory(ball_trajectory_jsonl)

    ball_cfg = cfg.get("ball", {}) or {}
    default_ball_color = tuple(int(v) for v in ball_cfg.get("color", (0, 255, 0)))
    default_ball_thickness = int(ball_cfg.get("thickness", 2))
    seg_viz_cfg = ball_cfg.get("segmentation_viz", {}) or {}
    seg_enable = bool(seg_viz_cfg.get("enable", True)) and bool(ball_tracks)
    seg_palette = seg_viz_cfg.get("palette") or [
        (244, 67, 54),
        (0, 188, 212),
        (76, 175, 80),
        (255, 235, 59),
        (156, 39, 176),
        (33, 150, 243),
        (255, 152, 0),
        (121, 85, 72),
        (205, 220, 57),
        (63, 81, 181),
    ]
    seg_palette = [tuple(int(v) for v in color) for color in seg_palette]
    seg_tail_length = int(seg_viz_cfg.get("tail_length_frames", 32))
    seg_tail_thickness = int(seg_viz_cfg.get("tail_thickness", 3))
    seg_event_radius = int(seg_viz_cfg.get("event_marker_radius", 10))
    seg_event_color = tuple(int(v) for v in seg_viz_cfg.get("event_marker_color", (255, 255, 255)))
    seg_label_color = tuple(int(v) for v in seg_viz_cfg.get("label_color", (240, 240, 240)))
    seg_label_bg = tuple(int(v) for v in seg_viz_cfg.get("label_bg", (0, 0, 0)))
    seg_label_scale = float(seg_viz_cfg.get("label_scale", 0.55))

    segment_color_map: Dict[int, Tuple[int, int, int]] = {}
    if seg_enable:
        ordered_segment_ids = sorted(segments_meta.keys())
        for idx, seg_id in enumerate(ordered_segment_ids):
            segment_color_map[seg_id] = seg_palette[idx % len(seg_palette)] if seg_palette else (0, 200, 0)

    segment_tails: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

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
    action_hud: Optional[ActionHudOverlay] = None
    if actions_enabled:
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
    last_net_state: Optional[Dict[str, Any]] = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Draw court
        if frame_idx in court_ts:
            corners = np.array(court_ts[frame_idx], dtype=np.int32)
            court_cfg = cfg.get("court", {}) or {}
            court_color = tuple(int(v) for v in court_cfg.get("color", (255, 255, 255)))
            thickness = int(court_cfg.get("thickness", 2))
            cv2.polylines(frame, [corners], isClosed=True, color=court_color, thickness=thickness)

            net_cfg = court_cfg.get("net", {}) or {}
            info_frame = court_infos.get(frame_idx)
            net_payload = info_frame.get("net") if isinstance(info_frame, dict) else None
            if isinstance(net_payload, dict):
                last_net_state = net_payload
            if net_cfg.get("enable", True):
                net_to_draw = net_payload or last_net_state
                if isinstance(net_to_draw, dict):
                    net_color = tuple(int(v) for v in net_cfg.get("color", (255, 255, 255)))
                    net_thickness = int(net_cfg.get("thickness", 2))
                    net_fill_alpha = float(net_cfg.get("fill_alpha", 0.18))
                    net_post_radius = int(net_cfg.get("post_radius", 4))
                    _draw_net_overlay(
                        frame,
                        net_to_draw,
                        color=net_color,
                        thickness=net_thickness,
                        fill_alpha=net_fill_alpha,
                        post_radius=net_post_radius,
                    )

        # Draw ball and segmentation cues
        if frame_idx in ball_tracks:
            track = ball_tracks[frame_idx]
            x, y, w, h = track['box']
            tl = (int(x), int(y))
            br = (int(x + w), int(y + h))
            segment_id = track.get("segment_id")
            ball_color = (
                segment_color_map.get(segment_id, default_ball_color)
                if seg_enable and segment_id is not None
                else default_ball_color
            )
            ball_thickness = default_ball_thickness
            cv2.rectangle(frame, tl, br, ball_color, ball_thickness)

            if seg_enable and segment_id is not None:
                center = track.get("center")
                if center:
                    cx, cy = float(center[0]), float(center[1])
                    tail = segment_tails.setdefault(segment_id, [])
                    tail.append((frame_idx, (cx, cy)))
                    if seg_tail_length > 0:
                        while tail and frame_idx - tail[0][0] > seg_tail_length:
                            tail.pop(0)
                    if len(tail) >= 2 and seg_tail_thickness > 0:
                        pts = np.array(
                            [(int(pt[0]), int(pt[1])) for _, pt in tail if pt is not None],
                            dtype=np.int32,
                        )
                        if pts.size >= 4:
                            cv2.polylines(
                                frame,
                                [pts],
                                False,
                                ball_color,
                                seg_tail_thickness,
                                lineType=cv2.LINE_AA,
                            )

                    draw_boxed_text(
                        frame,
                        f"S{segment_id}",
                        (tl[0], max(0, tl[1] - 6)),
                        color=seg_label_color,
                        bg=seg_label_bg,
                        scale=seg_label_scale,
                        thickness=1,
                    )

                    touch_event = track.get("touch_event")
                    if isinstance(touch_event, dict):
                        center_px = (int(cx), int(cy))
                        if seg_event_radius > 0:
                            cv2.circle(
                                frame,
                                center_px,
                                seg_event_radius,
                                seg_event_color,
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )
                        label = touch_event.get("reason")
                        if label:
                            draw_boxed_text(
                                frame,
                                label,
                                (center_px[0] + seg_event_radius + 4, center_px[1] - seg_event_radius - 4),
                                color=seg_label_color,
                                bg=seg_label_bg,
                                scale=max(0.45, seg_label_scale - 0.05),
                                thickness=1,
                            )

        # Draw players
        if players_enabled and frame_idx in players_by_frame:
            for player in players_by_frame[frame_idx]:
                cx = float(player.get('x', 0.0))
                cy = float(player.get('y', 0.0))
                w = float(player.get('width', 0.0))
                h = float(player.get('height', 0.0))
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = x1 + w
                y2 = y1 + h
                player_color = tuple(players_cfg.get("color", (255, 0, 0)))
                player_thickness = int(players_cfg.get("thickness", 2))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), player_color, player_thickness)
                track_id = player.get('id') or player.get('track_id')
                if track_id is not None:
                    draw_boxed_text(frame, f"P{int(track_id)}", (int(x1), int(y1)))

        # Draw actions
        if actions_enabled and frame_idx in actions_by_frame:
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
                action_thickness = int(actions_cfg.get("thickness", 2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, action_thickness)
                draw_boxed_text(frame, label, (x1, y1), color=color)

        # Draw mini court
        mini_court.draw(frame, frame_idx, court_ts, players_by_frame, ball_tracks)
        
        # Draw action HUD
        if action_hud is not None:
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
