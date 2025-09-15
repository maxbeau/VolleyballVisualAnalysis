from typing import Dict, List, Any, Optional, Tuple
from court.utils import compute_homography, apply_homography_points


class CrossValidator:
    """
    Cross-validate active clips with realtime detections and players.
    - Validates clips: lowers confidence if detections/players do not support
    - Supplements clips: elevates high-confidence detections with inferred team
    Returns per-frame events: [{class, team_name, source, confidence}]
    """

    def __init__(self,
                 actions_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
                 players_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
                 court_timeseries: Optional[Dict[int, List[Tuple[float, float]]]] = None,
                 Wm: int = 1800,
                 Hm: int = 900,
                 side_to_team: Optional[Dict[str, str]] = None,
                 det_min_conf: float = 0.25,
                 det_high_conf: float = 0.6,
                 validate_window: int = 4,
                 player_conf_min: float = 0.1,
                 player_match_px: float = 120.0,
                 ):
        self.actions_by_frame = actions_by_frame or {}
        self.players_by_frame = players_by_frame or {}
        self.court_timeseries = court_timeseries or {}
        self.Wm, self.Hm = int(Wm), int(Hm)
        self.side_to_team = side_to_team or {}
        self.det_min_conf = float(det_min_conf)
        self.det_high_conf = float(det_high_conf)
        self.validate_window = int(validate_window)
        self.player_conf_min = float(player_conf_min)
        self.player_match_px = float(player_match_px)

    def _H_for(self, fi: int):
        pts = self.court_timeseries.get(fi)
        if pts is None:
            ks = [k for k in self.court_timeseries.keys() if k <= fi]
            if ks:
                pts = self.court_timeseries.get(max(ks))
        if pts is None or len(pts) != 4:
            return None
        try:
            H, _ = compute_homography(pts, dst_size=(self.Wm, self.Hm))
            return H
        except Exception:
            return None

    def _point_side(self, fi: int, x: float, y: float) -> Optional[str]:
        H = self._H_for(fi)
        if H is None:
            return None
        try:
            tx, ty = apply_homography_points([(x, y)], H)[0]
            return "left" if tx < (self.Wm * 0.5) else "right"
        except Exception:
            return None

    def _team_from_xy(self, fi: int, x: float, y: float) -> Optional[str]:
        s = self._point_side(fi, x, y)
        if s is None:
            return None
        return self.side_to_team.get(s)

    def _nearest_player_xy(self, fi: int, ax: float, ay: float) -> Optional[Tuple[float, float]]:
        trs = self.players_by_frame.get(fi) or []
        best_d2 = None
        best_pt = None
        for t in trs:
            try:
                conf_p = float(t.get("confidence", 0.0))
            except Exception:
                conf_p = 0.0
            if conf_p < self.player_conf_min:
                continue
            cx = float(t.get("x", 0.0))
            cy = float(t.get("y", 0.0))
            h = float(t.get("height", 0.0))
            bx, by = cx, cy + 0.5 * h
            dx, dy = bx - ax, by - ay
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_pt = (bx, by)
        if best_pt is None:
            return None
        if best_d2 is not None and best_d2 <= (self.player_match_px * self.player_match_px):
            return best_pt
        return None

    def events_for(self,
                   fi: int,
                   active_clips: Dict[str, Optional[Dict[str, Any]]],
                   min_conf: float = 0.25) -> List[Dict[str, Any]]:
        evs: List[Dict[str, Any]] = []
        # Validate existing clips
        for cname, clip in active_clips.items():
            if clip is None:
                continue
            base_conf = float(clip.get("side_conf", 1.0))
            team = clip.get("team_name")
            supported_det = False
            supported_player = False
            # detection support within window
            for k in range(max(0, fi - self.validate_window), fi + self.validate_window + 1):
                dets = self.actions_by_frame.get(k) or []
                for p in dets:
                    if str(p.get("class", "")).lower() != cname:
                        continue
                    try:
                        c = float(p.get("confidence", 0.0))
                    except Exception:
                        c = 0.0
                    if c >= self.det_min_conf:
                        supported_det = True
                        break
                if supported_det:
                    break
            # player proximity support (use clip med bottom if available, else skip)
            ax = clip.get("med_btm_x")
            ay = clip.get("med_btm_y")
            if ax is not None and ay is not None:
                supported_player = self._nearest_player_xy(fi, float(ax), float(ay)) is not None
            # confidence adjustment
            conf = base_conf
            if not supported_det and not supported_player:
                conf = max(0.0, base_conf * 0.5)
                source = "clip-xval-low"
            elif supported_det and supported_player:
                conf = min(1.0, base_conf * 1.05)
                source = "clip-xval-high"
            else:
                source = "clip"
            evs.append({
                "class": cname,
                "team_name": team,
                "source": source,
                "confidence": conf,
            })

        # Supplement with high-confidence detections (team inferred via court+players)
        dets_now = self.actions_by_frame.get(fi) or []
        for cname in ("serve", "set", "spike", "block"):
            if active_clips.get(cname) is not None:
                continue
            best = None
            best_c = -1.0
            for p in dets_now:
                if str(p.get("class", "")).lower() != cname:
                    continue
                try:
                    c = float(p.get("confidence", 0.0))
                except Exception:
                    c = 0.0
                if c > best_c:
                    best_c = c
                    best = p
            if best is not None and best_c >= self.det_high_conf:
                # infer team: prefer detection bottom center side via court; fallback nearest player
                x = float(best.get("x", 0.0))
                y = float(best.get("y", 0.0)) + 0.5 * float(best.get("height", 0.0))
                team = self._team_from_xy(fi, x, y)
                if team is None:
                    nb = self._nearest_player_xy(fi, x, y)
                    if nb is not None:
                        team = self._team_from_xy(fi, nb[0], nb[1])
                evs.append({
                    "class": cname,
                    "team_name": team,
                    "source": "det-xval",
                    "confidence": best_c,
                })
        return evs


__all__ = ["CrossValidator"]
