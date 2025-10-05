from typing import Dict, Any, List, Optional


class DriftSentinel:
    """
    Monitors tracking quality metrics to detect drift and trigger re-detection.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._reset_sentinel_state(frame_index=-1)

    # ---------- sentinel helpers ----------
    def _reset_sentinel_state(self, frame_index: int) -> None:
        try:
            warmup = max(int(self.cfg.sentinel.warmup_frames), 0)
            min_gap = max(int(self.cfg.sentinel.min_gap_frames), 0)
        except AttributeError:
            warmup = 0
            min_gap = 0
        self._sentinel_state = {
            "hold_streak": 0,
            "bad_inlier": 0,
            "bad_matches": 0,
            "bad_err": 0,
            "bad_template": 0,
            "bad_geo": 0,
            "last_request_frame": int(frame_index) - min_gap,
            "last_reset_frame": int(frame_index),
            "last_good_frame": int(frame_index),
            "cooldown_until": int(frame_index) + warmup,
            "template_ref": None,
        }
        self._last_sentinel_reasons: Optional[List[str]] = None

    def _sentinel_enabled(self) -> bool:
        try:
            return bool(self.cfg.sentinel.enable)
        except AttributeError:
            return False

    def _sentinel_can_trigger(self, frame_idx: int) -> bool:
        if not self._sentinel_enabled():
            return False
        cfg = self.cfg.sentinel
        state = self._sentinel_state
        if frame_idx - state.get("last_reset_frame", -10**9) < max(int(cfg.warmup_frames), 0):
            return False
        if frame_idx < state.get("cooldown_until", -10**9):
            return False
        if frame_idx - state.get("last_request_frame", -10**9) < max(int(cfg.min_gap_frames), 0):
            return False
        return True

    def on_success(
        self,
        info: Dict[str, Any],
        *,
        frame_idx: int,
        ratio: float,
        area: float,
        med_err: float,
        matches: int,
        inlier_ratio: float,
        template_score: Optional[float],
    ) -> None:
        if not self._sentinel_enabled():
            return
        cfg = self.cfg.sentinel
        state = self._sentinel_state

        state["last_good_frame"] = frame_idx
        state["hold_streak"] = 0

        reasons: List[str] = []

        if inlier_ratio < cfg.inlier_ratio_floor:
            state["bad_inlier"] = state.get("bad_inlier", 0) + 1
        else:
            state["bad_inlier"] = 0
        if state["bad_inlier"] >= max(cfg.inlier_ratio_patience, 1):
            reasons.append("inlier_ratio")

        if matches < cfg.matches_floor:
            state["bad_matches"] = state.get("bad_matches", 0) + 1
        else:
            state["bad_matches"] = 0
        if state["bad_matches"] >= max(cfg.matches_patience, 1):
            reasons.append("tracks")

        if med_err > cfg.reproj_median_ceiling:
            state["bad_err"] = state.get("bad_err", 0) + 1
        else:
            state["bad_err"] = 0
        if state["bad_err"] >= max(cfg.reproj_patience, 1):
            reasons.append("reproj_error")

        if template_score is not None:
            ref = state.get("template_ref")
            if ref is None or template_score > ref:
                state["template_ref"] = float(template_score)
                state["bad_template"] = 0
            elif (ref - template_score) > cfg.template_drop:
                state["bad_template"] = state.get("bad_template", 0) + 1
            else:
                state["bad_template"] = 0
            if state["bad_template"] >= max(cfg.template_patience, 1):
                reasons.append("template_drop")

        geo_flag = False
        # These reference values need to be passed in or managed externally
        ref_ratio = info.get("ref_ratio")
        ref_area = info.get("ref_area")
        if ref_ratio is not None:
            if abs(ratio - ref_ratio) > cfg.geo_ratio_tol * max(1.0, ref_ratio):
                geo_flag = True
        if ref_area is not None and ref_area > 1e-6:
            rel_area = abs(area - ref_area) / max(ref_area, 1e-6)
            if rel_area > cfg.geo_area_tol:
                geo_flag = True
        if geo_flag:
            state["bad_geo"] = state.get("bad_geo", 0) + 1
        else:
            state["bad_geo"] = 0
        if state["bad_geo"] >= max(cfg.geo_patience, 1):
            reasons.append("geometry")

        score = 0.0
        score += state.get("bad_inlier", 0) / max(cfg.inlier_ratio_patience, 1)
        score += state.get("bad_matches", 0) / max(cfg.matches_patience, 1)
        score += state.get("bad_err", 0) / max(cfg.reproj_patience, 1)
        score += state.get("bad_template", 0) / max(cfg.template_patience, 1)
        score += state.get("bad_geo", 0) / max(cfg.geo_patience, 1)
        if score >= max(cfg.drift_score_threshold, 0.0):
            reasons.append("drift_score")

        if reasons and self._sentinel_can_trigger(frame_idx):
            info["needs_redetect"] = True
            info["sentinel_reasons"] = sorted(set(reasons))
            self._last_sentinel_reasons = info["sentinel_reasons"]
            state["last_request_frame"] = frame_idx
            state["cooldown_until"] = frame_idx + max(cfg.min_gap_frames, cfg.hold_bad_frames)
            state["bad_inlier"] = state["bad_matches"] = state["bad_err"] = state["bad_template"] = state["bad_geo"] = 0

    def on_hold(self, info: Dict[str, Any], *, frame_idx: int, reason: str) -> None:
        if not self._sentinel_enabled():
            return
        cfg = self.cfg.sentinel
        state = self._sentinel_state
        state["hold_streak"] = state.get("hold_streak", 0) + 1
        if state["hold_streak"] >= max(cfg.hold_bad_frames, 1) and self._sentinel_can_trigger(frame_idx):
            reasons = info.setdefault("sentinel_reasons", [])
            reasons.append(reason)
            info["needs_redetect"] = True
            self._last_sentinel_reasons = reasons
            state["last_request_frame"] = frame_idx
            state["cooldown_until"] = frame_idx + max(cfg.min_gap_frames, cfg.hold_bad_frames)
            state["hold_streak"] = 0
            state["bad_inlier"] = state["bad_matches"] = state["bad_err"] = state["bad_template"] = state["bad_geo"] = 0
        elif state["hold_streak"] > cfg.hold_bad_frames:
            state["hold_streak"] = cfg.hold_bad_frames

    def set_template_ref(self, score: float) -> None:
        if self._sentinel_enabled():
            self._sentinel_state["template_ref"] = float(score)

    def reset(self, frame_index: int) -> None:
        self._reset_sentinel_state(frame_index)

    @property
    def last_sentinel_reasons(self) -> Optional[List[str]]:
        return self._last_sentinel_reasons
