from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

from court.utils import standard_court_model_size, build_court_model_template, template_precision_score


Point = Tuple[float, float]


class CourtOverlay:
    """
    Render court border and internal lines (center, attack) on a frame
    given the 4 image-space corners for that frame.

    Implementation details:
    - Chooses model->image homography among 0/90/180/270 deg rotations
      by template precision score, with EMA smoothing and hysteresis to
      avoid flicker.
    - Draws outer border from provided corners; draws internal lines by
      projecting model lines via the chosen homography.
    """

    def __init__(
        self,
        border_color: Tuple[int, int, int],
        thickness: int,
        model_size: Optional[Tuple[int, int]] = None,
        tpl_line_px: int = 2,
        center_color: Tuple[int, int, int] = (0, 255, 255),
        attack_color: Tuple[int, int, int] = (255, 0, 255),
        diag: bool = False,
    ) -> None:
        self.border_color = border_color
        self.center_color = center_color
        self.attack_color = attack_color
        self.thickness = max(1, int(thickness))
        self.diag = bool(diag)
        self.Wm, self.Hm = model_size or standard_court_model_size(100.0)
        # Use vertical-line template (matches overlay's internal scoring)
        self.model_tpl = build_court_model_template(self.Wm, self.Hm, max(1, int(tpl_line_px)), orientation="vertical")
        # Orientation temporal smoothing + hysteresis
        self._score_ema = np.zeros(4, dtype=np.float32)
        self._ema_inited = False
        self._ema_alpha = 0.3
        self._improve_eps = 0.05
        self._switch_patience = 3
        self._consecutive_wins = 0
        self._lock_frames = 10
        self._lock_left = 0
        self._last_rot = 0

    def _choose_H(self, frame_gray: np.ndarray, pts: List[Point]) -> np.ndarray:
        Wm, Hm = self.Wm, self.Hm
        dst = np.array([[pts[0][0], pts[0][1]], [pts[1][0], pts[1][1]], [pts[2][0], pts[2][1]], [pts[3][0], pts[3][1]]], dtype=np.float32)
        src0 = np.array([[0.0, 0.0], [Wm - 1.0, 0.0], [Wm - 1.0, Hm - 1.0], [0.0, Hm - 1.0]], dtype=np.float32)
        src90 = np.array([[0.0, Hm - 1.0], [0.0, 0.0], [Wm - 1.0, 0.0], [Wm - 1.0, Hm - 1.0]], dtype=np.float32)
        src180 = np.array([[Wm - 1.0, Hm - 1.0], [0.0, Hm - 1.0], [0.0, 0.0], [Wm - 1.0, 0.0]], dtype=np.float32)
        src270 = np.array([[Wm - 1.0, 0.0], [Wm - 1.0, Hm - 1.0], [0.0, Hm - 1.0], [0.0, 0.0]], dtype=np.float32)
        candidates = [src0, src90, src180, src270]

        def score_for(src: np.ndarray):
            Hmi = cv2.getPerspectiveTransform(src, dst)
            prec = template_precision_score(frame_gray, Hmi, self.model_tpl)
            return float(prec), Hmi

        raw_scores = np.zeros(4, dtype=np.float32)
        Hcands = {}
        for k, s in enumerate(candidates):
            sc, Hmi = score_for(s)
            raw_scores[k] = sc
            Hcands[k] = Hmi

        if not self._ema_inited:
            self._score_ema = raw_scores.copy()
            self._ema_inited = True
            self._last_rot = int(np.argmax(self._score_ema))
            return Hcands[self._last_rot]

        # EMA smoothing + hysteresis
        self._score_ema = (1.0 - self._ema_alpha) * self._score_ema + self._ema_alpha * raw_scores
        best_rot = int(np.argmax(self._score_ema))

        if self._lock_left > 0:
            self._lock_left -= 1
            return Hcands[self._last_rot]

        curr = float(self._score_ema[self._last_rot])
        cand = float(self._score_ema[best_rot])
        if best_rot != self._last_rot and cand > curr * (1.0 + self._improve_eps):
            self._consecutive_wins += 1
            if self._consecutive_wins >= self._switch_patience:
                self._last_rot = best_rot
                self._consecutive_wins = 0
                self._lock_left = self._lock_frames
        else:
            self._consecutive_wins = 0

        return Hcands[self._last_rot]

    def draw(self, frame: np.ndarray, pts: List[Point], info: Optional[Dict] = None) -> np.ndarray:
        # Draw border
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for a, b in pairs:
            ax, ay = int(round(pts[a][0])), int(round(pts[a][1]))
            bx, by = int(round(pts[b][0])), int(round(pts[b][1]))
            cv2.line(frame, (ax, ay), (bx, by), self.border_color, self.thickness)

        # Choose H and draw internal lines
        gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = self._choose_H(gr, pts)

        def proj_vline(x_model: float):
            P = np.array([[x_model, 0.0], [x_model, self.Hm - 1.0]], dtype=np.float32).reshape(-1, 1, 2)
            Q = cv2.perspectiveTransform(P, H).reshape(-1, 2)
            return (int(round(Q[0, 0])), int(round(Q[0, 1]))), (int(round(Q[1, 0])), int(round(Q[1, 1])))

        x_center = (self.Wm - 1.0) * 0.5
        c0, c1 = proj_vline(x_center)
        cv2.line(frame, c0, c1, self.center_color, max(1, self.thickness))
        x_a1 = (self.Wm - 1.0) * (6.0 / 18.0)
        x_a2 = (self.Wm - 1.0) * (12.0 / 18.0)
        a10, a11 = proj_vline(x_a1)
        a20, a21 = proj_vline(x_a2)
        cv2.line(frame, a10, a11, self.attack_color, max(1, self.thickness))
        cv2.line(frame, a20, a21, self.attack_color, max(1, self.thickness))

        # Optional diagnostics (kept here for future use)
        if self.diag and isinstance(info, dict):
            try:
                txt = []
                for k in ("inlier_ratio", "med_reproj_err", "condH", "scale_x", "scale_y", "tpl_prec", "roi_scale", "matches", "inliers"):
                    v = info.get(k)
                    if v is not None:
                        try:
                            txt.append(f"{k}:{float(v):.3f}")
                        except Exception:
                            txt.append(f"{k}:{v}")
                if txt:
                    s = "  ".join(txt)
                    cv2.putText(frame, s, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass

        return frame


__all__ = ["CourtOverlay"]

