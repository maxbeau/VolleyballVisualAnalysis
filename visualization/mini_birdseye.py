from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
from court.utils import compute_homography, apply_homography_points


def render_template(W: int, H: int, colors: Dict[str, Tuple[int, int, int]], thickness: int, orientation: str) -> np.ndarray:
    """Render an abstract bird's-eye court template.
    - orientation: "horizontal" (width=18m) -> draw vertical lines at x=6/18,9/18,12/18
                  "vertical"   (height=18m) -> draw horizontal lines at y=6/18,9/18,12/18
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), colors.get("border", (0, 255, 0)), max(1, thickness))
    if str(orientation).lower().startswith("h"):
        x_center = int(round((W - 1) * (9.0 / 18.0)))
        x_attack_l = int(round((W - 1) * (6.0 / 18.0)))
        x_attack_r = int(round((W - 1) * (12.0 / 18.0)))
        cv2.line(img, (x_center, 0), (x_center, H - 1), colors.get("center", (0, 255, 255)), max(1, thickness))
        cv2.line(img, (x_attack_l, 0), (x_attack_l, H - 1), colors.get("attack", (255, 0, 255)), max(1, thickness))
        cv2.line(img, (x_attack_r, 0), (x_attack_r, H - 1), colors.get("attack", (255, 0, 255)), max(1, thickness))
    else:
        y_center = int(round((H - 1) * (9.0 / 18.0)))
        y_attack_t = int(round((H - 1) * (6.0 / 18.0)))
        y_attack_b = int(round((H - 1) * (12.0 / 18.0)))
        cv2.line(img, (0, y_center), (W - 1, y_center), colors.get("center", (0, 255, 255)), max(1, thickness))
        cv2.line(img, (0, y_attack_t), (W - 1, y_attack_t), colors.get("attack", (255, 0, 255)), max(1, thickness))
        cv2.line(img, (0, y_attack_b), (W - 1, y_attack_b), colors.get("attack", (255, 0, 255)), max(1, thickness))
    return img


__all__ = ["render_template"]


class MiniBirdseyeOverlay:
    def __init__(
        self,
        colors: Dict[str, Tuple[int, int, int]],
        thickness: int,
        placement: str = "top-right",
        scale: float = 0.24,
        margin: int = 12,
        show_label: bool = True,
        draw_poly: bool = True,
    ) -> None:
        self.colors = colors
        self.thickness = max(1, int(thickness))
        self.placement = str(placement or "top-right").lower()
        self.scale = float(scale)
        self.margin = int(margin)
        self.show_label = bool(show_label)
        self.draw_poly = bool(draw_poly)
        self._cache = {}

    def _compute_rect(self, frame_w: int, frame_h: int, orientation: str) -> Tuple[int, int, int, int]:
        m = self.margin
        orient = str(orientation or "horizontal").lower()
        if orient == "vertical":
            ov_h = max(60, int(round(frame_h * max(0.01, self.scale, 0.30))))
            ov_w = max(60, int(round(ov_h * (9.0 / 18.0))))
        else:
            ov_w = max(120, int(round(frame_w * max(0.01, self.scale))))
            ov_h = max(60, int(round(ov_w * (9.0 / 18.0))))
        # Clamp within frame with margins
        ov_w = min(ov_w, frame_w - 2 * m)
        ov_h = min(ov_h, frame_h - 2 * m)
        # Placement
        if self.placement == "top-left":
            x1, y1 = m, m
        elif self.placement == "bottom-left":
            x1, y1 = m, frame_h - m - ov_h
        elif self.placement == "bottom-right":
            x1, y1 = frame_w - m - ov_w, frame_h - m - ov_h
        else:  # top-right
            x1, y1 = frame_w - m - ov_w, m
        x2, y2 = x1 + ov_w, y1 + ov_h
        return x1, y1, x2, y2

    def _get_template(self, ov_w: int, ov_h: int, orientation: str) -> np.ndarray:
        key = (ov_w, ov_h, orientation)
        if key in self._cache:
            return self._cache[key]
        tpl = render_template(ov_w, ov_h, self.colors, max(1, int(round(self.thickness * 0.8))), orientation)
        self._cache[key] = tpl
        return tpl

    def render(
        self,
        frame: np.ndarray,
        orientation: str,
        corners: Optional[List[Tuple[float, float]]] = None,
        model_size: Tuple[int, int] = (1800, 900),
        players_xy: Optional[List[Tuple[float, float]]] = None,
        players_color: Tuple[int, int, int] = (0, 165, 255),
    ) -> None:
        Hf, Wf = frame.shape[:2]
        x1, y1, x2, y2 = self._compute_rect(Wf, Hf, orientation)
        ov_w, ov_h = x2 - x1, y2 - y1
        tpl = self._get_template(ov_w, ov_h, orientation)
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.25, tpl, 0.75, 0.0)
        # Draw polygon if available
        if self.draw_poly and corners and len(corners) >= 4:
            try:
                H_img2model, _ = compute_homography(corners, dst_size=model_size)
                pts_model = apply_homography_points(corners, H_img2model)
                poly = []
                Wm, Hm = model_size
                for (mx, my) in pts_model:
                    sx = int(round(mx * (ov_w - 1) / max(1, (Wm - 1))))
                    sy = int(round(my * (ov_h - 1) / max(1, (Hm - 1))))
                    poly.append((x1 + sx, y1 + sy))
                line_color = self.colors.get("border", (0, 255, 0))
                for k in range(4):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % 4]
                    cv2.line(frame, p1, p2, line_color, max(1, int(round(self.thickness * 0.9))))
            except Exception:
                pass
        # Draw players (projected to bird's-eye) if provided and corners available
        if players_xy and corners and len(corners) >= 4:
            try:
                H_img2model, _ = compute_homography(corners, dst_size=model_size)
                pts_model = apply_homography_points(players_xy, H_img2model)
                Wm, Hm = model_size
                for (mx, my) in pts_model:
                    sx = int(round(mx * (ov_w - 1) / max(1, (Wm - 1))))
                    sy = int(round(my * (ov_h - 1) / max(1, (Hm - 1))))
                    cx = x1 + sx
                    cy = y1 + sy
                    cv2.circle(frame, (cx, cy), max(2, int(round(self.thickness * 1.2))), players_color, -1, lineType=cv2.LINE_AA)
            except Exception:
                pass
        # Label
        if self.show_label:
            try:
                label = "horizontal" if str(orientation).startswith("h") else "vertical"
                tx, ty = x1, y2 + 18
                if ty + 6 > Hf:
                    ty = max(12, y1 - 6)
                from visualization.hud import draw_boxed_text
                draw_boxed_text(frame, label, (tx, ty))
            except Exception:
                pass
