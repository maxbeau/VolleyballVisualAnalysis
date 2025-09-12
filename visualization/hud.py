from typing import Tuple

import cv2


def draw_boxed_text(frame, text: str, pos: Tuple[int, int], color=(220, 220, 220), bg=(0, 0, 0), scale=0.6, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    bg_x1 = max(0, x - 2)
    bg_y1 = max(0, y - th - 6)
    bg_x2 = min(frame.shape[1] - 1, x + tw + 2)
    bg_y2 = min(frame.shape[0] - 1, y + 4)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg, thickness=-1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_hud(frame, fps: float, frame_idx: int, total_frames: int):
    try:
        hud_txt = f"FPS {fps:.2f} | frame {frame_idx}/{max(0,total_frames-1)}"
        cv2.putText(frame, hud_txt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, hud_txt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass
    return frame


__all__ = ["draw_boxed_text", "draw_hud"]

