from typing import Any, Dict, Tuple

import cv2
import numpy as np

from court.utils import (
    compute_homography,
    build_court_model_template,
    template_precision_score,
)


def decide_orientation(
    cap: Any,
    ts: Dict[int, Dict],
    model_size: Tuple[int, int],
    max_samples: int = 6,
    search_window: int = 300,
) -> str:
    """
    Decide court orientation ("horizontal" or "vertical") by voting
    over template precision on a few early samples.
    Args:
      cap: an opened cv2.VideoCapture (position will be restored)
      ts: dict mapping frame index -> {"corners": [(x,y)*4], ...}
      model_size: (W,H) of the standard court canvas
    """
    keys = sorted(ts.keys())
    if not keys:
        return "horizontal"

    # Template voting
    try:
        Wm, Hm = model_size
        tpl_h = build_court_model_template(Wm, Hm, line_px=6, orientation="horizontal")
        tpl_v = build_court_model_template(Wm, Hm, line_px=6, orientation="vertical")
        # Restrict to early window
        early = [k for k in keys if k <= keys[0] + max(1, int(search_window))]
        sel = early if early else keys
        if len(sel) > max_samples:
            stride = max(1, len(sel) // max_samples)
            samp = sel[::stride][:max_samples]
        else:
            samp = sel
        sum_h = 0.0; sum_v = 0.0; cnt = 0
        pos_backup = cap.get(cv2.CAP_PROP_POS_FRAMES)
        for fi in samp:
            rec = ts.get(fi)
            if not rec:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                H_img2model, _ = compute_homography(rec["corners"], dst_size=(Wm, Hm))
                H_model2img = np.linalg.inv(H_img2model)
                sh = float(template_precision_score(gray, H_model2img, tpl_h))
                sv = float(template_precision_score(gray, H_model2img, tpl_v))
            except Exception:
                continue
            sum_h += sh; sum_v += sv; cnt += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_backup)
        if cnt == 0:
            return "horizontal"
        margin = 0.02
        # The template score logic is counter-intuitive:
        # - A "vertical" template has vertical lines (net, attack lines), which matches a HORIZONTAL court (side view).
        # - A "horizontal" template has horizontal lines, which matches a VERTICAL court (back view).
        # We swap the return values to reflect the actual court orientation.
        # The margin adds robustness against ambiguous cases where scores are very close.
        return "horizontal" if (sum_v / cnt) > (sum_h / cnt) + margin else "vertical"
    except Exception:
        return "horizontal"


__all__ = ["decide_orientation"]

