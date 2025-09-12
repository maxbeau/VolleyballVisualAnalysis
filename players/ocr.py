import re
from typing import Optional, Tuple

import numpy as np


class JerseyOCR:
    def __init__(self, languages=("en",)):
        try:
            import easyocr  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "easyocr is required for JerseyOCR. Install with: python3 -m pip install easyocr"
            ) from e
        # lazy heavy init
        self._reader = easyocr.Reader(languages, gpu=False)

    @staticmethod
    def _pick_digits(results) -> Tuple[str, float]:
        """Pick best numeric string from EasyOCR results.
        Returns (text, conf). Empty text if none passes.
        """
        best_text = ""
        best_conf = 0.0
        for text, conf in results:
            if not isinstance(text, str):
                continue
            t = re.sub(r"\D+", "", text)  # keep digits only
            if not t:
                continue
            c = float(conf or 0.0)
            # prefer longer digit strings and higher conf
            score = c * (1.0 + 0.2 * (len(t) - 1))
            if score > best_conf:
                best_conf = score
                best_text = t
        return best_text, best_conf

    def infer_digits(self, bgr: np.ndarray) -> Tuple[str, float]:
        # easyocr expects RGB
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # the reader accepts ndarray
        outs = self._reader.readtext(rgb, detail=0, paragraph=False)
        # When detail=0, outs is list[str], but some versions return (bbox, text, conf)
        if len(outs) > 0 and isinstance(outs[0], str):
            pairs = [(t, 0.5) for t in outs]  # assign mid conf if not provided
        else:
            # detail=0 sometimes returns [] but detail=1 returns bbox,text,conf
            outs2 = self._reader.readtext(rgb, detail=1, paragraph=False)
            pairs = [(o[1], float(o[2])) for o in outs2]
        return self._pick_digits(pairs)


__all__ = ["JerseyOCR"]

