"""Utility for jersey number OCR using EasyOCR (optional)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


_LOG = logging.getLogger(__name__)


@dataclass
class JerseyOCROptions:
    allowlist: str = "0123456789"
    upper_frac: float = 0.6
    resize_h: int = 320
    gpu: bool = False
    min_conf: float = 0.5
    smooth_beta: float = 0.6


class JerseyOCR:
    """Thin wrapper around EasyOCR for reading jersey numbers.

    The reader is initialised lazily to avoid importing heavy backends when OCR
    is disabled. When EasyOCR is unavailable the class degrades gracefully and
    simply returns ``None`` for every inference request.
    """

    def __init__(self, opts: JerseyOCROptions):
        self.opts = opts
        self._reader = None
        self._available = False
        self._init_err: Optional[str] = None

    def _ensure_reader(self):
        if self._reader is not None or self._available:
            return
        try:
            import easyocr  # type: ignore

            languages = ["en"]  # digits are language agnostic; keep set minimal
            self._reader = easyocr.Reader(
                languages,
                gpu=bool(self.opts.gpu),
                verbose=False,
            )
            self._available = True
        except ImportError as exc:  # pragma: no cover - optional dependency
            self._init_err = (
                "EasyOCR is not installed. Install easyocr (and its dependencies) "
                "to enable jersey number OCR."
            )
            _LOG.warning(self._init_err)
            _LOG.debug("EasyOCR import failed", exc_info=exc)
        except Exception as exc:  # pragma: no cover - runtime initialisation failure
            self._init_err = f"Failed to initialise EasyOCR reader: {exc}"
            _LOG.warning(self._init_err)
            _LOG.debug("EasyOCR initialisation failed", exc_info=exc)

    def available(self) -> bool:
        self._ensure_reader()
        return self._available and self._reader is not None

    def infer(self, frame_bgr: np.ndarray, tlbr: Tuple[int, int, int, int]) -> Optional[Tuple[str, float]]:
        if frame_bgr is None:
            return None
        if not self.available():
            return None
        x1, y1, x2, y2 = tlbr
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, int(round(x1))))
        x2 = max(0, min(w, int(round(x2))))
        y1 = max(0, min(h - 1, int(round(y1))))
        y2 = max(0, min(h, int(round(y2))))
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        frac = float(self.opts.upper_frac)
        frac = max(0.05, min(1.0, frac))
        y2_crop = y1 + int(round((y2 - y1) * frac))
        y2_crop = max(y1 + 4, min(y2 - 1, y2_crop))
        crop = frame_bgr[y1:y2_crop, x1:x2]
        if crop.size == 0:
            return None

        target_h = max(32, int(self.opts.resize_h))
        scale = target_h / float(crop.shape[0])
        target_w = max(16, int(round(crop.shape[1] * scale)))
        if target_w < 16:
            target_w = 16
        crop_resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        proc = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        allow = str(self.opts.allowlist or "0123456789")
        try:
            results = self._reader.readtext(proc, allowlist=allow)
        except Exception as exc:  # pragma: no cover - runtime OCR failure
            _LOG.debug("EasyOCR inference failed", exc_info=exc)
            return None

        best_txt: Optional[str] = None
        best_conf = 0.0
        for (_bbox, text, conf) in results:
            if not text:
                continue
            text = text.strip()
            if not text:
                continue
            text = "".join(ch for ch in text if ch in allow)
            if not text:
                continue
            conf = float(conf)
            if conf < float(self.opts.min_conf):
                continue
            if conf > best_conf:
                best_txt = text
                best_conf = conf
        if best_txt is None:
            return None
        return best_txt, best_conf


__all__ = ["JerseyOCROptions", "JerseyOCR"]
