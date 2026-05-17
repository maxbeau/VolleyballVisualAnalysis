"""HTTP client for Ultralytics Platform inference."""
from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

import cv2
import requests


class UltralyticsClient:
    """Small requests-based client for Ultralytics shared or dedicated endpoints."""

    def __init__(self, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Ultralytics API key must not be empty.")
        self._api_key = api_key.strip()
        self._session = requests.Session()

    def infer_frame(
        self,
        frame,
        *,
        endpoint_url: str,
        confidence: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Run inference on a BGR frame and return the raw JSON response."""
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode frame for Ultralytics inference.")

        data: Dict[str, Any] = {}
        if confidence is not None:
            data["conf"] = float(confidence)
        if iou is not None:
            data["iou"] = float(iou)
        if imgsz is not None:
            data["imgsz"] = int(imgsz)

        _add_no_proxy_host(endpoint_url)
        response = self._session.post(
            endpoint_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            data=data,
            files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(_format_error(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Ultralytics returned a non-JSON response.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Ultralytics returned an unexpected payload.")
        return payload


def _format_error(response: requests.Response) -> str:
    status = response.status_code
    message = response.text.strip()
    try:
        payload = response.json()
        if isinstance(payload, dict):
            message = str(payload.get("message") or payload.get("error") or message)
    except ValueError:
        pass

    hints = {
        401: "Check ULTRALYTICS_API_KEY.",
        404: "Check the model ID or dedicated endpoint URL.",
        429: "Rate limited; reduce FPS or use a dedicated endpoint.",
    }
    suffix = f" {hints[status]}" if status in hints else ""
    return f"Ultralytics inference failed with HTTP {status}: {message}.{suffix}"


def _add_no_proxy_host(endpoint_url: str) -> None:
    import os

    host = urlparse(endpoint_url).hostname
    if not host:
        return
    for key in ("NO_PROXY", "no_proxy"):
        curr = os.environ.get(key, "")
        parts = [p.strip() for p in curr.split(",") if p.strip()]
        if host not in parts:
            parts.append(host)
        os.environ[key] = ",".join(parts)


__all__ = ["UltralyticsClient"]
