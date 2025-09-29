"""Roboflow cloud inference backend."""
from __future__ import annotations

from typing import Any, Dict

from core.roboflow_client import RoboflowClient

from .base import DetectionBackend, ensure_confidence


class RoboflowBackend(DetectionBackend):
    name = "roboflow"

    def __init__(self, common_settings, target_settings) -> None:
        super().__init__(common_settings, target_settings)
        api_key_secret = common_settings.roboflow_api_key
        if not api_key_secret:
            raise RuntimeError(
                "ROBOFLOW_API_KEY environment variable must be set when using the 'roboflow' backend."
            )
        
        self._client = RoboflowClient(api_key=api_key_secret.get_secret_value())

    def infer(self, frame, *, frame_idx: int, model_id: str, confidence: float) -> Dict[str, Any]:
        try:
            result = self._client.infer_frame(
                frame, model_id=model_id, confidence=ensure_confidence(confidence)
            )
        except Exception as exc:
            # Improve diagnostics for common Roboflow HTTP errors
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            api_message = getattr(exc, "api_message", None)
            # 401/403 typically indicate auth/workspace/model access issues
            if status in (401, 403):
                hints = (
                    [
                        "Roboflow returned an authorization error (status %s)." % status,
                        f"Model ID: {model_id}",
                        "Likely causes:",
                        "- ROBOFLOW_API_KEY invalid or not loaded (.env).",
                        "- API key lacks access to this project/workspace.",
                        "- Model/version not deployed to Hosted Inference.",
                        "- Model ID is wrong (copy from Roboflow Deploy page).",
                    ]
                )
                msg = "\n".join(hints)
                if api_message:
                    msg += f"\nRoboflow message: {api_message}"
                msg += (
                    "\nFix suggestions:\n"
                    "- Verify ROBOFLOW_API_KEY in your .env matches the workspace of this model.\n"
                    "- Open Roboflow -> Project -> Deploy and copy the exact Model ID (e.g., 'my-project/1').\n"
                    "- Ensure the version is trained/published for Hosted API.\n"
                    "- As a fallback, set detection.backend to 'local-yolo' and provide local weights."
                )
                raise RuntimeError(msg) from exc

            # For other errors, re-raise with minimal context
            raise

        if not isinstance(result, dict):  # pragma: no cover - defensive
            raise RuntimeError("Roboflow client returned an unexpected payload")
        return result


__all__ = ["RoboflowBackend"]
