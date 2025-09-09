from typing import Any, Dict, Optional


class RoboflowClient:
    """Thin wrapper over inference-sdk for Roboflow.

    Keeps Roboflow usage isolated so other workflows can import this module,
    and future providers (e.g., 魔搭/本地模型) can replace this implementation.
    """

    def __init__(self, api_key: str, base_url: str = "https://detect.roboflow.com"):
        import os
        from inference_sdk import InferenceHTTPClient  # lazy import

        # Mitigate proxy issues by adding Roboflow hosts to NO_PROXY
        for key in ("NO_PROXY", "no_proxy"):
            curr = os.environ.get(key, "")
            parts = [p.strip() for p in curr.split(",") if p.strip()]
            for host in ("detect.roboflow.com", "roboflow.com"):
                if host not in parts:
                    parts.append(host)
            os.environ[key] = ",".join(parts)

        self._client = InferenceHTTPClient(api_url=base_url, api_key=api_key)

    def infer_image(self, image_path: str, model_id: str, confidence: Optional[float] = None) -> Dict[str, Any]:
        if confidence is not None:
            try:
                self._client.inference_configuration.confidence_threshold = float(confidence)
            except Exception:
                pass
        return self._client.infer(image_path, model_id=model_id)

    def infer_frame(self, frame, model_id: str, confidence: Optional[float] = None) -> Dict[str, Any]:
        """Run inference directly on a NumPy frame to avoid disk I/O."""
        if confidence is not None:
            try:
                self._client.inference_configuration.confidence_threshold = float(confidence)
            except Exception:
                pass
        return self._client.infer(frame, model_id=model_id)
