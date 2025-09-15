import os
import urllib.request
from typing import Tuple, Optional, Callable

import numpy as np


class DeepReIDOnnx:
    def __init__(self, model_path: str, auto_download: bool = True):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.input_hw: Tuple[int, int] = (256, 128)  # (H, W)
        self._ensure_model(auto_download)
        self._load_session()

    def _ensure_model(self, auto_download: bool):
        if os.path.exists(self.model_path):
            return
        if not auto_download:
            raise FileNotFoundError(f"ReID ONNX not found: {self.model_path}")
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        urls = [
            "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/tracker/cfg/strongsort/osnet_x0_25_msmt17.onnx",
            "https://raw.githubusercontent.com/mikel-brostrom/Yolov5_StrongSORT_OSNet/master/osnet_x0_25_msmt17.onnx",
        ]
        last_err = None
        for url in urls:
            try:
                print(f"Downloading ReID model: {url} -> {self.model_path}")
                urllib.request.urlretrieve(url, self.model_path)
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to download ReID model to {self.model_path}: {last_err}")

    def _load_session(self):
        import onnxruntime as ort  # type: ignore
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        # Expected [1, 3, H, W]
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            H = shape[2] if isinstance(shape[2], int) else 256
            W = shape[3] if isinstance(shape[3], int) else 128
            self.input_hw = (int(H), int(W))

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        H, W = self.input_hw
        # BGR->RGB, resize to (W,H)
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = np.expand_dims(x, 0)  # NCHW
        return x

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("ONNX session not initialized")
        x = self._preprocess(bgr)
        outs = self.session.run(None, {self.input_name: x})
        feat = outs[0]
        if isinstance(feat, list):
            feat = feat[0]
        vec = np.array(feat).reshape(-1).astype(np.float32)
        n = np.linalg.norm(vec) + 1e-6
        return vec / n


def build_reid_embedder(settings) -> Optional[callable]:
    # Prefer nested players settings; fallback to top-level prefixed fields
    players = getattr(settings, 'players', None)
    backend = None
    if players is not None and hasattr(players, 'REID_BACKEND'):
        backend = str(getattr(players, 'REID_BACKEND', 'onnx')).lower()
    else:
        backend = str(getattr(settings, "PLAYERS_REID_BACKEND", "onnx")).lower()
    if backend == "hist":
        # Explicitly request simple HSV histogram embedding handled in tracker
        return None
    if backend == "onnx":
        try:
            emb = DeepReIDOnnx(
                model_path=(getattr(players, 'REID_ONNX', None) if players is not None else None) or getattr(settings, "PLAYERS_REID_ONNX", "weights/osnet_x0_25_msmt17.onnx"),
                auto_download=bool((getattr(players, 'REID_AUTO_DOWNLOAD', True) if players is not None else getattr(settings, "PLAYERS_REID_AUTO_DOWNLOAD", True))),
            )
            return emb.embed
        except Exception as e:
            print(f"Deep ReID ONNX init failed, falling back to HSV hist. Error: {e}")
            # try torch vision as fallback
    # TorchVision ResNet fallback
    try:
        import torch
        import torchvision.transforms as T
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        device = 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
        model.to(device).eval()
        preprocess = weights.transforms()

        def _embed(bgr):
            import cv2
            from PIL import Image
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = preprocess(pil).unsqueeze(0)  # 1x3x224x224
            with torch.no_grad():
                feat = model(x.to(device)).cpu().numpy().reshape(-1).astype(np.float32)
            n = np.linalg.norm(feat) + 1e-6
            return feat / n

        print("Using TorchVision ResNet50 as deep ReID embedder.")
        return _embed
    except Exception as e:
        print(f"TorchVision ReID fallback failed, using HSV hist. Error: {e}")
        return None


__all__ = ["DeepReIDOnnx", "build_reid_embedder"]
 
