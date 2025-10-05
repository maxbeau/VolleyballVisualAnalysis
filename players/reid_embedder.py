import numpy as np
from typing import Optional, Callable


def build_reid_embedder(settings) -> Optional[Callable]:
    """
    Builds a ReID embedder based on the provided settings.
    The default deep learning backend is TorchVision ResNet50.
    An alternative is a simple HSV histogram backend.
    """
    # Prefer nested players settings; fallback to top-level prefixed fields
    players = getattr(settings, 'players', None)
    backend = None
    if players is not None and hasattr(players, 'REID_BACKEND'):
        backend = str(getattr(players, 'REID_BACKEND', 'torchvision')).lower()
    else:
        backend = str(getattr(settings, "PLAYERS_REID_BACKEND", "torchvision")).lower()

    if backend == "hist":
        # Explicitly request simple HSV histogram embedding handled in tracker
        print("Using HSV histogram as ReID embedder.")
        return None

    # Default to TorchVision ResNet50 as the deep ReID embedder
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

        def _embed(bgr: np.ndarray) -> np.ndarray:
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
        print(f"TorchVision ReID embedder failed, falling back to HSV hist. Error: {e}")
        return None


__all__ = ["build_reid_embedder"]
