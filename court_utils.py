from typing import Dict, List, Tuple, Optional


def _np():
    import numpy as _np
    return _np


def _cv2():
    import cv2 as _cv2
    return _cv2


def parse_polygon_from_pred(pred: Dict) -> Optional[List[Tuple[float, float]]]:
    pts = pred.get("points")
    if pts is None:
        return None
    if isinstance(pts, list) and len(pts) > 0:
        first = pts[0]
        if isinstance(first, dict) and "x" in first and "y" in first:
            return [(float(p["x"]), float(p["y"])) for p in pts]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return [(float(p[0]), float(p[1])) for p in pts]
    if isinstance(pts, dict) and "x" in pts and "y" in pts:
        xs, ys = pts["x"], pts["y"]
        if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys):
            return [(float(x), float(y)) for x, y in zip(xs, ys)]
    return None


def corners_from_polygon_extremes(poly: List[Tuple[float, float]]):
    import numpy as np

    pts = np.array(poly, dtype=np.float32)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1])), (float(br[0]), float(br[1])), (float(bl[0]), float(bl[1]))]


def rect_from_bbox(pred: Dict):
    x = float(pred.get("x", 0.0))
    y = float(pred.get("y", 0.0))
    w = float(pred.get("width", 0.0))
    h = float(pred.get("height", 0.0))
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def order_corners(pts: List[Tuple[float, float]]):
    import numpy as np
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1])), (float(br[0]), float(br[1])), (float(bl[0]), float(bl[1]))]


def corners_from_prediction(pred: Dict) -> Optional[List[Tuple[float, float]]]:
    poly = parse_polygon_from_pred(pred)
    if poly and len(poly) >= 4:
        corners = corners_from_polygon_extremes(poly)
        return order_corners(corners)
    corners = rect_from_bbox(pred)
    return order_corners(corners)


def standard_court_model_size(scale_px_per_meter: float = 100.0) -> Tuple[int, int]:
    """
    Returns a canonical volleyball court bird's-eye canvas size (W,H) in pixels.
    Default uses 18m x 9m mapped by the given scale.
    """
    width_m, height_m = 18.0, 9.0
    W = int(round(width_m * scale_px_per_meter))
    H = int(round(height_m * scale_px_per_meter))
    return max(4, W), max(4, H)


def standard_court_dst_corners(W: int, H: int) -> List[Tuple[float, float]]:
    """
    Canonical destination corner ordering: TL, TR, BR, BL for a W x H canvas.
    """
    return [(0.0, 0.0), (float(W - 1), 0.0), (float(W - 1), float(H - 1)), (0.0, float(H - 1))]


def compute_homography(
    src_corners: List[Tuple[float, float]],
    dst_corners: Optional[List[Tuple[float, float]]] = None,
    dst_size: Optional[Tuple[int, int]] = None,
):
    """
    Compute homography H (3x3) from src_corners (image) to dst_corners (bird's-eye).
    - src_corners: 4 points in TL,TR,BR,BL order (float tuples)
    - dst_corners: 4 points in TL,TR,BR,BL order; if None, inferred from dst_size
    - dst_size: (W,H) for a standard 18x9 court if dst_corners is None
    Returns: (H: np.ndarray shape (3,3), dst_size: (W,H))
    """
    np = _np()
    cv2 = _cv2()

    if dst_corners is None:
        if dst_size is None:
            dst_size = standard_court_model_size()
        W, H = dst_size
        dst_corners = standard_court_dst_corners(W, H)
    else:
        # derive W,H from provided corners' bbox if not given
        xs = [p[0] for p in dst_corners]
        ys = [p[1] for p in dst_corners]
        W = int(round(max(xs) - min(xs) + 1))
        H = int(round(max(ys) - min(ys) + 1))

    src = np.array(src_corners, dtype=np.float32)
    dst = np.array(dst_corners, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("compute_homography expects 4 src and 4 dst points (TL,TR,BR,BL)")

    Hmat = cv2.getPerspectiveTransform(src, dst)
    return Hmat, (W, H)


def warp_birdseye(image, H, dst_size: Tuple[int, int]):
    """
    Apply perspective warp using H to produce a bird's-eye view of the court.
    - image: BGR numpy array
    - H: 3x3 homography matrix
    - dst_size: (W,H) output size
    Returns: warped image (W x H)
    """
    cv2 = _cv2()
    np = _np()
    W, Hh = dst_size
    img_c = np.ascontiguousarray(image)
    Hm = np.asarray(H, dtype=np.float64)
    return cv2.warpPerspective(img_c, Hm, (W, Hh))


def median_corners_from_tracking(tracking_jsonl: str) -> Optional[List[Tuple[float, float]]]:
    """
    Load a court tracking JSONL (with per-frame `corners`) and compute the
    median corner across frames (robust to outliers). Returns 4 ordered points.
    """
    import json
    import os

    if not os.path.exists(tracking_jsonl):
        return None

    all_pts: List[List[Tuple[float, float]]] = []
    with open(tracking_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            corners = rec.get("corners")
            if not corners or len(corners) < 4:
                continue
            # ensure consistent ordering per frame
            ord_pts = order_corners([(float(p[0]), float(p[1])) for p in corners[:4]])
            all_pts.append(ord_pts)

    if not all_pts:
        return None

    np = _np()
    arr = np.array(all_pts, dtype=np.float64)  # shape: (N,4,2)
    med = np.median(arr, axis=0)  # (4,2)
    return [(float(med[i, 0]), float(med[i, 1])) for i in range(4)]


__all__ = [
    "parse_polygon_from_pred",
    "corners_from_polygon_extremes",
    "rect_from_bbox",
    "order_corners",
    "corners_from_prediction",
    "standard_court_model_size",
    "standard_court_dst_corners",
    "compute_homography",
    "warp_birdseye",
    "median_corners_from_tracking",
]


def apply_homography_points(
    pts: List[Tuple[float, float]],
    H,
) -> List[Tuple[float, float]]:
    """
    Applies homography H to a list of (x,y) image points.
    Returns transformed (x',y') in the destination plane.
    """
    np = _np()
    cv2 = _cv2()
    if not pts:
        return []
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    Hm = np.asarray(H, dtype=np.float64)
    out = cv2.perspectiveTransform(arr, Hm).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]
