from typing import Dict, List, Tuple, Optional


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


__all__ = [
    "parse_polygon_from_pred",
    "corners_from_polygon_extremes",
    "rect_from_bbox",
    "order_corners",
    "corners_from_prediction",
]
