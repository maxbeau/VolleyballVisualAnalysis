import os
from pathlib import Path
import cv2
from .types import VideoInfo


def get_video_info(video_path: str) -> VideoInfo:
    """
    Reads a video file and returns its metadata.

    Args:
        video_path: Path to the video file.

    Returns:
        A VideoInfo object containing video properties.

    Raises:
        FileNotFoundError: If the video file does not exist.
        IOError: If the video file cannot be opened or read.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)


def ensure_dir(path: str) -> None:
    """Creates a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

