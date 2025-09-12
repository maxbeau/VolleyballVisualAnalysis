import os
import sys
import cv2

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import settings
from core.utils import ensure_dir


def main():
    video_path = settings.VIDEO_PATH
    out_dir = os.path.join("outputs", "preview_frames")
    out_clip = os.path.join("outputs", "preview_clip.mp4")
    ensure_dir(out_dir)
    ensure_dir(os.path.dirname(out_clip) or ".")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Extract 8 preview frames from the first ~4 seconds
    num_frames = 8
    duration_s = min(4.0, (total_frames / fps) if fps else 4.0)
    indices = [int(i * (duration_s * fps) / max(1, num_frames - 1)) for i in range(num_frames)]
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1

    # Write a short 3-second preview clip from the start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    clip_frames = int(min(3.0, total_frames / fps if fps else 3.0) * fps)
    # Ensure even dimensions for some codecs
    out_w = width - (width % 2)
    out_h = height - (height % 2)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(out_clip, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_clip, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        # Final fallback to MJPG/AVI
        out_clip_avi = os.path.splitext(out_clip)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_clip_avi, fourcc, fps, (out_w, out_h))
        out_clip_local = out_clip_avi
    else:
        out_clip_local = out_clip

    count = 0
    while count < clip_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (frame.shape[1] != out_w) or (frame.shape[0] != out_h):
            frame = cv2.resize(frame, (out_w, out_h))
        writer.write(frame)
        count += 1

    writer.release()
    cap.release()

    print(f"Preview frames saved: {saved} -> {out_dir}")
    print(f"Preview clip saved: {out_clip_local}")


if __name__ == "__main__":
    main()
