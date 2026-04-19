import cv2
import os
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: int = 3):
    """
    Extract frames from a video at given FPS rate.
    Also handles single images directly.
    Returns list of saved frame paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Handle image files directly
    ext = os.path.splitext(video_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        import shutil
        out_path = os.path.join(output_dir, 'frame_0000.jpg')
        shutil.copy(video_path, out_path)
        return [out_path]

    # Handle video files
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 25
    interval = max(1, int(video_fps // fps))

    frame_paths = []
    count       = 0
    saved       = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved += 1
        count += 1

    cap.release()
    print(f"Extracted {saved} frames from {os.path.basename(video_path)}")
    return frame_paths