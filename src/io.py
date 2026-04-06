"""I/O utilities for reading/writing images, GIFs, and frames."""

import cv2
import imageio
import numpy as np
import os


def list_files_walk(start_path="."):
    out = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if str(os.path.join(root, file)).endswith(".png"):
                out.append(os.path.join(root, file))
    return out


def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    """Read an image from disk."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_gif(path: str) -> list[np.ndarray]:
    """Read all frames from a GIF or video file."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_gif(frames: list[np.ndarray], save_path: str, fps: int = 15) -> None:
    """Save a list of BGR frames as a GIF."""
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(save_path, rgb_frames, fps=fps, loop=0)


def loop_gif(frames: list[np.ndarray], n_total_frames: int) -> list[np.ndarray]:
    """Loop a set of frames to fill n_total_frames by repeating them."""
    frames_copy = frames.copy()
    n_frames = len(frames_copy)

    looped = []
    for _ in range(n_total_frames // n_frames):
        looped.extend(frames_copy)

    remainder = n_total_frames % n_frames
    if remainder > 0:
        looped.extend(frames_copy[:remainder])

    return looped
