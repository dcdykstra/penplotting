"""Snoopy GIF → pen-plotter pipeline."""

import cv2
import numpy as np

from src.interactive import clean_canny_edges_manual
from src.io import read_gif, save_gif, loop_gif
from src.processing import (
    canny_edges,
    hollow_and_skeletonize,
    threshold_binary,
    to_hsv,
)


def _crop_snoopy(frame: np.ndarray) -> np.ndarray:
    """Crop Snoopy frames (remove top padding)."""
    return frame[24:, :, :]


def _slice_snoopy_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Remove 1 frame so we have 13 frames, 1 for each card."""
    return frames[:6] + frames[-7:]


def clean_snoopy_edges_manual():
    """Launch the interactive eraser for Snoopy frames."""
    clean_canny_edges_manual(
        gif_path="images/raw/gif/snoopy_shuffle.gif",
        output_dir="snoopy",
        crop_fn=_crop_snoopy,
        hsv_channel=2,
        binary_thresh=140,
        invert_thresh=False,
    )


def process_snoopy_gif():
    gif_frames = read_gif("images/raw/gif/snoopy_shuffle.gif")
    print(f"{len(gif_frames)=}")

    gif_frames = _slice_snoopy_frames(gif_frames)
    print(f"{len(gif_frames)=}")

    save_gif(gif_frames, "snoopy/sliced_snoopy_shuffle.gif", fps=15)
    looped_frames = loop_gif(gif_frames, n_total_frames=52)
    save_gif(looped_frames, "snoopy/looped_snoopy_shuffle.gif", fps=15)

    # Collect per-channel outputs for GIF export
    channel_frames = {
        "hsv": [],
        "hue": [],
        "sat": [],
        "val": [],
        "bin": [],
        "can": [],
        "skel": [],
    }

    for i, frame in enumerate(gif_frames):
        print(f"Processing frame {i + 1}/{len(gif_frames)}...")

        frame = _crop_snoopy(frame)
        hsv_img = to_hsv(frame)

        channel_frames["hsv"].append(hsv_img)
        channel_frames["hue"].append(hsv_img[:, :, 0])
        channel_frames["sat"].append(hsv_img[:, :, 1])
        channel_frames["val"].append(hsv_img[:, :, 2])

        cv2.imwrite(f"snoopy/hsv/{i}.png", hsv_img)
        cv2.imwrite(f"snoopy/hue/{i}.png", hsv_img[:, :, 0])
        cv2.imwrite(f"snoopy/sat/{i}.png", hsv_img[:, :, 1])
        cv2.imwrite(f"snoopy/val/{i}.png", hsv_img[:, :, 2])

        binary = threshold_binary(hsv_img[:, :, 2], thresh=140, invert=True)
        channel_frames["bin"].append(binary)
        cv2.imwrite(f"snoopy/bin/{i}.png", binary)

        skeleton = hollow_and_skeletonize(binary, thickness_threshold=4.0)
        channel_frames["skel"].append(skeleton)
        cv2.imwrite(f"snoopy/skel/{i}.png", skeleton)

        canny = canny_edges(binary)
        channel_frames["can"].append(canny)
        cv2.imwrite(f"snoopy/can/{i}.png", canny)

    # Save channel GIFs
    for name, frames in channel_frames.items():
        save_gif(frames, f"snoopy/{name}/snoopy.gif", fps=15)
