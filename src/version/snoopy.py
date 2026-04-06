"""Snoopy GIF → pen-plotter pipeline."""

import cv2
import numpy as np
import os

from src.interactive import clean_canny_edges_manual
from src.io import read_gif, save_gif, loop_gif, list_files_walk
from src.processing import (
    hollow_and_skeletonize,
    quantize_colors,
    threshold_binary,
    to_hsv,
    find_contours,
    generate_crosshatch_lines,
)
from src.draw import build_contour_gcode, hatch_lines_to_contour_format


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


def remap_colors(frames, color_map, save_dir=None, gif_path=None, fps=15):
    """
    Replace quantized colors in frames with new colors.

    Args:
        frames: list of quantized BGR images (numpy arrays).
        color_map: dict mapping old BGR tuples to new BGR tuples.
                   e.g. {(234, 209, 188): (255, 255, 255)}
        save_dir: optional directory to save individual frames.
        gif_path: optional path to save the remapped GIF.
        fps: frames per second for the GIF.

    Returns:
        list of remapped frames.
    """
    remapped = []
    for i, frame in enumerate(frames):
        result = frame.copy()
        for old_color, new_color in color_map.items():
            # Create a mask where all 3 channels match the old color
            mask = np.all(result == np.array(old_color, dtype=np.uint8), axis=2)
            result[mask] = new_color
        remapped.append(result)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f"{save_dir}/{i}.png", result)

    if gif_path:
        save_gif(remapped, gif_path, fps=fps)

    return remapped


def process_snoopy_gif():
    gif_frames = read_gif("images/raw/gif/snoopy_shuffle.gif")
    gif_frames = _slice_snoopy_frames(gif_frames)

    save_gif(gif_frames, "snoopy/sliced_snoopy_shuffle.gif", fps=15)
    looped_frames = loop_gif(gif_frames, n_total_frames=52)
    save_gif(looped_frames, "snoopy/looped_snoopy_shuffle.gif", fps=15)

    # Collect per-channel outputs for GIF export
    channel_frames = {
        "skel": [],
    }

    ### Plot skeletonized Snoopy
    for i, frame in enumerate(gif_frames):
        print(f"Processing frame {i + 1}/{len(gif_frames)}...")

        frame = _crop_snoopy(frame)
        hsv_img = to_hsv(frame)

        binary = threshold_binary(hsv_img[:, :, 2], thresh=140, invert=True)
        skeleton = hollow_and_skeletonize(binary, thickness_threshold=4.0)
        channel_frames["skel"].append(skeleton)
        cv2.imwrite(f"snoopy/skel/{i}.png", skeleton)

    # Save channel GIFs
    for name, frames in channel_frames.items():
        save_gif(frames, f"snoopy/{name}/snoopy.gif", fps=15)

    ### Quantize Snoopy Colors to 5 colors
    cropped_frames = [frame[24:, :, :] for frame in gif_frames]

    quantized, global_centers = quantize_colors(
        cropped_frames, k=5, max_iter=100, epsilon=0.1
    )
    print(f"Global palette (BGR): {global_centers.tolist()}")

    for i, result in enumerate(quantized):
        cv2.imwrite(f"snoopy/quantized/{i}.png", result)

    save_gif(quantized, "snoopy/quantized/snoopy.gif", fps=15)

    # Define your color replacements: old BGR -> new BGR
    color_map = {
        (234, 209, 188): (255, 255, 255),
        (12, 52, 69): (0, 0, 255),
    }

    remapped = remap_colors(
        quantized,
        color_map,
        save_dir="snoopy/remapped",
        gif_path="snoopy/remapped/snoopy.gif",
        fps=15,
    )
    ### Clean color masks with interactive.interactive_layer_editor()

    ### Find contours and fill in areas for each color mask of each frame
    directory_path = "snoopy/layers/"
    files = list_files_walk(directory_path)

    for i, file in enumerate(files):
        layer = file.split("/")[-1][6]
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # Ensure it's a clean binary mask (uint8)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = find_contours(binary)

        height, width = binary.shape[:2]
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if layer == "0":
            hatch_lines = generate_crosshatch_lines(
                sorted_contours[0], spacing=4, angles=[45, -45]
            )
            hatch_lines_1 = generate_crosshatch_lines(
                sorted_contours[1], spacing=4, angles=[45, -45]
            )
            hatch_lines_2 = generate_crosshatch_lines(
                sorted_contours[2], spacing=4, angles=[45, -45]
            )
            hatch_contours = hatch_lines_to_contour_format(
                hatch_lines + hatch_lines_1 + hatch_lines_2
            )
            filled_contours = list(contours) + hatch_contours
        if layer == "1":
            filled_contours = contours
        if layer == "2":
            hatch_lines = generate_crosshatch_lines(
                sorted_contours[0], spacing=4, angles=[45, -45]
            )
            hatch_lines_2 = generate_crosshatch_lines(
                sorted_contours[1], spacing=4, angles=[45, -45]
            )
            hatch_contours = hatch_lines_to_contour_format(hatch_lines + hatch_lines_2)
            filled_contours = list(contours) + hatch_contours
        if layer == "3":
            hatch_lines = generate_crosshatch_lines(
                sorted_contours[0], spacing=4, angles=[45, -45]
            )
            hatch_lines_2 = generate_crosshatch_lines(
                sorted_contours[1], spacing=4, angles=[45, -45]
            )
            hatch_contours = hatch_lines_to_contour_format(hatch_lines + hatch_lines_2)
            filled_contours = list(contours) + hatch_contours
        if layer == "4":
            continue

        blank_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(blank_mask, filled_contours, -1, 255, 2)

        path = file.replace("layers", "layers_contour_filled")
        cv2.imwrite(path, blank_mask)
        build_contour_gcode(
            filled_contours,
            0.20,
            output_path=file.replace("layers", "gcode_filled").replace(
                ".png", ".gcode"
            ),
        )
