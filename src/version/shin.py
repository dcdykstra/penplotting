"""Shin-chan GIF → pen-plotter pipeline."""

import cv2
import numpy as np

from src.draw import build_contour_gcode
from src.interactive import clean_canny_edges_manual
from src.io import read_gif, read_image
from src.processing import (
    canny_edges,
    filter_contours_by_margin,
    find_contours,
    fuse_mask,
    isolate_by_mask,
    largest_contour_filled_mask,
    quantize_colors,
    remove_colors_hsv,
    threshold_binary,
    to_hsv,
)


def _crop_shin(frame: np.ndarray) -> np.ndarray:
    """Crop Shin-chan frames (remove left/right padding)."""
    return frame[:, 120:-80, :]


def clean_shin_edges_manual():
    """Launch the interactive eraser for Shin-chan frames."""
    clean_canny_edges_manual(
        gif_path="images/raw/gif/shinmooning.gif",
        output_dir="shin",
        crop_fn=_crop_shin,
        hsv_channel=2,
        binary_thresh=208,
        invert_thresh=False,
    )


def process_shin_gif():
    gif_frames = read_gif("images/raw/gif/shinmooning.gif")

    for i, frame in enumerate(gif_frames):
        frame = _crop_shin(frame)

        # Read cleaned canny for sequential processing
        loaded_canny = read_image(f"shin/SAVE01_cleaned_canny/{i}.jpg", grayscale=True)

        # Snap JPEG noise back to 0 or 255
        canny = threshold_binary(loaded_canny, thresh=127)
        canny = cv2.flip(canny, 1)

        contours, _ = find_contours(canny)

        # Filter and draw contours
        height, width = frame.shape[:2]
        filtered_contours = filter_contours_by_margin(contours, height, width)

        blank_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(blank_mask, filtered_contours, -1, 255, 2)
        cv2.imwrite(f"shin/SAVE03_shin_contour/{i}.png", blank_mask)

        build_contour_gcode(filtered_contours, 0.1, output_path=f"shin/gcode/{i}.gcode")
        build_contour_gcode(
            filtered_contours,
            0.23166023166023167,
            output_path=f"shin/gcodebig/{i}.gcode",
        )

        # Fuse mask and extract largest contour
        fused = fuse_mask(blank_mask, kernel_size=15)
        solid_character_mask = largest_contour_filled_mask(fused)

        if solid_character_mask is not None:
            cv2.imwrite(f"shin/SAVE04_solid_mask/{i}.png", solid_character_mask)

            # Draw character outline on original frame
            outline_contours, _ = find_contours(solid_character_mask)
            if outline_contours:
                character_contour = max(outline_contours, key=cv2.contourArea)
                final_frame = frame.copy()
                cv2.drawContours(final_frame, [character_contour], -1, (0, 0, 255), 3)
                cv2.imwrite(f"shin/SAVE05_character_outline/{i}.png", final_frame)

        # HSV-based processing for character isolation
        hsv_img = to_hsv(cv2.flip(frame, 1))
        binary = threshold_binary(hsv_img[:, :, 2], thresh=208)
        cv2.imwrite(f"shin/SAVE06_binary/{i}.png", binary)

        all_edges = canny_edges(binary, low=50, high=150)
        character_edges = isolate_by_mask(all_edges, solid_character_mask)
        cv2.imwrite(f"shin/SAVE07_character_edges/{i}.png", character_edges)

        isolated = isolate_by_mask(frame, solid_character_mask)
        cv2.imwrite(f"shin/SAVE08_isolated_character/{i}.png", isolated)

        # Remove green/cyan background within the character mask
        bg_ranges = [
            (np.array([35, 30, 20]), np.array([85, 255, 255])),  # green
            (np.array([85, 30, 20]), np.array([130, 255, 255])),  # cyan
        ]
        final_binary_mask = remove_colors_hsv(isolated, bg_ranges)
        character_isolated_final = isolate_by_mask(isolated, final_binary_mask)
        cv2.imwrite(
            f"shin/SAVE09_hsv_filtered_isolated_character/{i}.png",
            character_isolated_final,
        )
        cv2.imwrite(f"shin/SAVE10_final_binary_mask/{i}.png", final_binary_mask)

        # Color quantization
        quantized = quantize_colors(character_isolated_final, final_binary_mask, k=5)
        cv2.imwrite(f"shin/SAVE11_quantized/{i}.png", quantized)
