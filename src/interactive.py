"""Interactive OpenCV tools for manual image editing."""

import cv2
import numpy as np

from src.io import read_gif
from src.processing import (
    canny_edges,
    filter_contours_by_margin,
    find_contours,
    threshold_binary,
    to_hsv,
)

# Shared state for the mouse callback
_drawing = False
_brush_size = 5


def _manual_eraser(event, x, y, flags, param):
    """Mouse callback: paint black circles to erase edges."""
    global _drawing, _brush_size
    image = param

    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        cv2.circle(image, (x, y), _brush_size, 0, -1)
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        cv2.circle(image, (x, y), _brush_size, 0, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        cv2.circle(image, (x, y), _brush_size, 0, -1)


def clean_canny_edges_manual(
    gif_path: str,
    output_dir: str,
    crop_fn=None,
    hsv_channel: int = 2,
    binary_thresh: int = 140,
    invert_thresh: bool = False,
    frame_slice: slice | None = None,
):
    """Interactive eraser for cleaning Canny edges on GIF frames.

    Presents each frame's Canny edges in a window. The user erases
    unwanted connections, then presses ESC to advance.

    Args:
        gif_path: Path to the source GIF.
        output_dir: Base directory for saves (e.g. "snoopy" or "shin").
        crop_fn: Optional callable ``crop_fn(frame) -> frame`` to crop
            each frame before processing.
        hsv_channel: HSV channel index to threshold (default 2 = Value).
        binary_thresh: Threshold value for binarisation.
        invert_thresh: Whether to invert the binary threshold.
        frame_slice: Optional slice to select a subset of frames
            (applied before processing).
    """
    gif_frames = read_gif(gif_path)
    print(f"{len(gif_frames)=}")

    if frame_slice is not None:
        gif_frames = gif_frames[frame_slice]
        print(f"After slicing: {len(gif_frames)=}")

    window_name = "Eraser Tool - Erase lines, then press ESC for next frame"
    cv2.namedWindow(window_name)

    print("--- Interactive Eraser Started ---")
    print("- Click and drag to erase connections to the frame boundaries.")
    print("- Press 'ESC' to save the frame and move to the next one.")

    for i, frame in enumerate(gif_frames):
        print(f"Processing frame {i + 1}/{len(gif_frames)}...")

        if crop_fn is not None:
            frame = crop_fn(frame)

        hsv_img = to_hsv(frame)
        binary = threshold_binary(
            hsv_img[:, :, hsv_channel], binary_thresh, invert=invert_thresh
        )
        canny = canny_edges(binary)

        cv2.setMouseCallback(window_name, _manual_eraser, canny)

        while True:
            cv2.imshow(window_name, canny)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        # Save cleaned edges
        cv2.imwrite(f"{output_dir}/SAVE01_cleaned_canny/{i}.png", canny)

        # Find and filter contours
        contours, _ = find_contours(canny)
        h, w = frame.shape[:2]
        filtered = filter_contours_by_margin(contours, h, w)

        # Draw contours onto a copy of the frame
        frame_copy = frame.copy()
        cv2.drawContours(frame_copy, filtered, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{output_dir}/SAVE02_final_contours/{i}.png", frame_copy)

    cv2.destroyAllWindows()
    print("Finished processing all frames!")
