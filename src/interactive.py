"""Interactive OpenCV tools for manual image editing."""

import cv2
import numpy as np
import os

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


def interactive_layer_editor(frames, save_dir="snoopy/layers", brush_size=10):
    """
    For each frame, isolate each unique color into a binary mask layer,
    open an interactive cv2 window for manual erasing, then save the
    cleaned mask.

    Controls:
        - Left-click + drag: erase (paint black) to remove noise
        - Right-click + drag: restore (paint white) to undo erasing
        - Up arrow: increase brush size
        - Down arrow: decrease brush size
        - 'r': reset the current layer to the original mask
        - ESC: save and advance to the next layer/frame

    Saves:
        {save_dir}/frame_{i}/layer_{j}_bgr_{B}_{G}_{R}.png  (binary mask)
    """
    drawing = False
    draw_color = 0  # 0 = erase, 255 = restore
    brush = [brush_size]  # mutable so callback can modify

    def _mouse_callback(event, x, y, flags, param):
        nonlocal drawing, draw_color
        image = param

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            draw_color = 0  # erase
            cv2.circle(image, (x, y), brush[0], draw_color, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            draw_color = 255  # restore
            cv2.circle(image, (x, y), brush[0], draw_color, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(image, (x, y), brush[0], draw_color, -1)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False

    window_name = (
        "Layer Editor | L-click=erase, R-click=restore, Up/Down=brush, ESC=next"
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("--- Interactive Layer Editor ---")
    print("  Left-click:  erase pixels")
    print("  Right-click: restore pixels")
    print("  Up arrow:    increase brush size")
    print("  Down arrow:  decrease brush size")
    print("  'r':         reset layer to original")
    print("  ESC:         save layer & advance")
    print()

    for i, frame in enumerate(frames):
        unique_colors = np.unique(frame.reshape(-1, 3), axis=0)
        frame_dir = f"{save_dir}/frame_{i}"
        os.makedirs(frame_dir, exist_ok=True)
        if i <= 3:
            continue
        print(f"Frame {i + 1}/{len(frames)} — {len(unique_colors)} layers")

        for j, color in enumerate(unique_colors):
            color_tuple = tuple(color.tolist())
            b, g, r = color_tuple
            layer_name = f"layer_{j}_bgr_{b}_{g}_{r}"
            print(f"  Layer {j + 1}/{len(unique_colors)}: BGR({b}, {g}, {r})")

            # Create binary mask for this color
            original_mask = np.all(frame == color, axis=2).astype(np.uint8) * 255
            mask = original_mask.copy()

            cv2.setMouseCallback(window_name, _mouse_callback, mask)

            while True:
                # Show the mask with a small info bar
                display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                # Tint the display with the actual color for reference
                tinted = display.copy()
                tinted[mask > 0] = color
                cv2.putText(
                    tinted,
                    f"Frame {i} | Layer {j}: BGR({b},{g},{r}) | Brush: {brush[0]}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow(window_name, tinted)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC — save and next
                    break
                elif key == ord("r"):  # reset
                    np.copyto(mask, original_mask)
                    print("    Layer reset.")
                elif key == 0:  # Up arrow
                    brush[0] = min(brush[0] + 2, 100)
                    print(f"  Brush size: {brush[0]}")
                elif key == 1:  # Down arrow
                    brush[0] = max(brush[0] - 2, 1)
                    print(f"  Brush size: {brush[0]}")

            # Save the cleaned mask
            save_path = f"{frame_dir}/{layer_name}.png"
            cv2.imwrite(save_path, mask)
            print(f"    Saved: {save_path}")

    cv2.destroyAllWindows()
    print("\nAll layers saved!")
