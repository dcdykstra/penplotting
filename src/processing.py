"""Shared image processing utilities for the penplotting pipeline."""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def to_hsv(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to HSV."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def threshold_binary(
    image: np.ndarray, thresh: int = 127, invert: bool = False
) -> np.ndarray:
    """Apply a binary threshold.

    Args:
        image: Single-channel grayscale image.
        thresh: Threshold value.
        invert: If True, use THRESH_BINARY_INV (white below thresh).
    """
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(image, thresh, 255, mode)
    return binary


def canny_edges(image: np.ndarray, low: int = 50, high: int = 200) -> np.ndarray:
    """Run Canny edge detection."""
    return cv2.Canny(image, low, high)


def find_contours(
    image: np.ndarray,
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE,
) -> tuple:
    """Find contours in a binary image."""
    contours, hierarchy = cv2.findContours(image, mode, method)
    return contours, hierarchy


def filter_contours_by_margin(
    contours: list,
    frame_height: int,
    frame_width: int,
    margin: int = 5,
) -> list:
    """Remove contours that touch the edges of the frame.

    Contours whose bounding rectangle is within `margin` pixels of any
    border are discarded.
    """
    filtered = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (
            x <= margin
            or x + w >= frame_width - margin
            or y <= margin
            or y + h >= frame_height - margin
        ):
            continue
        filtered.append(contour)
    return filtered


def hollow_and_skeletonize(
    binary: np.ndarray, thickness_threshold: float = 4.0
) -> np.ndarray:
    """Remove thick filled regions and skeletonize the remaining lines.

    1. Distance transform to find thick interior regions.
    2. Subtract thick cores to hollow out filled areas.
    3. Skeletonize to 1-pixel-wide lines.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, thick_cores = cv2.threshold(dist, thickness_threshold, 255, cv2.THRESH_BINARY)
    thick_cores = thick_cores.astype(np.uint8)

    hollowed = cv2.subtract(binary, thick_cores)
    skeleton = skeletonize(hollowed > 0)
    return (skeleton * 255).astype(np.uint8)


def fuse_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Close small gaps in a binary mask using morphological closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def largest_contour_filled_mask(
    binary: np.ndarray,
) -> np.ndarray | None:
    """Find the largest contour and return a filled mask of it.

    Returns None if no contours are found.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    return mask


def isolate_by_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out pixels outside a binary mask."""
    return cv2.bitwise_and(image, image, mask=mask)


def remove_colors_hsv(
    image: np.ndarray,
    hsv_ranges: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Create a binary mask that excludes pixels matching any HSV range.

    Args:
        image: BGR image.
        hsv_ranges: List of (lower_hsv, upper_hsv) array pairs to exclude.

    Returns:
        Binary mask where 255 = keep, 0 = excluded color.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    combined = np.zeros(image.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
    return cv2.bitwise_not(combined)


def quantize_colors(
    image: np.ndarray,
    mask: np.ndarray,
    k: int = 5,
    max_iter: int = 100,
    epsilon: float = 0.2,
) -> np.ndarray:
    """Reduce the number of colors in an image using K-means clustering.

    Only pixels where `mask == 255` are clustered; the rest remain black.
    """
    pixels = image[mask == 255]
    pixel_data = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    _, labels, centers = cv2.kmeans(
        pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    quantized_pixels = centers[labels.flatten()]

    result = np.zeros_like(image)
    result[mask == 255] = quantized_pixels
    return result
