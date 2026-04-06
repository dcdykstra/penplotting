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


def quantize_masked_colors(
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


def quantize_colors(
    images: list[np.ndarray],
    k: int = 5,
    max_iter: int = 100,
    epsilon: float = 0.1,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Reduce the number of colors across images using a shared K-means palette.

    Pools pixel data from all images, runs K-means once to compute a global
    palette, then quantizes each image against that palette.

    Args:
        images: List of BGR images (all must have the same number of channels).
        k: Number of color clusters.
        max_iter: Maximum K-means iterations.
        epsilon: K-means convergence epsilon.

    Returns:
        A tuple of (quantized_images, global_centers) where
        ``quantized_images`` is a list of images recolored to the shared
        palette and ``global_centers`` is the (k, channels) uint8 array of
        palette colors.
    """
    # 1. Pool all pixel data from every image into one array
    all_pixels = np.vstack(
        [img.reshape(-1, img.shape[2] if img.ndim == 3 else 1) for img in images]
    ).astype(np.float32)

    # 2. Run K-means once on the combined pixels to get a shared palette
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    _, _, global_centers = cv2.kmeans(
        all_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    global_centers = np.uint8(global_centers)

    # 3. Quantize each image using the shared palette
    quantized_images = []
    for img in images:
        pixel_data = img.reshape((-1, img.shape[2] if img.ndim == 3 else 1)).astype(
            np.float32
        )
        distances = np.linalg.norm(
            pixel_data[:, None] - global_centers[None, :].astype(np.float32), axis=2
        )
        labels = np.argmin(distances, axis=1)
        result = global_centers[labels].reshape(img.shape)
        quantized_images.append(result)

    return quantized_images, global_centers


def generate_crosshatch_lines(contour, spacing=5, angles=[45, -45]):
    """
    Generate cross-hatching lines (as coordinate pairs) that fill the interior
    of a contour. Returns a list of line segments, each as [(x1,y1), (x2,y2)].

    Parameters:
        contour: a single contour from cv2.findContours
        spacing: distance in pixels between parallel hatch lines
        angles: list of angles (degrees) for hatching directions
                 e.g. [45] for single-direction, [45, -45] for cross-hatch

    Returns:
        hatch_lines: list of line segments [((x1,y1), (x2,y2)), ...]
    """
    # Get bounding rect to know the scan range
    x, y, w, h = cv2.boundingRect(contour)

    # Create a filled mask of just this contour
    # Use a tight canvas around the bounding rect for efficiency
    mask = np.zeros((y + h + 1, x + w + 1), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    hatch_lines = []

    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Center of the bounding box (rotation pivot)
        cx, cy = x + w / 2.0, y + h / 2.0

        # The diagonal of the bounding box — guarantees full coverage after rotation
        diag = int(np.ceil(np.sqrt(w**2 + h**2)))

        # Generate parallel lines perpendicular to the angle direction
        # We sweep offsets from -diag/2 to +diag/2
        for offset in np.arange(-diag / 2, diag / 2, spacing):
            # A line at this offset, rotated by `angle`:
            # The line runs in the direction of `angle`, shifted perpendicular by `offset`
            # Line parametric: P(t) = center + t*(cos, sin) + offset*(-sin, cos)
            # We just need two endpoints far enough apart
            t_range = diag  # half-length of the scan line

            # Perpendicular offset direction
            px = -sin_a * offset
            py = cos_a * offset

            # Two endpoints of the scan line (long enough to cross the entire shape)
            x1 = cx + cos_a * (-t_range) + px
            y1 = cy + sin_a * (-t_range) + py
            x2 = cx + cos_a * (t_range) + px
            y2 = cy + sin_a * (t_range) + py

            # Now clip this line to the filled contour mask by sampling along it
            # Walk along the line and find segments that are inside the mask
            num_samples = int(2 * t_range)
            if num_samples < 2:
                continue

            ts = np.linspace(0, 1, num_samples)
            xs = (x1 + (x2 - x1) * ts).astype(int)
            ys = (y1 + (y2 - y1) * ts).astype(int)

            # Check bounds and mask membership
            in_bounds = (
                (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
            )
            inside = np.zeros(len(ts), dtype=bool)
            valid_idx = np.where(in_bounds)[0]
            inside[valid_idx] = mask[ys[valid_idx], xs[valid_idx]] > 0

            # Extract contiguous segments of "inside" points
            # Find transitions (entering / leaving the mask)
            diff = np.diff(inside.astype(int))
            starts = np.where(diff == 1)[0] + 1  # entering mask
            ends = np.where(diff == -1)[0]  # leaving mask

            # Handle edge case: line starts inside
            if inside[0]:
                starts = np.concatenate(([0], starts))
            # Handle edge case: line ends inside
            if inside[-1]:
                ends = np.concatenate((ends, [len(inside) - 1]))

            for s, e in zip(starts, ends):
                if e > s:
                    seg_start = (float(xs[s]), float(ys[s]))
                    seg_end = (float(xs[e]), float(ys[e]))
                    hatch_lines.append((seg_start, seg_end))

    return hatch_lines
