"""Maki image → pen-plotter pipeline (static B&W manga image)."""

import cv2

from src.draw import build_contour_gcode, build_contour_svg
from src.io import read_image
from src.processing import find_contours, threshold_binary


def find_contours_manga_bw(img_path: str):
    """Convert a B&W manga image to contours."""
    img = read_image(img_path)
    img_name = img_path.split("/")[-1].split(".")[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"maki/{img_name}_grayscale.jpg", gray)

    binary = threshold_binary(gray, thresh=127)
    cv2.imwrite(f"maki/{img_name}_binary.jpg", binary)

    inverted = cv2.bitwise_not(binary)
    cv2.imwrite(f"maki/{img_name}_binary_inverted.jpg", inverted)

    contours, hierarchy = find_contours(inverted, mode=cv2.RETR_LIST)
    return contours, hierarchy


def process_maki_img():
    img_path = "images/raw/maki.jpg"
    contours, _ = find_contours_manga_bw(img_path)
    build_contour_svg(contours, img_path)
    build_contour_gcode(
        contours,
        scaling_factor=0.2,
        output_path="maki/maki_contour.gcode",
    )
