import cv2
import pandas as pd
import numpy as np


def find_countours_manga_bw(img_path):
    img = cv2.imread(img_path)
    img_name = img_path.split("/")[-1].split(".")[0]
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"images/output/{img_name}_grayscale.jpg", gray_image)

    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"images/output/{img_name}_binary.jpg", binary_image)

    # 2. Invert the image (Black becomes White, White becomes Black)
    inverted_image = cv2.bitwise_not(binary_image)
    cv2.imwrite(f"images/output/{img_name}_binary_inverted.jpg", inverted_image)

    contours, hierarchy = cv2.findContours(
        inverted_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, hierarchy


def build_contour_svg(contours, img_path):
    img = cv2.imread(img_path)
    img_name = img_path.split("/")[-1].split(".")[0]

    svg_paths = []
    for contour in contours:
        path_string = ""
        for i, point in enumerate(contour):
            if i == 0:
                prefix = "M"
            else:
                prefix = "L"
            x = point[0][0]
            y = point[0][1]

            path_string += f" {prefix} {x} {y}"

        path_string += " Z"  # Seal the shape at the end
        svg_paths.append(path_string)

    with open(f"images/output/{img_name}_contour.svg", "w") as file:
        start = f'<svg xmlns="http://www.w3.org/2000/svg" width="{img.shape[1]}" height="{img.shape[0]}">'
        file.write(start)
        for path in svg_paths:
            path_string = (
                '\n\t<path d="' + path + '"' + ' stroke="black" fill="none" />'
            )
            file.write(path_string)
        file.write("\n</svg>")


def build_contour_gcode(contours, img_path):
    img_name = img_path.split("/")[-1].split(".")[0]

    path_string = "G21\nG90\nF3000"
    scaling_factor = 0.2
    for contour in contours:
        for i, point in enumerate(contour):
            x = point[0][0] * scaling_factor
            y = point[0][1] * scaling_factor
            if i == 0:
                path_string += "\nM5\nM5\nG4 P0.3"
                path_string += f"\nG0 X{x} Y{y} F3000"
                path_string += f"\nM3 S1000\nG4 P0.3"
            else:
                path_string += f"\nG1 X{x} Y{y} F3000"
    path_string += "\nM5\nG4 P0.3\nG0 X0 Y0 Z0"

    with open(f"images/output/{img_name}_contour.gcode", "w") as file:
        file.write(path_string)


if __name__ == "__main__":
    img_path = "images/raw/maki.jpg"
    contours, hierarchy = find_countours_manga_bw(img_path)
    build_contour_svg(contours, img_path)
    build_contour_gcode(contours, img_path)
