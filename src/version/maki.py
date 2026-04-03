import cv2
from src.draw import build_contour_gcode, build_contour_svg


def find_countours_manga_bw(img_path):
    img = cv2.imread(img_path)
    img_name = img_path.split("/")[-1].split(".")[0]
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"maki/{img_name}_grayscale.jpg", gray_image)

    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"maki/{img_name}_binary.jpg", binary_image)

    inverted_image = cv2.bitwise_not(binary_image)
    cv2.imwrite(f"maki/{img_name}_binary_inverted.jpg", inverted_image)

    contours, hierarchy = cv2.findContours(
        inverted_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, hierarchy


def process_maki_img():
    img_path = "images/raw/maki.jpg"
    contours, hierarchy = find_countours_manga_bw(img_path)
    build_contour_svg(contours, img_path)
    build_contour_gcode(
        contours,
        scaling_factor=0.2,
        output_path=f"maki/maki_contour.gcode",
    )
