import cv2
import numpy as np
import os
from src.draw import build_contour_gcode

drawing = False
brush_size = 5


def read_gif(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def manual_eraser(event, x, y, flags, param):
    global drawing, brush_size
    image = param  # The canny image passed into the callback

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(image, (x, y), brush_size, 0, -1)  # 0 paints black

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(image, (x, y), brush_size, 0, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(image, (x, y), brush_size, 0, -1)


def clean_canny_edges_manual():
    # --- 3. Main Processing Loop ---
    # Load your frames
    gif_frames = read_gif("images/raw/gif/shinmooning.gif")

    # Create output directories if they don't exist
    os.makedirs("shin/SAVE01_cleaned_canny", exist_ok=True)
    os.makedirs("shin/SAVE02_final_contours", exist_ok=True)

    # Create the persistent window and attach the callback
    window_name = "Eraser Tool - Erase lines, then press ESC for next frame"
    cv2.namedWindow(window_name)

    print("--- Interactive Eraser Started ---")
    print("- Click and drag to erase connections to the frame boundaries.")
    print("- Press 'ESC' to save the frame and move to the next one.")

    for i, frame in enumerate(gif_frames):
        print(f"Processing frame {i + 1}/{len(gif_frames)}...")

        # Crop and Preprocess
        frame = frame[:, 120:-80, :]
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        _, binary_image = cv2.threshold(hsv_img[:, :, 2], 208, 255, cv2.THRESH_BINARY)

        # Generate Canny edges
        canny = cv2.Canny(binary_image, 50, 200)

        # Attach the current frame's canny image to the mouse callback
        cv2.setMouseCallback(window_name, manual_eraser, canny)

        # --- UI Loop for the Current Frame ---
        while True:
            cv2.imshow(window_name, canny)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # Break inner loop and move to next frame if ESC is pressed
            if key == 27:
                break

        # --- Process the manually cleaned frame ---
        # Save the cleaned edges
        cv2.imwrite(f"shin/SAVE01_cleaned_canny/{i}.png", canny)

        # Find contours on the cleaned image
        contours, hierarchy = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours (Bounding Box Check)
        filtered_contours = []
        height, width = frame.shape[:2]
        margin = 5

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            touches_left = x <= margin
            touches_right = x + w >= width - margin
            touches_top = y <= margin
            touches_bottom = y + h >= height - margin

            if touches_left or touches_right or touches_top or touches_bottom:
                continue

            filtered_contours.append(contour)

        # Draw the final isolated character contours onto a copy of the frame
        frame_copy = frame.copy()
        cv2.drawContours(frame_copy, filtered_contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"shin/SAVE02_final_contours/{i}.png", frame_copy)

    # Cleanup when all frames are done
    cv2.destroyAllWindows()
    print("Finished processing all frames!")


def process_shin_gif():
    gif_frames = read_gif("images/raw/gif/shinmooning.gif")
    for i, frame in enumerate(gif_frames):
        frame = frame[:, 120:-80, :]

        ### Read cleaned canny for sequential processing
        loaded_canny = cv2.imread(
            f"shin/SAVE01_cleaned_canny/{i}.jpg", cv2.IMREAD_GRAYSCALE
        )

        # Snap the JPEG noise back to 0 or 255
        _, canny = cv2.threshold(loaded_canny, 127, 255, cv2.THRESH_BINARY)

        canny = cv2.flip(canny, 1)
        contours, hierarchy = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ### Filter and draw contours
        filtered_contours = []
        height, width = frame.shape[:2]
        margin = 5
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            touches_left = x <= margin
            touches_right = x + w >= width - margin
            touches_top = y <= margin
            touches_bottom = y + h >= height - margin

            if touches_left or touches_right or touches_top or touches_bottom:
                continue

            filtered_contours.append(contour)

        blank_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(blank_mask, filtered_contours, -1, 255, 2)
        cv2.imwrite(f"shin/SAVE03_shin_contour/{i}.png", blank_mask)

        build_contour_gcode(filtered_contours, 0.1, output_path=f"shin/gcode/{i}.gcode")
        build_contour_gcode(
            filtered_contours,
            0.23166023166023167,
            output_path=f"shin/gcodebig/{i}.gcode",
        )

        ### Smooth and fuse mask to get the entire area around shin
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fused_mask = cv2.morphologyEx(blank_mask, cv2.MORPH_CLOSE, closing_kernel)

        # Now that the pieces are fused into one outline, find the contour of this new shape
        final_contours, _ = cv2.findContours(
            fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if final_contours:
            character_contour = max(final_contours, key=cv2.contourArea)
            solid_character_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(
                solid_character_mask, [character_contour], -1, 255, cv2.FILLED
            )
            cv2.imwrite(f"shin/SAVE04_solid_mask/{i}.png", solid_character_mask)

            final_frame = frame.copy()
            cv2.drawContours(final_frame, [character_contour], -1, (0, 0, 255), 3)
            cv2.imwrite(f"shin/SAVE05_character_outline/{i}.png", final_frame)

        ### HSV hue filtering for edges of background in the character outline
        hsv_img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2HSV)

        _, binary_image = cv2.threshold(hsv_img[:, :, 2], 208, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"shin/SAVE06_binary/{i}.png", binary_image)
        all_edges = cv2.Canny(binary_image, 50, 150)

        character_edges = cv2.bitwise_and(
            all_edges, all_edges, mask=solid_character_mask
        )
        cv2.imwrite(f"shin/SAVE07_character_edges/{i}.png", character_edges)

        isolated_color_character = cv2.bitwise_and(
            frame, frame, mask=solid_character_mask
        )
        cv2.imwrite(f"shin/SAVE08_isolated_character/{i}.png", isolated_color_character)

        hsv_image = cv2.cvtColor(isolated_color_character, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 30, 20])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

        lower_cyan = np.array([85, 30, 20])
        upper_cyan = np.array([130, 255, 255])
        mask_cyan = cv2.inRange(hsv_image, lower_cyan, upper_cyan)

        combined_bg_mask = cv2.bitwise_or(mask_green, mask_cyan)
        final_binary_mask = cv2.bitwise_not(combined_bg_mask)

        character_isolated_final = cv2.bitwise_and(
            isolated_color_character, isolated_color_character, mask=final_binary_mask
        )
        cv2.imwrite(
            f"shin/SAVE09_hsv_filtered_isolated_character/{i}.png",
            character_isolated_final,
        )
        cv2.imwrite(f"shin/SAVE10_final_binary_mask/{i}.png", final_binary_mask)

        ### Smooth image pixels to 5 colors
        char_pixels = character_isolated_final[final_binary_mask == 255]
        pixel_data = np.float32(char_pixels)

        # Stop the algorithm after 100 iterations OR if the centers move less than 0.2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Shin roughly has 5: Black (hair/shoes), Red (shirt), Peach (skin), Cream (pants), White (eyes/socks)
        K = 5

        # (cv2.KMEANS_RANDOM_CENTERS defines how the initial guesses are made)
        _, labels, centers = cv2.kmeans(
            pixel_data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Reconstruct the image with the simplified colors
        centers = np.uint8(centers)
        quantized_pixels = centers[labels.flatten()]

        # Create a blank (black) canvas with the exact same dimensions as your frame
        quantized_image = np.zeros_like(character_isolated_final)

        # Inject the simplified pixels directly back into the canvas,
        quantized_image[final_binary_mask == 255] = quantized_pixels
        cv2.imwrite(f"shin/SAVE11_quantized/{i}.png", quantized_image)
