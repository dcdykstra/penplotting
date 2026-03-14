import cv2
import numpy as np
import os


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


# --- 2. Setup Mouse Callback for Erasing ---
drawing = False
brush_size = 5


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


# --- 3. Main Processing Loop ---
# Load your frames
gif_frames = read_gif("images/raw/gif/shinmooning.gif")

# Create output directories if they don't exist
os.makedirs("shin/cleaned_canny", exist_ok=True)
os.makedirs("shin/final_contours", exist_ok=True)

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
    cv2.imwrite(f"shin/cleaned_canny/{i}.png", canny)

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
    cv2.imwrite(f"shin/final_contours/{i}.png", frame_copy)

# Cleanup when all frames are done
cv2.destroyAllWindows()
print("Finished processing all frames!")
