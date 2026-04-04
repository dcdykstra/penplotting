import cv2
import numpy as np
import os
import imageio
from skimage.morphology import skeletonize

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


def save_gif(frames, save_path, fps=15):
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    imageio.mimsave(save_path, rgb_frames, fps=fps, loop=0)


def loop_gif(gif_frames, n_total_frames=52):
    gif_frames_copy = gif_frames.copy()
    n_frames = len(gif_frames)

    looped_frames = []
    for loop in range(n_total_frames // n_frames):
        looped_frames.extend(gif_frames_copy)

    remainder = n_total_frames % n_frames
    if remainder > 0:
        looped_frames.extend(gif_frames_copy[:remainder])

    return looped_frames


def process_snoopy_gif():
    gif_frames = read_gif("images/raw/gif/snoopy_shuffle.gif")
    print(f"{len(gif_frames)=}")
    # Remove 1 frame so we can have 13 frames, 1 for each card
    gif_frames = gif_frames[:6] + gif_frames[-7:]
    print(f"{len(gif_frames)=}")
    save_gif(gif_frames, "snoopy/sliced_snoopy_shuffle.gif", fps=15)
    looped_frames = loop_gif(gif_frames, n_total_frames=52)
    save_gif(looped_frames, "snoopy/looped_snoopy_shuffle.gif", fps=15)

    hue = []
    sat = []
    val = []
    bin = []
    can = []
    skel = []
    hsv = []
    for i, frame in enumerate(gif_frames):
        print(f"Processing frame {i + 1}/{len(gif_frames)}...")

        # Crop and Preprocess
        frame = frame[24:, :, :]
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv.append(hsv_img)
        hue.append(hsv_img[:, :, 0])
        sat.append(hsv_img[:, :, 1])
        val.append(hsv_img[:, :, 2])
        cv2.imwrite(f"snoopy/hsv/{i}.png", hsv_img)
        cv2.imwrite(f"snoopy/hue/{i}.png", hsv_img[:, :, 0])
        cv2.imwrite(f"snoopy/sat/{i}.png", hsv_img[:, :, 1])
        cv2.imwrite(f"snoopy/val/{i}.png", hsv_img[:, :, 2])

        _, binary_image = cv2.threshold(
            hsv_img[:, :, 2], 140, 255, cv2.THRESH_BINARY_INV
        )
        bin.append(binary_image)
        cv2.imwrite(f"snoopy/bin/{i}.png", binary_image)

        # 2. Distance Transform
        # Calculates how far every white pixel is from a black pixel
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

        # 3. Isolate the "thick" cores (like the inside of the ear)
        # --- IMPORTANT: TUNE THIS NUMBER ---
        # If your normal drawn lines are 6 pixels thick, the max distance is ~3.
        # Set this threshold just above your max line radius (e.g., 4.0).
        thickness_threshold = 4.0
        _, thick_cores = cv2.threshold(
            dist_transform, thickness_threshold, 255, cv2.THRESH_BINARY
        )
        thick_cores = thick_cores.astype(np.uint8)

        # 4. Hollow out the original drawing
        # Subtracting the core turns solid filled areas into hollow outlines
        hollowed_drawing = cv2.subtract(binary_image, thick_cores)

        bool_image = hollowed_drawing > 0
        # Skeletonize thins the lines down to exactly 1 pixel wide
        skeleton = skeletonize(bool_image)
        skeleton_img = (skeleton * 255).astype(np.uint8)
        skel.append(skeleton_img)
        cv2.imwrite(f"snoopy/skel/{i}.png", skeleton_img)

        # Generate Canny edges
        canny = cv2.Canny(binary_image, 50, 200)
        can.append(canny)
        cv2.imwrite(f"snoopy/can/{i}.png", canny)

    save_gif(hue, "snoopy/hue/snoopy.gif", fps=15)
    save_gif(sat, "snoopy/sat/snoopy.gif", fps=15)
    save_gif(val, "snoopy/val/snoopy.gif", fps=15)
    save_gif(bin, "snoopy/bin/snoopy.gif", fps=15)
    save_gif(can, "snoopy/can/snoopy.gif", fps=15)
    save_gif(skel, "snoopy/skel/snoopy.gif", fps=15)
    save_gif(hsv, "snoopy/hsv/snoopy.gif", fps=15)


def clean_canny_edges_manual():
    gif_frames = read_gif("images/raw/gif/snoopy_shuffle.gif")
    print(f"{len(gif_frames)=}")
    # Remove 1 frame so we can have 13 frames, 1 for each card
    gif_frames = gif_frames[:6] + gif_frames[-7:]
    print(f"{len(gif_frames)=}")

    window_name = "Eraser Tool - Erase lines, then press ESC for next frame"
    cv2.namedWindow(window_name)

    print("--- Interactive Eraser Started ---")
    print("- Click and drag to erase connections to the frame boundaries.")
    print("- Press 'ESC' to save the frame and move to the next one.")

    for i, frame in enumerate(gif_frames):
        print(f"Processing frame {i + 1}/{len(gif_frames)}...")

        frame = frame[24:, :, :]
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        _, binary_image = cv2.threshold(hsv_img[:, :, 2], 140, 255, cv2.THRESH_BINARY)

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
        cv2.imwrite(f"snoopy/SAVE01_cleaned_canny/{i}.png", canny)

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
        cv2.imwrite(f"snoopy/SAVE02_final_contours/{i}.png", frame_copy)

    # Cleanup when all frames are done
    cv2.destroyAllWindows()
    print("Finished processing all frames!")
