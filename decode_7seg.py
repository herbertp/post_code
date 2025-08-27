import cv2
import numpy as np
import sys
import time

def get_digit_from_roi(digit_roi):
    """
    Takes a clean, tight bounding box of a single digit,
    samples points, and returns the decoded character.
    """
    h, w = digit_roi.shape
    if h < 10 or w < 10:
        return '?'

    DIGITS_LOOKUP = {
        (1, 1, 1, 1, 1, 1, 0): '0', (0, 1, 1, 0, 0, 0, 0): '1',
        (1, 1, 0, 1, 1, 0, 1): '2', (1, 1, 1, 1, 0, 0, 1): '3',
        (0, 1, 1, 0, 0, 1, 1): '4', (1, 0, 1, 1, 0, 1, 1): '5',
        (1, 0, 1, 1, 1, 1, 1): '6', (1, 1, 1, 0, 0, 0, 0): '7',
        (1, 1, 1, 1, 1, 1, 1): '8', (1, 1, 1, 1, 0, 1, 1): '9',
        (1, 1, 1, 0, 1, 1, 1): 'A', (0, 0, 1, 1, 1, 1, 1): 'b',
        (1, 0, 0, 1, 1, 1, 0): 'C', (0, 1, 1, 1, 1, 0, 1): 'd',
        (1, 0, 0, 1, 1, 1, 1): 'E', (1, 0, 0, 0, 1, 1, 1): 'F'
    }

    # Proportional coordinates within this digit's specific bounding box
    y_top = h * 0.20; y_mid = h * 0.5; y_bot = h * 0.80
    x_left = w * 0.20; x_mid = w * 0.5; x_right = w * 0.80

    segment_centers = [
        (y_top, x_mid), (y_top, x_right), (y_bot, x_right), (y_bot, x_mid),
        (y_bot, x_left), (y_top, x_left), (y_mid, x_mid)
    ]

    segments = []
    for y, x in segment_centers:
        # Check a small 3x3 area around the center for robustness
        is_on = False
        for y_offset in range(-1, 2):
            for x_offset in range(-1, 2):
                y_s, x_s = int(y + y_offset), int(x + x_offset)
                if 0 <= y_s < h and 0 <= x_s < w:
                    if digit_roi[y_s, x_s] > 128:
                        is_on = True
                        break
            if is_on:
                break
        segments.append(1 if is_on else 0)

    return DIGITS_LOOKUP.get(tuple(segments), '?')


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # --- Step 1: Analyze first N frames to find a single, stable bounding box for the whole display ---
    N_FRAMES_FOR_BOX = 100
    frame_count = 0
    union_box = None

    while cap.isOpened() and frame_count < N_FRAMES_FOR_BOX:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        h, w, _ = frame.shape
        roi = frame[h//3:2*h//3, w//3:2*w//3]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.concatenate(contours)
            x, y, w_box, h_box = cv2.boundingRect(all_points)
            if w_box > 10 and h_box > 10:
                if union_box is None: union_box = [x, y, x + w_box, y + h_box]
                else:
                    union_box[0] = min(union_box[0], x)
                    union_box[1] = min(union_box[1], y)
                    union_box[2] = max(union_box[2], x + w_box)
                    union_box[3] = max(union_box[3], y + h_box)

    if union_box is None:
        print("Error: Could not find display in the first 100 frames.")
        return

    x_m, y_m = union_box[0], union_box[1]
    w_m = union_box[2] - x_m
    h_m = union_box[3] - y_m

    # --- Step 2: Process all frames using the stable master box + local refinement ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_printed_map = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        h, w, _ = frame.shape
        roi = frame[h//3:2*h//3, w//3:2*w//3]

        display_area = roi[y_m:y_m+h_m, x_m:x_m+w_m]
        if display_area.size == 0: continue

        gray_display = cv2.cvtColor(display_area, cv2.COLOR_BGR2GRAY)
        _, thresh_display = cv2.threshold(gray_display, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh_display.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue

        center_x_display = thresh_display.shape[1] / 2
        digit1_contours = [c for c in contours if cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2 < center_x_display]
        digit2_contours = [c for c in contours if cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]/2 >= center_x_display]

        d1, d2 = '?', '?'
        if digit1_contours:
            all_points1 = np.concatenate(digit1_contours)
            x1, y1, w1, h1 = cv2.boundingRect(all_points1)
            d1_roi = thresh_display[y1:y1+h1, x1:x1+w1]
            d1 = get_digit_from_roi(d1_roi)

        if digit2_contours:
            all_points2 = np.concatenate(digit2_contours)
            x2, y2, w2, h2 = cv2.boundingRect(all_points2)
            d2_roi = thresh_display[y2:y2+h2, x2:x2+w2]
            d2 = get_digit_from_roi(d2_roi)

        current_val = f"{d1}{d2}"
        if '?' not in current_val:
            # Use a map to track sightings of each value to handle single frames
            # and prevent re-printing the same value in a long sequence.
            if last_printed_map.get(current_val) is None:
                 print(current_val)
                 last_printed_map[current_val] = True

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python decode_7seg.py <path_to_video>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
