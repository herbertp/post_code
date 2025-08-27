import cv2
import numpy as np
import sys
import time

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # --- Step 1: Analyze first N frames to find stable bounding box ---
    N_FRAMES_FOR_BOX = 10
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

            if w_box > 0 and h_box > 0:
                if union_box is None:
                    union_box = [x, y, x + w_box, y + h_box]
                else:
                    union_box[0] = min(union_box[0], x)
                    union_box[1] = min(union_box[1], y)
                    union_box[2] = max(union_box[2], x + w_box)
                    union_box[3] = max(union_box[3], y + h_box)

    if union_box is None:
        print("Error: Could not find display in the first frames.")
        return

    x_master, y_master = union_box[0], union_box[1]
    w_master = union_box[2] - union_box[0]
    h_master = union_box[3] - union_box[1]

    # --- Step 2: Define fixed sample points with corrected coordinates ---
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

    single_digit_w = w_master / 2
    y_top = h_master * 0.20; y_mid = h_master * 0.5; y_bot = h_master * 0.80

    # Corrected x-coordinates to be near the edges for vertical segments
    x_left_edge = single_digit_w * 0.10
    x_right_edge = single_digit_w * 0.90
    x_mid = single_digit_w * 0.5

    single_digit_segment_centers = [
        (y_top, x_mid),       # a
        (y_top, x_right_edge),# b
        (y_bot, x_right_edge),# c
        (y_bot, x_mid),       # d
        (y_bot, x_left_edge), # e
        (y_top, x_left_edge), # f
        (y_mid, x_mid)        # g
    ]

    sample_points = []
    for y, x in single_digit_segment_centers:
        sample_points.append((int(y), int(x)))
    for y, x in single_digit_segment_centers:
        sample_points.append((int(y), int(x + single_digit_w)))

    # --- Step 3: Process all frames ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_printed = ""
    last_val = ""
    stable_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        roi = frame[h//3:2*h//3, w//3:2*w//3]

        digit_area = roi[y_master:y_master+h_master, x_master:x_master+w_master]
        if digit_area.size == 0: continue
        gray_digits = cv2.cvtColor(digit_area, cv2.COLOR_BGR2GRAY)

        frame_thresh, _ = cv2.threshold(gray_digits, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        segments = []
        for y, x in sample_points:
            if y < gray_digits.shape[0] and x < gray_digits.shape[1]:
                if gray_digits[y, x] > frame_thresh:
                    segments.append(1)
                else:
                    segments.append(0)
            else:
                segments.append(0)

        digit1_segs = tuple(segments[0:7])
        digit2_segs = tuple(segments[7:14])
        d1 = DIGITS_LOOKUP.get(digit1_segs, '?')
        d2 = DIGITS_LOOKUP.get(digit2_segs, '?')

        current_val = f"{d1}{d2}"

        if current_val == last_val:
            stable_count += 1
        else:
            last_val = current_val
            stable_count = 1

        if stable_count >= 2 and '?' not in current_val:
            if current_val != last_printed:
                print(current_val)
                last_printed = current_val

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python decode_7seg.py <path_to_video>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
