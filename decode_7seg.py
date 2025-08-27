import cv2
import numpy as np
import sys
import time

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # These points were calibrated from frame 100 of the video, which shows '88'.
    # The coordinates are (y, x) relative to the center 1/3 ROI of the frame.
    sample_points = [
        (13, 191), (61, 233), (161, 213), (205, 152), (158, 111), (59, 132), (107, 173),
        (18, 379), (71, 418), (164, 402), (210, 346), (161, 301), (63, 320), (113, 361)
    ]

    DIGITS_LOOKUP = {
        (1, 1, 1, 1, 1, 1, 0): '0', (0, 1, 1, 0, 0, 0, 0): '1',
        (1, 1, 0, 1, 1, 0, 1): '2', (1, 1, 1, 1, 0, 0, 1): '3',
        (0, 1, 1, 0, 0, 1, 1): '4', (1, 0, 1, 1, 0, 1, 1): '5',
        (1, 0, 1, 1, 1, 1, 1): '6', (1, 1, 1, 0, 0, 0, 0): '7',
        (1, 1, 1, 1, 1, 1, 1): '8', (1, 1, 1, 1, 0, 1, 1): '9',
        (1, 1, 1, 0, 1, 1, 1): 'A', (0, 0, 1, 1, 1, 1, 1): 'B',
        (1, 0, 0, 1, 1, 1, 0): 'C', (0, 1, 1, 1, 1, 0, 1): 'D',
        (1, 0, 0, 1, 1, 1, 1): 'E', (1, 0, 0, 0, 1, 1, 1): 'F'
    }

    last_printed_map = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        h, w, _ = frame.shape
        roi = frame[h//3:2*h//3, w//3:2*w//3]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        frame_thresh, _ = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        segments = []
        for y, x in sample_points:
            if y < gray_roi.shape[0] and x < gray_roi.shape[1]:
                if gray_roi[y, x] > frame_thresh:
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
        if '?' not in current_val:
            if last_printed_map.get(current_val) is None:
                # Print in the format requested by the user in the interactive script
                print(f"{frame_idx},{current_val}")
                last_printed_map[current_val] = True

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python decode_7seg.py <path_to_video>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
