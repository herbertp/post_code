import cv2
import numpy as np
import sys
import time
import argparse

def calibrate(cap, num_frames_to_search=200):
    """
    Searches the first `num_frames_to_search` of a video stream for a
    frame that looks like '88' (i.e., has 14 distinct segments), then
    calculates the bounding box and sample points from it.
    """
    print(f"Searching for '88' calibration frame in the first {num_frames_to_search} frames...")

    calibration_frame_roi = None
    frame_count = 0

    while frame_count < num_frames_to_search:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        h, w, _ = frame.shape
        roi = frame[h//3:2*h//3, w//3:2*w//3]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3), np.uint8)
        eroded_thresh = cv2.erode(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(eroded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 10
        segment_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if len(segment_contours) >= 14:
            print(f"Found potential '88' frame at frame #{frame_count}.")
            calibration_frame_roi = roi.copy()
            break

    if calibration_frame_roi is None:
        print("Error: Calibration failed. Could not find a suitable '88' frame.")
        return None, None

    # --- Now that we have the frame, get the box and points ---
    gray = cv2.cvtColor(calibration_frame_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    all_points = np.concatenate(contours)
    x_m, y_m, w_m, h_m = cv2.boundingRect(all_points)
    roi_box = (x_m, y_m, w_m, h_m)

    kernel = np.ones((3,3), np.uint8)
    eroded_thresh = cv2.erode(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(eroded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segment_contours = sorted([c for c in contours if cv2.contourArea(c) > 10], key=cv2.contourArea, reverse=True)[:14]

    centroids = [ (int(cv2.moments(c)["m10"] / cv2.moments(c)["m00"]), int(cv2.moments(c)["m01"] / cv2.moments(c)["m00"])) for c in segment_contours if cv2.moments(c)["m00"] !=0]

    centroids.sort(key=lambda p: p[0])
    left_digit_centroids = centroids[:7]
    right_digit_centroids = centroids[7:]

    def sort_digit_segments(digit_centroids):
        y_sorted = sorted(digit_centroids, key=lambda p: p[1])
        top_row = sorted(y_sorted[:3], key=lambda p: p[0])
        if len(top_row) < 3: return None
        seg_f, seg_a, seg_b = top_row[0], top_row[1], top_row[2]
        if len(y_sorted) < 4: return None
        seg_g = y_sorted[3]
        bot_row = sorted(y_sorted[4:], key=lambda p: p[0])
        if len(bot_row) < 3: return None
        seg_e, seg_d, seg_c = bot_row[0], bot_row[1], bot_row[2]
        return [(y,x) for x,y in [seg_a, seg_b, seg_c, seg_d, seg_e, seg_f, seg_g]]

    ordered_left = sort_digit_segments(left_digit_centroids)
    ordered_right = sort_digit_segments(right_digit_centroids)

    if ordered_left is None or ordered_right is None:
        print("Error: Calibration failed. Could not sort segments."); return None, None

    sample_points = ordered_left + ordered_right
    print("Calibration successful.")
    return roi_box, sample_points

def main(args):
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened(): print(f"Error: Could not open source '{args.source}'"); return

    roi_box, sample_points = calibrate(cap)
    if roi_box is None: print("Exiting."); return

    if isinstance(args.source, str): cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    DIGITS_LOOKUP = {
        (1, 1, 1, 1, 1, 1, 0): '0', (0, 1, 1, 0, 0, 0, 0): '1', (1, 1, 0, 1, 1, 0, 1): '2',
        (1, 1, 1, 1, 0, 0, 1): '3', (0, 1, 1, 0, 0, 1, 1): '4', (1, 0, 1, 1, 0, 1, 1): '5',
        (1, 0, 1, 1, 1, 1, 1): '6', (1, 1, 1, 0, 0, 0, 0): '7', (1, 1, 1, 1, 1, 1, 1): '8',
        (1, 1, 1, 1, 0, 1, 1): '9', (1, 1, 1, 0, 1, 1, 1): 'A', (0, 0, 1, 1, 1, 1, 1): 'B',
        (1, 0, 0, 1, 1, 1, 0): 'C', (0, 1, 1, 1, 1, 0, 1): 'D', (1, 0, 0, 1, 1, 1, 1): 'E',
        (1, 0, 0, 0, 1, 1, 1): 'F'
    }
    frame_idx = 0; last_val = ""

    print("\nStarting decoder...")
    while True:
        ret, frame = cap.read()
        if not ret: print("End of video stream."); break
        frame_idx += 1

        h, w, _ = frame.shape
        roi = frame[h//3:2*h//3, w//3:2*w//3]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        frame_thresh, _ = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        segments = [1 if gray_roi[y,x] > frame_thresh else 0 for y,x in sample_points if y < gray_roi.shape[0] and x < gray_roi.shape[1]]
        if len(segments) != 14: continue

        d1 = DIGITS_LOOKUP.get(tuple(segments[0:7]), '?'); d2 = DIGITS_LOOKUP.get(tuple(segments[7:14]), '?')
        current_val = f"{d1}{d2}"

        if '?' not in current_val and current_val != last_val:
            print(f"{frame_idx},{current_val}"); last_val = current_val

        if args.visualize:
            x_m, y_m, w_m, h_m = roi_box
            display_frame = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(display_frame, (x_m, y_m), (x_m + w_m, y_m + h_m), (255, 0, 0), 1)
            for y, x in sample_points: cv2.circle(display_frame, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(display_frame, current_val, (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Decoder", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                print("Re-calibrating..."); new_box, new_points = calibrate(cap);
                if new_box is not None: roi_box, sample_points = new_box, new_points

    cap.release()
    if args.visualize: cv2.destroyAllWindows()
    print("Decoder stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode 7-segment display from video file or camera.")
    parser.add_argument("source", help="Path to video file or camera index (e.g., 0).")
    parser.add_argument("--visualize", action="store_true", help="Enable live visualization of the decoding process.")
    args = parser.parse_args()
    try: args.source = int(args.source)
    except ValueError: pass
    main(args)
