import cv2
import numpy as np
import sys

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # --- Go to frame #73 ---
    FRAME_TO_ANALYZE = 73
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_ANALYZE - 1)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame #{FRAME_TO_ANALYZE}")
        return

    # --- Process the frame to get a clean binary image ---
    h, w, _ = frame.shape
    roi = frame[h//3:2*h//3, w//3:2*w//3]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Erode the image to separate segments that might be touching
    kernel = np.ones((3,3), np.uint8)
    eroded_thresh = cv2.erode(thresh, kernel, iterations=2)

    # --- Find the 14 segment contours ---
    contours, _ = cv2.findContours(eroded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 10
    segment_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Save the debug image to see what's wrong
    cv2.imwrite("calibrate_debug.png", eroded_thresh)
    print("Saved calibrate_debug.png")

    if len(segment_contours) < 14:
        print(f"Error: Expected at least 14 segments, but found {len(segment_contours)}. Please check frame/threshold.")
        return

    segment_contours.sort(key=cv2.contourArea, reverse=True)
    segment_contours = segment_contours[:14]

    # --- Calculate and sort centroids ---
    centroids = []
    for c in segment_contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    centroids.sort(key=lambda p: p[0])
    left_digit_centroids = centroids[:7]
    right_digit_centroids = centroids[7:]

    def sort_digit_segments(digit_centroids):
        y_sorted = sorted(digit_centroids, key=lambda p: p[1])
        top_row = sorted(y_sorted[:3], key=lambda p: p[0])
        seg_f, seg_a, seg_b = top_row[0], top_row[1], top_row[2]
        seg_g = y_sorted[3]
        bot_row = sorted(y_sorted[4:], key=lambda p: p[0])
        seg_e, seg_d, seg_c = bot_row[0], bot_row[1], bot_row[2]
        return [seg_a, seg_b, seg_c, seg_d, seg_e, seg_f, seg_g]

    ordered_left = sort_digit_segments(left_digit_centroids)
    ordered_right = sort_digit_segments(right_digit_centroids)

    print("COPY THE FOLLOWING COORDINATES INTO decode_7seg.py:\n")
    print("sample_points = [")
    for x, y in ordered_left:
        print(f"    ({y}, {x}),")
    for x, y in ordered_right:
        print(f"    ({y}, {x}),")
    print("]")

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calibrate.py <path_to_video>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
