import cv2
import numpy as np
import sys
import time

# Standard 7-segment mapping (a, b, c, d, e, f, g)
#   a
# f   b
#   g
# e   c
#   d
DIGITS_LOOKUP = {
    # a, b, c, d, e, f, g
    (1, 1, 1, 1, 1, 1, 0): '0',
    (0, 1, 1, 0, 0, 0, 0): '1',
    (1, 1, 0, 1, 1, 0, 1): '2',
    (1, 1, 1, 1, 0, 0, 1): '3',
    (0, 1, 1, 0, 0, 1, 1): '4',
    (1, 0, 1, 1, 0, 1, 1): '5',
    (1, 0, 1, 1, 1, 1, 1): '6',
    (1, 1, 1, 0, 0, 0, 0): '7',
    (1, 1, 1, 1, 1, 1, 1): '8',
    (1, 1, 1, 1, 0, 1, 1): '9',
    (1, 1, 1, 0, 1, 1, 1): 'A',
    (0, 0, 1, 1, 1, 1, 1): 'b',
    (1, 0, 0, 1, 1, 1, 0): 'C',
    (0, 1, 1, 1, 1, 0, 1): 'd',
    (1, 0, 0, 1, 1, 1, 1): 'E',
    (1, 0, 0, 0, 1, 1, 1): 'F'
}

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def decode_digit(digit_roi):
    h, w = digit_roi.shape
    if h < 10 or w < 5:
        return '?'

    ph = max(1, h // 6)
    pw = max(1, w // 4)
    h_half, w_half = h // 2, w // 2
    h_qtr = h // 4

    inset = 2
    segment_patches = [
        (ph//2, w_half - pw//2, ph, pw),
        (h_qtr, w - pw - inset, ph, pw),
        (h_half + h_qtr, w - pw - inset, ph, pw),
        (h - ph - ph//2, w_half - pw//2, ph, pw),
        (h_half + h_qtr, inset, ph, pw),
        (h_qtr, inset, ph, pw),
        (h_half - ph//2, w_half - pw//2, ph, pw)
    ]

    segments = []
    for y, x, patch_h, patch_w in segment_patches:
        y, x, patch_h, patch_w = int(y), int(x), int(patch_h), int(patch_w)
        if y + patch_h >= h or x + patch_w >= w or y < 0 or x < 0:
            segments.append(0)
            continue
        patch = digit_roi[y:y+patch_h, x:x+patch_w]
        if cv2.countNonZero(patch) / (patch_h * patch_w) > 0.5:
            segments.append(1)
        else:
            segments.append(0)
    return DIGITS_LOOKUP.get(tuple(segments), '?')

def process_digit_contours(contours, base_image, std_w, std_h):
    if not contours:
        return '?'
    all_points = np.concatenate(contours)
    if cv2.contourArea(all_points) < 50:
        return '?'
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    src_pts = order_points(box)
    dst_pts = np.array([[0, 0], [std_w - 1, 0], [std_w - 1, std_h - 1], [0, std_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(base_image, M, (std_w, std_h))
    return decode_digit(warped)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    STD_W, STD_H = 50, 100
    last_printed = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        roi = frame[h//3:2*h//3, w//3:2*w//3]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        thresh = mask1 + mask2

        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue

        all_cnts = np.concatenate(contours)
        x_all, y_all, w_all, h_all = cv2.boundingRect(all_cnts)
        center_x_all = x_all + w_all / 2

        digit1_contours = []
        digit2_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 20: continue
            x, y, wc, hc = cv2.boundingRect(cnt)
            if (x + wc/2) < center_x_all:
                digit1_contours.append(cnt)
            else:
                digit2_contours.append(cnt)

        d1 = process_digit_contours(digit1_contours, thresh, STD_W, STD_H)
        d2 = process_digit_contours(digit2_contours, thresh, STD_W, STD_H)
        current_val = f"{d1}{d2}"

        if current_val != last_printed:
            if '?' not in current_val:
                print(current_val)
                last_printed = current_val

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python decode_7seg.py <path_to_video>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
