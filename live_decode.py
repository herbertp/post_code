import cv2
import numpy as np
import sys
import time
import argparse
import os
import re
from collections import deque
from threading import Thread
from queue import Queue, Empty

class VideoStream:
    """A threaded video stream reader to prevent blocking I/O."""
    def __init__(self, src=0, width=None, height=None, fps=None, fourcc=None):
        self.stream = cv2.VideoCapture(src)
        if width: self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fourcc: self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if fps: self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stopped = False
        self.queue = Queue(maxsize=1)
    def start(self):
        Thread(target=self.update, args=()).start(); return self
    def update(self):
        while not self.stopped:
            if self.queue.full(): time.sleep(0.001); continue
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                return
            self.queue.put(frame)
    def read(self):
        try: return self.queue.get(timeout=1)
        except Empty: self.stopped = True; return None
    def stop(self):
        self.stopped = True; time.sleep(0.2)
        if hasattr(self.stream, 'release'): self.stream.release()

def set_camera_properties_v4l2(device_index, settings):
    if not settings: return
    device_path = f"/dev/video{device_index}"
    print(f"Applying custom camera settings to {device_path} via v4l2-ctl...")
    command_parts = ["v4l2-ctl", "-d", device_path]
    for setting in settings:
        try:
            key, value = setting.split('=', 1)
            command_parts.append(f"--set-ctrl={key}={value}")
        except ValueError: print(f"Warning: Invalid format for setting '{setting}'. Use key=value.")
    command = " ".join(command_parts)
    print(f"Executing: {command}")
    os.system(command)
    time.sleep(0.5)

def get_segment_centroids(roi_for_calib):
    gray = cv2.cvtColor(roi_for_calib, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segment_contours = sorted([c for c in contours if cv2.contourArea(c) > 10], key=cv2.contourArea, reverse=True)[:14]
    if len(segment_contours) < 14: return None
    centroids = [ (int(cv2.moments(c)["m10"] / cv2.moments(c)["m00"]), int(cv2.moments(c)["m01"] / cv2.moments(c)["m00"])) for c in segment_contours if cv2.moments(c)["m00"] !=0]
    if len(centroids) < 14: return None
    centroids.sort(key=lambda p: p[0])
    left_centroids, right_centroids = centroids[:7], centroids[7:]
    def sort_digit_segments(digit_centroids):
        y_sorted = sorted(digit_centroids, key=lambda p: p[1])
        if len(y_sorted) < 7: return None
        top_row = sorted(y_sorted[:3], key=lambda p: p[0]);
        if len(top_row) < 3: return None
        seg_f, seg_a, seg_b = top_row[0], top_row[1], top_row[2]
        seg_g = y_sorted[3]
        bot_row = sorted(y_sorted[4:], key=lambda p: p[0])
        if len(bot_row) < 3: return None
        seg_e, seg_d, seg_c = bot_row[0], bot_row[1], bot_row[2]
        return [(y,x) for x,y in [seg_a, seg_b, seg_c, seg_d, seg_e, seg_f, seg_g]]
    ordered_left = sort_digit_segments(left_centroids)
    ordered_right = sort_digit_segments(right_centroids)
    if ordered_left is None or ordered_right is None: return None
    return ordered_left + ordered_right

def auto_calibrate(vs, args, search_limit=150):
    print(f"Attempting auto-calibration by searching for '88' in the first {search_limit} frames...")
    frame_count = 0
    while frame_count < search_limit:
        frame = vs.read()
        if frame is None: return None, None
        frame_count += 1
        h, w, _ = frame.shape
        x_frac, y_frac, w_frac, h_frac = parse_geometry(args.geometry)
        x_px, y_px = int(x_frac * w), int(y_frac * h)
        w_px, h_px = int(w_frac * w), int(h_frac * h)
        roi = frame[y_px:y_px+h_px, x_px:x_px+w_px]
        points = get_segment_centroids(roi)
        if points:
            print(f"Auto-calibration successful on frame #{frame_count}.")
            all_points_np = np.array([[x,y] for y,x in points])
            x_m, y_m, w_m, h_m = cv2.boundingRect(all_points_np)
            roi_box = (x_m, y_m, w_m, h_m)
            return roi_box, points
    print("Auto-calibration failed.")
    return None, None

def parse_geometry(geometry_str):
    match = re.match(r"(\d+(\.\d+)?)x(\d+(\.\d+)?)@\((\d+(\.\d+)?),(\d+(\.\d+)?)\)", geometry_str)
    if match:
        w = float(match.group(1)); h = float(match.group(3))
        cx = float(match.group(5)); cy = float(match.group(7))
        return (cx - w/2), (cy - h/2), w, h
    match = re.match(r"(\d+(\.\d+)?)x(\d+(\.\d+)?)", geometry_str)
    if match:
        w = float(match.group(1)); h = float(match.group(3))
        return (0.5 - w/2), (0.5 - h/2), w, h
    print("Warning: Invalid geometry format. Using default.")
    return 1/3, 1/3, 1/3, 1/3

def load_decode_map(filepath):
    if not filepath: return {}
    try:
        decode_map = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = re.split(r'[:\s]+', line, 1)
                if len(parts) == 2:
                    decode_map[parts[0].upper()] = parts[1].strip()
        print(f"Successfully loaded {len(decode_map)} POST code descriptions.")
        return decode_map
    except FileNotFoundError:
        print(f"Warning: Decode file not found at '{filepath}'")
        return {}

def main(args):
    roi_props = parse_geometry(args.geometry)
    decode_map = load_decode_map(args.decode)
    vs = VideoStream(src=args.source, width=args.cam_width, height=args.cam_height, fps=args.fps, fourcc=args.format).start()
    time.sleep(1.0)

    if isinstance(args.source, int) and args.set_ctrl:
        set_camera_properties_v4l2(args.source, args.set_ctrl)

    state, roi_box, sample_points = 'AWAITING_CALIBRATION', None, None

    if not args.visualize:
        roi_box, sample_points = auto_calibrate(vs, args)
        if roi_box:
            state = 'DECODING'
            if isinstance(args.source, str):
                vs.stop(); vs = VideoStream(src=args.source).start(); time.sleep(1.0)
        else:
            print("Could not auto-calibrate. Exiting."); vs.stop(); return

    DIGITS_LOOKUP = {
        (1,1,1,1,1,1,0):'0', (0,1,1,0,0,0,0):'1', (1,1,0,1,1,0,1):'2', (1,1,1,1,0,0,1):'3',
        (0,1,1,0,0,1,1):'4', (1,0,1,1,0,1,1):'5', (1,0,1,1,1,1,1):'6', (1,1,1,0,0,0,0):'7',
        (1,1,1,1,1,1,1):'8', (1,1,1,1,0,1,1):'9', (1,1,1,0,1,1,1):'A', (0,0,1,1,1,1,1):'B',
        (1,0,0,1,1,1,0):'C', (0,1,1,1,1,0,1):'D', (1,0,0,1,1,1,1):'E', (1,0,0,0,1,1,1):'F'
    }
    frame_idx, last_val, stable_count, last_printed_val = 0, "", 0, ""
    fps_buffer = deque(maxlen=30)
    last_time = time.time()

    print("Starting decoder. In visual mode: 'c' to calibrate, 's' to apply settings, 'q' to quit.")

    while True:
        frame = vs.read()
        if frame is None: break
        frame_idx += 1

        current_time = time.time()
        time_delta = current_time - last_time
        last_time = current_time
        if time_delta > 0: fps_buffer.append(1.0 / time_delta)

        h, w, _ = frame.shape
        x_frac, y_frac, w_frac, h_frac = roi_props
        x_px, y_px = int(x_frac * w), int(y_frac * h)
        w_px, h_px = int(w_frac * w), int(h_frac * h)
        roi = frame[y_px:y_px+h_px, x_px:x_px+w_px]

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        display_frame = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR) if args.visualize else None
        current_val = "??"

        if state == 'DECODING':
            frame_thresh, _ = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            segments = [1 if gray_roi[y,x] > frame_thresh else 0 for y,x in sample_points]
            d1 = DIGITS_LOOKUP.get(tuple(segments[0:7]), '?'); d2 = DIGITS_LOOKUP.get(tuple(segments[7:14]), '?')
            current_val = f"{d1}{d2}"

            if current_val == last_val: stable_count += 1
            else: last_val = current_val; stable_count = 1

            if '?' not in current_val and stable_count >= args.stable and current_val != last_printed_val:
                description = decode_map.get(current_val, "")
                print(f"{frame_idx},{current_val} {description}".strip())
                last_printed_val = current_val

            if args.visualize:
                x_m, y_m, w_m, h_m = roi_box
                cv2.rectangle(display_frame, (x_m, y_m), (x_m + w_m, y_m + h_m), (255, 0, 0), 1)
                for y, x in sample_points: cv2.circle(display_frame, (x, y), 2, (0, 0, 255), -1)

        if args.visualize:
            msg = current_val
            if state == 'AWAITING_CALIBRATION': msg = "Calibrating... ('c' to force)"
            cv2.putText(display_frame, msg, (10, display_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            avg_fps = np.mean(fps_buffer) if fps_buffer else 0
            fps_text = f"FPS: {avg_fps:.2f}"; text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(display_frame, fps_text, (display_frame.shape[1] - text_size[0] - 10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Decoder", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): vs.stop(); break
            elif key == ord('c'):
                print("Attempting manual calibration..."); points = get_segment_centroids(roi)
                if points:
                    all_points_np = np.array([[x,y] for y,x in points]); x_m, y_m, w_m, h_m = cv2.boundingRect(all_points_np)
                    roi_box, sample_points = (x_m, y_m, w_m, h_m), points; state = 'DECODING'; print("Calibration successful.")
                else: print("Manual calibration failed.")
            elif key == ord('s') and isinstance(args.source, int) and args.set_ctrl: set_camera_properties_v4l2(args.source, args.set_ctrl)

        elif args.debug:
            if frame_idx % 150 == 0:
                avg_fps = np.mean(fps_buffer) if fps_buffer else 0
                print(f"[DEBUG] Average FPS: {avg_fps:.2f}")

    vs.stop()
    if args.visualize: cv2.destroyAllWindows()
    print("Decoder stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode 7-segment display from video file or camera.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("source", help="Path to video file or camera index (e.g., 0).")
    parser.add_argument("--visualize", action="store_true", help="Enable live visualization of the decoding process.")
    parser.add_argument("--debug", action="store_true", help="Print periodic FPS to the console in non-visual mode.")
    parser.add_argument("--decode", type=str, default="", help="Path to a file with POST code descriptions (e.g., AA:Description).")
    parser.add_argument("-c", "--set-ctrl", action="append", dest="set_ctrl", help="Set a camera property using v4l2-ctl. Use key=value format. Can be used multiple times.")
    parser.add_argument("--fps", type=int, help="Request a specific FPS from the camera.")
    parser.add_argument("--cam-width", type=int, help="Set camera frame width.")
    parser.add_argument("--cam-height", type=int, help="Set camera frame height.")
    parser.add_argument("--format", type=str, help="Set camera FOURCC format (e.g., MJPG).")
    parser.add_argument("--geometry", type=str, default="0.33x0.33@0.5,0.5", help="ROI geometry as WxH or WxH@(X,Y) in fractions of frame dimensions.")
    parser.add_argument("--stable", type=int, default=2, help="Number of stable frames required before reporting a new value.")
    args = parser.parse_args()
    try: args.source = int(args.source)
    except ValueError: pass
    main(args)
