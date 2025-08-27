import cv2
import sys
import os

def extract_specific_frame(video_path, frame_number, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # OpenCV's frame seeking can be inaccurate. A reliable way is to loop.
    # Frame numbers are 1-based for the user, but 0-based for the loop.
    target_frame_idx = frame_number - 1
    current_frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read up to frame #{frame_number}.")
            break

        if current_frame_idx == target_frame_idx:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            cv2.imwrite(output_path, frame)
            print(f"Successfully extracted frame #{frame_number} to {output_path}")
            break

        current_frame_idx += 1

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_frame.py <path_to_video> <frame_number> <output_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    frame_num = int(sys.argv[2])
    output_path = sys.argv[3]
    extract_specific_frame(video_path, frame_num, output_path)
