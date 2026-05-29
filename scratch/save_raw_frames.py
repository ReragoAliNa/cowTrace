import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import config

def main():
    for name in ["test.mp4", "cow.mp4"]:
        video_path = os.path.join(config.VIDEOS_DIR, name)
        if not os.path.exists(video_path):
            print(f"Not found: {video_path}")
            continue
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            out_path = f"scratch/raw_frame0_{os.path.splitext(name)[0]}.png"
            os.makedirs("scratch", exist_ok=True)
            cv2.imwrite(out_path, frame)
            print(f"Saved raw frame 0 of {name} to {out_path}")
        cap.release()

if __name__ == "__main__":
    main()
