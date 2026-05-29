import cv2
import numpy as np

def main():
    raw_img = cv2.imread("scratch/raw_frame0_test.png")
    out_img = cv2.imread("outputs/test/frame_000.png")
    
    if raw_img is None or out_img is None:
        print("Images not found!")
        return
        
    print(f"Raw image shape: {raw_img.shape}")
    print(f"Output image shape: {out_img.shape}")
    
    # We can count the number of colored pixels or check if there is a difference.
    # Let's save a diff image
    if raw_img.shape == out_img.shape:
        diff = cv2.absdiff(raw_img, out_img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} different regions between raw and output frame.")
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"  Region {idx}: x={x}, y={y}, w={w}, h={h}")
    else:
        print("Shapes are different, resizing raw to match output...")
        raw_resized = cv2.resize(raw_img, (out_img.shape[1], out_img.shape[0]))
        diff = cv2.absdiff(raw_resized, out_img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} different regions (resized comparison).")
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"  Region {idx}: x={x}, y={y}, w={w}, h={h}")

if __name__ == "__main__":
    main()
