import cv2
import numpy as np

def check_image(name, path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        return
        
    print(f"=== Inspecting pre-existing boxes in raw frame: {name} ===")
    # Look for sharp horizontal/vertical lines or typical box colors (like green, blue, red) in the raw frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rect_count = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            # A box is likely a detection box if it's large and has reasonable aspect ratio
            if w > 50 and h > 50 and 0.5 < aspect_ratio < 2.0:
                print(f"  Likely pre-existing rectangular box found at: x={x}, y={y}, w={w}, h={h}")
                rect_count += 1
                
    if rect_count == 0:
        print("  No pre-existing rectangular boxes detected in raw frame.")
    else:
        print(f"  Found {rect_count} potential pre-existing box(es) in the original source video.")

def main():
    check_image("test.mp4", "scratch/raw_frame0_test.png")
    check_image("cow.mp4", "scratch/raw_frame0_cow.png")

if __name__ == "__main__":
    main()
