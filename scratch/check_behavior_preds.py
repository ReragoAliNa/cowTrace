import cv2
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from models.lsnet_engine import LSNetEngine
from core.preprocessor import CowImagePreprocessor
import config

def main():
    video_path = os.path.join(config.VIDEOS_DIR, "cow.mp4")
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
        
    engine = LSNetEngine(model_path=config.BEHAVIOR_MODEL_PATH)
    preprocessor = CowImagePreprocessor(target_size=(640, 640))
    
    cap = cv2.VideoCapture(video_path)
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, meta = preprocessor.process(frame)
        bboxes, _, _ = engine.infer(processed_frame)
        print(f"Frame {i}: detected {len(bboxes)} behavior boxes")
        for box in bboxes:
            x1, y1, x2, y2, score, class_id = box
            print(f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Score: {score:.3f}, Class: {class_id} ({engine.model.names[int(class_id)]})")
            
    cap.release()

if __name__ == "__main__":
    main()
