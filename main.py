import os
import sys
import cv2
import numpy as np
import csv
from typing import List

# Ensure trace folder is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.preprocessor import CowImagePreprocessor
from core.tracker import CentroidTracker, SingleCattleData
from core.behavior_analyzer import CowBehaviorAnalyzer
from models.lsnet_engine import LSNetEngine
from utils.visualizer import CattleVisualizer
import config

def simulate_model_outputs(frame_idx: int) -> List[SingleCattleData]:
    """
    Simulates the outputs of the LSNet model (bounding boxes) for each frame.
    Simulates three cows:
    - Cow 1: Normal walking cow, moves left-to-right across the center.
    - Cow 2: A lying cow, static with low aspect ratio (width > height).
    - Cow 3: A limping cow, moves left-to-right but has speed fluctuations and abnormal bbox aspect ratio.
    
    Coordinates are simulated in the model's input size (640x640).
    """
    detections = []
    
    # 1. Cow 1: Normal walking cow (ID will be assigned by tracker)
    # Starts at x=50, y=200 on frame 0, moves by 12 px per frame
    c1_x = 50 + frame_idx * 12
    c1_y = 200 + int(3 * np.sin(frame_idx * 0.5))  # slight vertical bobbing
    if c1_x < 580:  # Active while on screen
        # Bbox in format [x1, y1, x2, y2]
        bbox_c1 = [c1_x, c1_y, c1_x + 90, c1_y + 110]
        detections.append(SingleCattleData(bbox=bbox_c1, label="cow"))

    # 2. Cow 2: Lying cow
    # Mostly static, width=120, height=70 (low aspect ratio)
    c2_x = 220
    c2_y = 350
    # Simulate slight detection jitter
    jitter_x = np.random.randint(-1, 2)
    jitter_y = np.random.randint(-1, 2)
    bbox_c2 = [c2_x + jitter_x, c2_y + jitter_y, c2_x + 120 + jitter_x, c2_y + 70 + jitter_y]
    detections.append(SingleCattleData(bbox=bbox_c2, label="cow"))

    # 3. Cow 3: Limping cow
    # Enters at frame 5, moves slower (6 px per frame) but with higher speed variance (limp)
    if frame_idx >= 5:
        step = (frame_idx - 5)
        # Limping gait speed fluctuation simulation (fast step, slow drag step)
        speed_modifier = 1.2 if step % 2 == 0 else 0.4
        c3_x = 80 + int(step * 8 * speed_modifier)
        # Bounding box bobs up and down significantly during limping
        c3_y = 120 + (10 if step % 2 == 0 else 0)
        bbox_c3 = [c3_x, c3_y, c3_x + 80, c3_y + 105]
        detections.append(SingleCattleData(bbox=bbox_c3, label="cow"))
        
    return detections

def generate_farm_background(w: int = 800, h: int = 600) -> np.ndarray:
    """Generates a synthetic green farm pasture background frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Paste green grass color
    frame[:, :] = (34, 139, 34)  # Forest Green
    # Add some fence posts or lines
    cv2.line(frame, (0, 150), (w, 150), (100, 100, 100), 4)
    cv2.line(frame, (0, 160), (w, 160), (120, 120, 120), 2)
    # Add soil patch
    cv2.ellipse(frame, (w//2, h-50), (w//3, h//6), 0, 0, 360, (20, 70, 100), -1)
    
    # Add low contrast lighting (low light)
    frame = (frame * 0.15).astype(np.uint8)
    return frame

def main():
    print("=== Cow Behavior Monitoring System Pipeline ===")
    
    # 1. Initialize core system components
    preprocessor = CowImagePreprocessor(
        target_size=config.MODEL_INPUT_SIZE,
        default_clip_limit=config.PREPROCESS_CLAHE_CLIP,
        tile_grid_size=config.PREPROCESS_CLAHE_GRID,
        bilateral_d=config.PREPROCESS_BILATERAL_D,
        bilateral_sigma_color=config.PREPROCESS_BILATERAL_SIGMA_COLOR,
        bilateral_sigma_space=config.PREPROCESS_BILATERAL_SIGMA_SPACE,
        pad_color=config.PREPROCESS_PAD_COLOR
    )
    
    tracker = CentroidTracker(
        max_age=config.TRACKING_MAX_AGE,
        min_iou=0.3
    )
    
    analyzer = CowBehaviorAnalyzer(
        speed_threshold_lying=1.5,
        aspect_ratio_lying=0.8,
        limping_std_threshold=1.5
    )
    
    visualizer = CattleVisualizer(
        trajectory_max_len=config.TRACKING_MAX_AGE
    )
    
    # Initialize LSNetEngine (supports .pt, .onnx, and .engine models)
    model_path = config.MODEL_PATH
    engine = LSNetEngine(model_path=model_path)
    is_model_present = os.path.exists(model_path)
    if is_model_present:
        print(f"Detected model file at {model_path}. Using real model inference.")
    else:
        print(f"Model file not found at {model_path}. Running pipeline in simulation mode.")
    
    # Check videos directory and let user select a video
    videos_dir = config.VIDEOS_DIR
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir, exist_ok=True)
        
    valid_extensions = (".mp4", ".avi", ".mkv", ".mov")
    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(valid_extensions)]
    
    use_video = len(video_files) > 0
    video_path = ""
    video_name = "synthetic"
    
    if use_video:
        print("\n=== Available Video Files ===")
        for idx, name in enumerate(video_files):
            print(f"  [{idx + 1}] {name}")
            
        choice = ""
        # Check if stdin is a TTY to allow interactive input
        if sys.stdin.isatty():
            try:
                choice = input(f"Select a video [1-{len(video_files)}] (default: 1): ").strip()
            except Exception:
                pass
                
        if not choice:
            selected_file = video_files[0]
            print(f"No selection made or non-interactive mode. Defaulting to: {selected_file}")
        else:
            try:
                sel_idx = int(choice) - 1
                if 0 <= sel_idx < len(video_files):
                    selected_file = video_files[sel_idx]
                else:
                    selected_file = video_files[0]
                    print(f"Invalid choice, defaulting to: {selected_file}")
            except ValueError:
                selected_file = video_files[0]
                print(f"Invalid input, defaulting to: {selected_file}")
                
        video_path = os.path.join(videos_dir, selected_file)
        video_name = os.path.splitext(selected_file)[0]
        
    if use_video:
        cap = cv2.VideoCapture(video_path)
        # Limit to 100 frames to keep the execution fast and resource-friendly
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = min(100, total_video_frames)
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Loaded video: {video_path} ({w_orig}x{h_orig}, running {num_frames}/{total_video_frames} frames)")
    else:
        num_frames = 30
        w_orig, h_orig = 800, 600
        print("Using synthetic farm background.")
        
    # Directory to save output files (specific subfolder for selected video)
    output_dir = os.path.join(config.OUTPUT_DIR, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Output Video Writer
    video_out_path = os.path.join(output_dir, "output_tracked.mp4")
    fps = 25.0
    if use_video:
        # Get frame rate of the original video (default to 25.0 if not read)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w_orig, h_orig))
    
    # Setup Output CSV Log File
    csv_out_path = os.path.join(output_dir, "behavior_log.csv")
    csv_file = open(csv_out_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame_Index", "Cattle_ID", "Bbox_x1", "Bbox_y1", "Bbox_x2", "Bbox_y2", "Status"])
    
    print(f"Running pipeline for {num_frames} frames...")
    
    for idx in range(num_frames):
        # A. Acquire frame
        if use_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = generate_farm_background(w_orig, h_orig)
        
        # B. Image Preprocessing (Adaptive CLAHE + Bilateral Denoise + Pad)
        processed_frame, meta = preprocessor.process(frame)
        
        # C. Model Inference & D. Coordinate Restoration
        original_space_detections = []
        inference_succeeded = False
        
        if is_model_present:
            try:
                # Real inference on processed (640x640) frame
                bboxes, masks, kpts = engine.infer(processed_frame)
                
                # Filter results and restore coordinates to original video space
                for box in bboxes:
                    x1, y1, x2, y2, score, cid = box
                    # Accept confidence >= 0.25 (typically standard threshold)
                    # Support COCO cow class (19) or custom cow class (typically 0)
                    if score >= 0.25:
                        restored_bbox = preprocessor.restore_coords([[x1, y1, x2, y2]], meta)[0]
                        original_space_detections.append(
                            SingleCattleData(bbox=list(restored_bbox), label="cow")
                        )
                inference_succeeded = True
            except Exception as e:
                print(f"[ERROR] Inference failed on frame {idx}: {e}. Falling back to simulation.")
                inference_succeeded = False
                
        if not is_model_present or not inference_succeeded:
            # Fallback: Simulated outputs from LSNet in model space coordinates
            sim_detections_model_space = simulate_model_outputs(idx)
            for det in sim_detections_model_space:
                restored_bbox = preprocessor.restore_coords([det.bbox], meta)[0]
                det.bbox = list(restored_bbox)
                original_space_detections.append(det)
            
        # E. Object Tracking (Associate detections across frames and assign stable cattle_id)
        tracked_cows = tracker.update(original_space_detections)
        
        # F. Behavior Analysis (Classify standing, walking, lying, limping)
        analyzed_cows = analyzer.analyze(tracked_cows, visualizer.histories, (w_orig, h_orig))
        
        # G. Render and Visualize
        annotated_frame = visualizer.draw(frame, analyzed_cows)
        
        # H. Save annotated frame to video writer
        video_writer.write(annotated_frame)
        
        # I. Log data to CSV file
        for cow in analyzed_cows:
            x1, y1, x2, y2 = cow.bbox
            csv_writer.writerow([idx, cow.cattle_id, int(x1), int(y1), int(x2), int(y2), cow.status])
        
        # Save every 5th frame and the final frame for visualization
        if idx % 5 == 0 or idx == num_frames - 1:
            frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
            cv2.imwrite(frame_path, annotated_frame)
            print(f"Frame {idx:02d}: Active tracks: {len(tracked_cows)}")
            for cow in analyzed_cows:
                print(f"  - Cow #{cow.cattle_id}: Bbox={cow.bbox}, Status={cow.status}")
                
    if use_video:
        cap.release()
    video_writer.release()
    csv_file.close()
        
    print(f"\nPipeline successfully completed!")
    print(f"Annotated tracking video saved to: {video_out_path}")
    print(f"Behavioral CSV data log saved to: {csv_out_path}")

if __name__ == "__main__":
    main()
