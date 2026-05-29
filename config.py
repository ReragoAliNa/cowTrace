# Configuration parameters for Cow Behavior and Physical Sign Monitoring System
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# LSNet Model Configuration
MODEL_INPUT_SIZE = (640, 640)      # (width, height) for LSNet engine
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")            # Individual Re-ID model weight file
BEHAVIOR_MODEL_PATH = os.path.join(BASE_DIR, "models", "behavior.pt") # Action/Behavior detection model weight file
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")              # Directory containing video files

# Image Preprocessing Configuration
PREPROCESS_CLAHE_CLIP = 2.0
PREPROCESS_CLAHE_GRID = (8, 8)
PREPROCESS_BILATERAL_D = 9
PREPROCESS_BILATERAL_SIGMA_COLOR = 75.0
PREPROCESS_BILATERAL_SIGMA_SPACE = 75.0
PREPROCESS_PAD_COLOR = (114, 114, 114)  # Standard grey background pad for YOLO/LSNet

# Behavior Monitoring Parameters
LIMPING_THRESHOLD = 0.5            # Threshold for limping classification
LYING_THRESHOLD = 0.6              # Threshold for lying classification
TRACKING_MAX_AGE = 30              # Max frames to keep a missing track active
TRACKING_MIN_HITS = 3              # Min frames to confirm a track

# Tracker Engine Configuration
# Choose tracking engine: "interactive" (startup menu), "rust", "python", or "auto"
TRACKER_ENGINE = "interactive"
