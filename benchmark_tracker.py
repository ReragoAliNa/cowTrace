import time
import numpy as np
import random
import sys
import os

# Ensure the root workspace path is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Temporarily disable RUST_AVAILABLE in tracker to force Python path for python profiling
import core.tracker as tracker
from core.tracker import SingleCattleData

# Attempt to load Rust Tracker
try:
    from cow_trace_rs import RustTracker, BBox
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

def generate_random_bboxes(num_boxes):
    boxes = []
    for _ in range(num_boxes):
        x1 = random.uniform(10.0, 1000.0)
        y1 = random.uniform(10.0, 1000.0)
        w = random.uniform(50.0, 200.0)
        h = random.uniform(50.0, 200.0)
        boxes.append([x1, y1, x1 + w, y1 + h])
    return boxes

def main():
    print("=== Tracker Performance Benchmark ===", flush=True)
    if not RUST_AVAILABLE:
        print("[ERROR] Rust tracker not installed! Please compile and install it first.", flush=True)
        return
        
    num_objects_list = [5, 15, 50, 100]
    iterations = 100
    
    for num_objects in num_objects_list:
        print(f"\nProfiling with {num_objects} target cows (iterations: {iterations})...", flush=True)
        
        # 1. Profile Python Tracker (force disable Rust flag temporarily)
        tracker.RUST_AVAILABLE = False
        py_tracker = tracker.CentroidTracker(max_age=30, min_iou=0.3)
        # Warmup
        initial_bboxes = generate_random_bboxes(num_objects)
        py_tracker.update([SingleCattleData(b) for b in initial_bboxes])
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            bboxes = generate_random_bboxes(num_objects)
            py_tracker.update([SingleCattleData(b) for b in bboxes])
        py_elapsed = time.perf_counter() - start_time
        
        # 2. Profile Rust Tracker
        rust_tracker = RustTracker(30, 0.3)
        # Warmup
        initial_rust_bboxes = [BBox(b[0], b[1], b[2], b[3]) for b in initial_bboxes]
        rust_tracker.update(initial_rust_bboxes)
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            bboxes = generate_random_bboxes(num_objects)
            rust_bboxes = [BBox(b[0], b[1], b[2], b[3]) for b in bboxes]
            rust_tracker.update(rust_bboxes)
        rust_elapsed = time.perf_counter() - start_time
        
        py_fps = iterations / py_elapsed
        rust_fps = iterations / rust_elapsed
        speedup = py_elapsed / rust_elapsed
        
        # Restore tracker flag
        tracker.RUST_AVAILABLE = True
        
        print(f"  Python Tracker: {py_elapsed*1000/iterations:.3f} ms/update ({py_fps:.1f} Hz)", flush=True)
        print(f"  Rust Tracker:   {rust_elapsed*1000/iterations:.3f} ms/update ({rust_fps:.1f} Hz)", flush=True)
        print(f"  Speedup Factor: {speedup:.2f}x", flush=True)

if __name__ == "__main__":
    main()
