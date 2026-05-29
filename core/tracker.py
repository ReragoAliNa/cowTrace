import numpy as np
from typing import List, Any
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# Optional Rust PyO3 tracker support
RUST_AVAILABLE = False
try:
    from cow_trace_rs import RustTracker, BBox
    RUST_AVAILABLE = True
except ImportError:
    pass

# Optional / default fallback implementation of SingleCattleData
# in case the user does not define it elsewhere.
class SingleCattleData:
    def __init__(self, bbox: List[float], cattle_id: int = -1, **kwargs):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.cattle_id = cattle_id
        for k, v in kwargs.items():
            setattr(self, k, v)


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    
    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)
    
    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bounding boxes.
    Uses Kalman Filter with a constant velocity model.
    """
    count = 0

    def __init__(self, bbox: np.ndarray):
        # State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        # Coordinates and their velocities
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix F
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix H
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Covariance setup
        self.kf.R *= 10.0  # Measurement noise covariance
        self.kf.P *= 10.0  # State covariance (initial uncertainty)
        self.kf.Q *= 0.01  # Process noise covariance
        
        # Set initial state with detected bbox
        self.kf.x[:4] = np.array(bbox, dtype=np.float32).reshape((4, 1))
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.ndarray):
        """Updates the state vector with a new measurement."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(np.array(bbox, dtype=np.float32).reshape((4, 1)))

    def predict(self) -> np.ndarray:
        """Predicts the state vector forward and returns the predicted bounding box."""
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Get bounding box coords from predicted state vector
        pred_box = self.kf.x[:4].reshape((4,))
        return pred_box


class CentroidTracker:
    """
    Minimally matches new detections to existing tracks using IoU matching
    and updates coordinates dynamically via Kalman Filtering.
    """
    def __init__(self, max_age: int = 30, min_iou: float = 0.3):
        self.max_age = max_age
        self.min_iou = min_iou
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, cattle_list: List[Any]) -> List[Any]:
        """
        Updates trackers with new detections and updates cattle_id in-place.
        
        Args:
            cattle_list: List of objects (e.g. SingleCattleData) containing a 'bbox' attribute.
            
        Returns:
            The modified cattle_list with updated 'cattle_id' fields.
        """
        if RUST_AVAILABLE:
            if not hasattr(self, 'rust_tracker'):
                self.rust_tracker = RustTracker(self.max_age, self.min_iou)
            
            rust_dets = []
            for c in cattle_list:
                x1, y1, x2, y2 = c.bbox
                rust_dets.append(BBox(float(x1), float(y1), float(x2), float(y2)))
                
            try:
                assigned_ids = self.rust_tracker.update(rust_dets)
                for idx, cid in enumerate(assigned_ids):
                    cattle_list[idx].cattle_id = int(cid)
                return cattle_list
            except Exception as e:
                print(f"[WARNING] Rust tracker failed: {e}. Falling back to Python tracker.")
                
        # If no trackers exist, initialize all detections as new tracks
        if len(self.trackers) == 0:
            for cattle in cattle_list:
                tracker = KalmanBoxTracker(cattle.bbox)
                self.trackers.append(tracker)
                cattle.cattle_id = tracker.id
            return cattle_list

        # Get predicted positions of current trackers
        predicted_boxes = []
        for t in self.trackers:
            pred = t.predict()
            # Handle potential negative dimensions by clipping or checking bounds
            predicted_boxes.append(pred)
            
        predicted_boxes_arr = np.array(predicted_boxes)
        detection_boxes_arr = np.array([c.bbox for c in cattle_list])
        
        # Create cost matrix (1 - IoU)
        num_tracks = len(predicted_boxes_arr)
        num_detections = len(detection_boxes_arr)
        cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)
        
        for t in range(num_tracks):
            for d in range(num_detections):
                iou = box_iou(predicted_boxes_arr[t], detection_boxes_arr[d])
                cost_matrix[t, d] = 1.0 - iou
                
        # Hungarian algorithm matching
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_detections = set()
        
        # Process matches
        for t_idx, d_idx in zip(track_indices, det_indices):
            iou = 1.0 - cost_matrix[t_idx, d_idx]
            
            # Match is valid if IoU is above threshold
            if iou >= self.min_iou:
                self.trackers[t_idx].update(cattle_list[d_idx].bbox)
                cattle_list[d_idx].cattle_id = self.trackers[t_idx].id
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)
                
        # Handle unmatched detections (spawn new tracker)
        for d_idx in range(num_detections):
            if d_idx not in matched_detections:
                tracker = KalmanBoxTracker(cattle_list[d_idx].bbox)
                self.trackers.append(tracker)
                cattle_list[d_idx].cattle_id = tracker.id
                
        # Clean up old, dead trackers
        remaining_trackers = []
        for t_idx, tracker in enumerate(self.trackers):
            # If not updated this frame, count as lost
            # Keep trackers that were matched or haven't exceeded max age
            if t_idx in matched_tracks or tracker.time_since_update < self.max_age:
                remaining_trackers.append(tracker)
                
        self.trackers = remaining_trackers
        return cattle_list
