import numpy as np
from typing import List, Dict, Tuple, Any

class CowBehaviorAnalyzer:
    """
    Analyzes historical trajectories and bounding boxes to identify specific cow behaviors:
    - Lying behavior (躺卧行为): Detected via low movement speed and squashed bbox aspect ratio.
    - Limping behavior (跛行行为): Detected via abnormal speed variations or vertical/lateral gait fluctuations.
    - Normal behavior: Standard walking or standing.
    """
    def __init__(
        self,
        speed_threshold_lying: float = 2.0,      # Maximum pixels moved per frame to count as static
        aspect_ratio_lying: float = 0.8,        # Height/Width ratio below which cow is likely lying
        limping_std_threshold: float = 1.5       # Std dev of speed variation indicating limping gait
    ):
        self.speed_threshold_lying = speed_threshold_lying
        self.aspect_ratio_lying = aspect_ratio_lying
        self.limping_std_threshold = limping_std_threshold
        # Stores historical speeds to analyze gait patterns: {cattle_id: list of speeds}
        self.speed_history: Dict[int, List[float]] = {}

    def analyze(self, cattle_list: List[Any], trajectory_history: Dict[int, List[Tuple[int, int]]], frame_shape: Tuple[int, int] = (800, 600)) -> List[Any]:
        """
        Analyzes the behavior of each tracked cow with scale-invariant speed normalization
        and close-up filters to prevent false positives.
        
        Args:
            cattle_list: List of tracked cattle objects (with 'bbox' and 'cattle_id' attributes).
            trajectory_history: Dictionary containing historical centroid trajectories for each ID.
            frame_shape: Tuple of (width, height) of the original frames.
            
        Returns:
            The cattle list with updated 'status' attribute.
        """
        frame_w, frame_h = frame_shape
        frame_area = frame_w * frame_h
        
        for cattle in cattle_list:
            cid = cattle.cattle_id
            if cid < 0:
                continue
                
            x1, y1, x2, y2 = cattle.bbox
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            aspect_ratio = h / w
            bbox_area = w * h
            relative_area = bbox_area / frame_area
            
            # Get trajectory history for this cow
            history = trajectory_history.get(cid, [])
            if len(history) < 2:
                cattle.status = "Walking"
                continue
                
            # Calculate instantaneous 2D displacement speed (in pixels)
            dx = history[-1][0] - history[-2][0]
            dy = history[-1][1] - history[-2][1]
            speed = np.sqrt(dx**2 + dy**2)
            
            # Scale-invariant speed normalization: speed relative to cow size (height)
            # Helps to normalize distance variations due to perspective effects.
            normalized_speed = speed / h
            
            # Store normalized speed history
            if cid not in self.speed_history:
                self.speed_history[cid] = []
            self.speed_history[cid].append(normalized_speed)
            if len(self.speed_history[cid]) > 20:
                self.speed_history[cid].pop(0)
                
            # Detect if cow is close-up (fills a major part of the frame)
            is_close_up = (w > 0.4 * frame_w) or (relative_area > 0.20)
            
            # Rule 1: Detect Lying Down (躺卧行为)
            # A lying cow is static (low speed) and has a flat bounding box aspect ratio.
            # Close-up cows taking up the screen should NOT be classified as lying down.
            is_static = speed < self.speed_threshold_lying
            if is_static and aspect_ratio < self.aspect_ratio_lying and not is_close_up:
                cattle.status = "Lying"
                continue
                
            # Rule 2: Detect Limping (跛行行为)
            # Limping cows show high fluctuations in scale-invariant speed (gait asymmetry).
            if len(self.speed_history[cid]) >= 5:
                recent_norms = self.speed_history[cid][-10:]
                norm_variance = np.std(recent_norms)
                mean_norm = np.mean(recent_norms)
                
                # If moving forward, check standard deviation of normalized speeds.
                # Threshold scaled appropriately (0.01 per unit of config threshold)
                if mean_norm > 0.003 and norm_variance > (self.limping_std_threshold * 0.015):
                    cattle.status = "Limping"
                    continue
            
            # Default to walking / standing based on speed
            if speed > self.speed_threshold_lying:
                cattle.status = "Walking"
            else:
                cattle.status = "Standing"
                
        return cattle_list
