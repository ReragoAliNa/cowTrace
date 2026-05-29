import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

class CattleVisualizer:
    """
    Visualizer for Cow Behavior and Physical Sign Monitoring.
    Draws bounding boxes, tracking IDs, historical movement trajectories,
    and behavior annotation text on frames.
    """
    def __init__(self, trajectory_max_len: int = 50):
        self.trajectory_max_len = trajectory_max_len
        # Key: cattle_id, Value: list of centroid (x, y) coordinates
        self.histories: Dict[int, List[Tuple[int, int]]] = {}
        # Key: cattle_id, Value: BGR color tuple
        self.colors: Dict[int, Tuple[int, int, int]] = {}

    def _get_color(self, cattle_id: int) -> Tuple[int, int, int]:
        """Generates or retrieves a unique bright color for a given cattle ID."""
        if cattle_id not in self.colors:
            # Avoid overly dark colors for better visibility
            color = tuple(int(x) for x in np.random.randint(60, 255, 3))
            # Convert tuple to BGR format
            self.colors[cattle_id] = color
        return self.colors[cattle_id]

    def draw(self, frame: np.ndarray, cattle_list: List[Any]) -> np.ndarray:
        """
        Draws bounding boxes, IDs, trajectory paths, and behavior statuses on the frame.
        
        Args:
            frame: BGR image.
            cattle_list: List of tracked cattle objects containing 'bbox', 'cattle_id',
                         and optionally 'status' / 'behavior'.
                         
        Returns:
            The annotated frame.
        """
        out_frame = frame.copy()
        
        for cattle in cattle_list:
            cid = cattle.cattle_id
            if cid < 0:
                continue
                
            bbox = [int(x) for x in cattle.bbox]  # [x1, y1, x2, y2]
            color = self._get_color(cid)
            
            # 1. Draw Bounding Box
            cv2.rectangle(out_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Calculate centroid for trajectory tracing
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            
            # 2. Update and Draw Trajectory History
            if cid not in self.histories:
                self.histories[cid] = []
            self.histories[cid].append((cx, cy))
            
            if len(self.histories[cid]) > self.trajectory_max_len:
                self.histories[cid].pop(0)
                
            # Draw connecting lines for history centroids
            for i in range(1, len(self.histories[cid])):
                pt1 = self.histories[cid][i - 1]
                pt2 = self.histories[cid][i]
                # Draw lines with fading thickness/color optionally, or standard line
                cv2.line(out_frame, pt1, pt2, color, 2)
                
            # Draw small dot on current centroid
            cv2.circle(out_frame, (cx, cy), 4, color, -1)
            
            # 3. Label text (Cattle ID + status/behavior)
            status = getattr(cattle, "status", "Normal")
            # If there's specific behaviors like lying or limping, highlight them
            label = f"Cow #{cid} ({status})"
            
            # Highlight warning behaviors (e.g. Limping, Lying down)
            text_color = (0, 0, 255) if status in ["Limping", "Lying"] else (0, 255, 0)
            
            # Draw label background box for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Text background rectangle
            bg_pt1 = (bbox[0], bbox[1] - text_h - 10)
            bg_pt2 = (bbox[0] + text_w + 10, bbox[1])
            cv2.rectangle(out_frame, bg_pt1, bg_pt2, (30, 30, 30), -1)
            
            # Put text
            cv2.putText(
                out_frame,
                label,
                (bbox[0] + 5, bbox[1] - 5),
                font,
                font_scale,
                text_color,
                thickness,
                lineType=cv2.LINE_AA
            )
            
        return out_frame
