import cv2
import numpy as np
from typing import Tuple, Dict, Any, Union, List

class CowImagePreprocessor:
    """
    Core preprocessor for Cow Behavior and Physical Sign Monitoring System.
    Handles image preprocessing and feature enhancement to prepare images for LSNet.
    
    Key Features:
    1. Adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization) for dynamic brightening.
    2. Bilateral filtering for dust and noise removal while preserving sharp edges.
    3. Proportional scaling with padding to fit the target model input resolution.
    4. Coordinate mapping to restore predictions from model space back to original image space.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        default_clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        pad_color: Tuple[int, int, int] = (114, 114, 114)  # Standard grey padding
    ):
        self.target_size = target_size  # (width, height)
        self.default_clip_limit = default_clip_limit
        self.tile_grid_size = tile_grid_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.pad_color = pad_color

    def adaptive_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) dynamically.
        Estimates the luminance of the image and adjusts clip limits to boost contrast in low-light environments.
        
        Args:
            image: BGR image.
            
        Returns:
            Enhanced BGR image.
        """
        # Convert BGR to LAB color space to extract lightness channel (L)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Calculate mean lightness
        mean_lightness = np.mean(l_channel)
        
        # Adaptive clip limit calculation:
        # Darker images receive higher clip limit to enhance local details more aggressively.
        # Range: mean_lightness near 0 -> clip_limit up to 4.0; mean_lightness near 255 -> clip_limit down to 1.0.
        if mean_lightness < 50:
            # Low light environment
            clip_limit = self.default_clip_limit * 1.5
        elif mean_lightness > 180:
            # Already bright environment
            clip_limit = max(1.0, self.default_clip_limit * 0.75)
        else:
            # Standard lighting environment
            clip_limit = self.default_clip_limit
            
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.tile_grid_size)
        enhanced_l = clahe.apply(l_channel)
        
        # Merge back channels and convert to BGR
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def remove_dust_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Uses Bilateral Filter to smooth out sensor noise and dust artifacts.
        Bilateral filtering is highly suitable here as it preserves edges (critical for pose/instance boundaries)
        while smoothing uniform regions.
        
        Args:
            image: BGR image.
            
        Returns:
            Denoised BGR image.
        """
        return cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )

    def resize_and_pad(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resizes the image proportionally to fit within target_size and pads the rest.
        Maintains the aspect ratio to prevent distorting cow proportions.
        
        Args:
            image: BGR image.
            
        Returns:
            Tuple of:
                - Preprocessed and padded image (shape matching target_size)
                - Dictionary containing scaling factors and paddings for coordinate restoration
        """
        h_orig, w_orig = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Determine scaling factor
        scale = min(target_w / w_orig, target_h / h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        
        # Resize image proportionally
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create target canvas and fill with padding color
        padded = np.full((target_h, target_w, 3), self.pad_color, dtype=np.uint8)
        
        # Calculate padding offsets (centering the image)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # Place resized image on canvas
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        meta = {
            "original_size": (w_orig, h_orig),
            "scale": scale,
            "pad_w": pad_w,
            "pad_h": pad_h,
            "new_size": (new_w, new_h)
        }
        return padded, meta

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs the full preprocessing pipeline:
        1. Adaptive CLAHE brightening
        2. Bilateral filtering for dust removal
        3. Proportional scaling and padding
        
        Args:
            image: Input BGR image (numpy array).
            
        Returns:
            Tuple of:
                - Preprocessed and padded image for LSNet model.
                - Meta dict containing scale and pad parameters for mapping predictions back.
        """
        # Step 1: Contrast enhancement & Brightening
        brightened = self.adaptive_clahe(image)
        
        # Step 2: Edge-preserving noise removal
        denoised = self.remove_dust_noise(brightened)
        
        # Step 3: Proportional resizing & padding
        preprocessed, meta = self.resize_and_pad(denoised)
        
        return preprocessed, meta

    @staticmethod
    def restore_coords(
        coords: Union[np.ndarray, List[Union[List[float], Tuple[float, float]]]],
        meta: Dict[str, Any]
    ) -> np.ndarray:
        """
        Restores coordinates (e.g., bounding boxes, keypoints, polygon points)
        from model's output space back to original image coordinates.
        
        Args:
            coords: Array/List of coordinates in model input shape. Can be of shape (N, 2) or flat.
                    E.g., [[x1, y1], [x2, y2], ...] or [x, y].
            meta: Metadata dictionary returned by `process()` or `resize_and_pad()`.
            
        Returns:
            Numpy array of coordinates mapped to original image dimensions.
        """
        coords_arr = np.array(coords, dtype=np.float32)
        scale = meta["scale"]
        pad_w = meta["pad_w"]
        pad_h = meta["pad_h"]
        w_orig, h_orig = meta["original_size"]
        
        # Check coordinate format to correctly apply translation and scaling
        reshaped = False
        orig_shape = coords_arr.shape
        
        if coords_arr.ndim == 1:
            # Single coordinate point [x, y] or flat format
            if len(coords_arr) % 2 == 0:
                coords_arr = coords_arr.reshape(-1, 2)
                reshaped = True
            else:
                raise ValueError("Coordinate array of 1D must have an even number of elements.")
        elif coords_arr.ndim > 2:
            # Flatten to N x 2 for easier calculation, then reshape back
            coords_arr = coords_arr.reshape(-1, 2)
            reshaped = True
            
        # Map: orig_coord = (model_coord - pad) / scale
        restored = np.zeros_like(coords_arr)
        # Even columns are X coordinates, odd columns are Y coordinates
        restored[..., 0::2] = (coords_arr[..., 0::2] - pad_w) / scale
        restored[..., 1::2] = (coords_arr[..., 1::2] - pad_h) / scale
        
        # Clip coordinates to original image bounds
        restored[..., 0::2] = np.clip(restored[..., 0::2], 0, w_orig - 1)
        restored[..., 1::2] = np.clip(restored[..., 1::2], 0, h_orig - 1)
        
        if reshaped:
            restored = restored.reshape(orig_shape)
            
        return restored
