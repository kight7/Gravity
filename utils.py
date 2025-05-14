"""
Utility functions for the UAV target detection system.
"""

import cv2
import numpy as np
import os
import time
from typing import List, Dict, Tuple, Optional, Union, Any

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from path.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Image as numpy array (BGR)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    return img

def resize_image(image: np.ndarray, width: Optional[int] = None, 
                height: Optional[int] = None, 
                max_dimension: Optional[int] = None) -> np.ndarray:
    """
    Resize an image, maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (if None, calculated from height)
        height: Target height (if None, calculated from width)
        max_dimension: Maximum dimension (overrides width/height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max_dimension is not None:
        if h > w:
            height = max_dimension
            width = None
        else:
            width = max_dimension
            height = None
    
    if width is None and height is None:
        return image
        
    if width is None:
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get the center point of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Center point (x, y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def get_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Area of the bounding box
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    # Get coordinates of intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Check if there is an intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate area of intersection
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    
    return iou

class FPSCounter:
    """Utility class for measuring frames per second."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize the FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.frame_times = []
        self.window_size = window_size
        self.last_frame_time = None
        
    def update(self) -> float:
        """
        Update the FPS counter with a new frame.
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        
        if self.last_frame_time is not None:
            # Calculate time since last frame
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            
            # Keep only the last window_size frame times
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_frame_time = current_time
        
        # Calculate FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / max(avg_frame_time, 1e-5)
        else:
            fps = 0.0
            
        return fps