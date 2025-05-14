"""
Core detection module for UAV target detection system.
This module handles the loading of YOLOv8 models and performs object detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Dict, Tuple, Optional, Union

class TargetDetector:
    """YOLOv8-based detector for identifying and tracking targets in UAV footage."""
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5, 
                 target_classes: Optional[List[int]] = None):
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        self.last_inference_time = 0
        
        print(f"Model loaded successfully. Available classes: {self.model.names}")
        
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        
        start_time = time.time()
        
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            
            if conf < self.conf_threshold:
                continue
                
            if self.target_classes is not None and int(cls) not in self.target_classes:
                continue
                
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': self.model.names[int(cls)]
            })
        
        inference_time = time.time() - start_time
        self.last_inference_time = inference_time
        
        return detections, inference_time
    
    def set_target_classes(self, target_classes: List[int]) -> None:
        
        self.target_classes = target_classes
        
    def get_performance_metrics(self) -> Dict[str, float]:
        
        return {
            'last_inference_time': self.last_inference_time,
            'fps': 1.0 / max(self.last_inference_time, 1e-5),
        }