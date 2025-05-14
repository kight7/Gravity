"""
Visualizer module for displaying detection results.
This module provides utilities for visualizing detections and performance metrics.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class DetectionVisualizer:
    
    def __init__(self, class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None):
        
        self.class_colors = class_colors or {}
        
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        
        if class_id not in self.class_colors:
            self.class_colors[class_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        return self.class_colors[class_id]
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       show_labels: bool = True) -> np.ndarray:
        
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            color = self._get_color(class_id)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                label = f"{class_name} {confidence:.2f}"
                
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv2.rectangle(
                    vis_frame, 
                    (x1, y1 - text_height - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                cv2.putText(
                    vis_frame, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
                
        return vis_frame
    
    def draw_performance_metrics(self, frame: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
        
        vis_frame = frame.copy()
        
        # Draw FPS
        fps_text = f"FPS: {metrics.get('fps', 0):.1f}"
        cv2.putText(
            vis_frame, 
            fps_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        inf_time_text = f"Inference: {metrics.get('last_inference_time', 0) * 1000:.1f} ms"
        cv2.putText(
            vis_frame, 
            inf_time_text, 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        return vis_frame
        
    def draw_targeting_overlay(self, frame: np.ndarray, detections: List[Dict], 
                              priority_classes: Optional[List[int]] = None) -> np.ndarray:
        
        vis_frame = frame.copy()
        
        sorted_dets = sorted(
            detections,
            key=lambda d: (
                1 if priority_classes and d['class_id'] in priority_classes else 0,
                d['confidence']
            ),
            reverse=True
        )
        
        for i, det in enumerate(sorted_dets[:3]):
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            cv2.line(vis_frame, (center_x - 20, center_y), (center_x + 20, center_y), color, 2)
            cv2.line(vis_frame, (center_x, center_y - 20), (center_x, center_y + 20), color, 2)
            cv2.circle(vis_frame, (center_x, center_y), 30, color, 2)
            cv2.putText(
                vis_frame, 
                f"T{i+1}", 
                (center_x + 35, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color, 
                2, 
                cv2.LINE_AA
            )
        
        return vis_frame