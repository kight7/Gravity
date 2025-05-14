"""
Demo script for UAV target detection system.
This script demonstrates the core functionality of the system with 
sample images or webcam feed.
"""

import os
import argparse
import cv2
import time
from typing import Optional, List

from src.detector import TargetDetector
from src.visualizer import DetectionVisualizer
from src.utils import load_image, resize_image, FPSCounter

def run_image_demo(detector: TargetDetector, visualizer: DetectionVisualizer, 
                  image_path: str, output_path: Optional[str] = None,
                  target_classes: Optional[List[int]] = None) -> None:
    """
    Run detection on a single image and visualize results.
    
    Args:
        detector: TargetDetector instance
        visualizer: DetectionVisualizer instance
        image_path: Path to input image
        output_path: Path to save output image (if None, just displays)
        target_classes: List of target class IDs to detect
    """
    # Set target classes if provided
    if target_classes is not None:
        detector.set_target_classes(target_classes)
    
    # Load and resize image
    image = load_image(image_path)
    image = resize_image(image, max_dimension=1280)
    
    # Detect objects
    detections, inference_time = detector.detect(image)
    
    # Get performance metrics
    metrics = detector.get_performance_metrics()
    
    # Print detection results
    print(f"Detected {len(detections)} objects in {inference_time:.3f} seconds")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']} ({det['confidence']:.2f})")
    
    # Visualize results
    vis_image = visualizer.draw_detections(image, detections)
    vis_image = visualizer.draw_performance_metrics(vis_image, metrics)
    vis_image = visualizer.draw_targeting_overlay(vis_image, detections, 
                                                priority_classes=[0, 2])  # Person and car
    
    # Display or save result
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Output saved to {output_path}")
    else:
        cv2.imshow("UAV Target Detection", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_webcam_demo(detector: TargetDetector, visualizer: DetectionVisualizer,
                   camera_id: int = 0, target_classes: Optional[List[int]] = None) -> None:
    """
    Run detection on webcam feed and visualize results in real-time.
    
    Args:
        detector: TargetDetector instance
        visualizer: DetectionVisualizer instance
        camera_id: Camera device ID
        target_classes: List of target class IDs to detect
    """
    # Set target classes if provided
    if target_classes is not None:
        detector.set_target_classes(target_classes)
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Initialize FPS counter
    fps_counter = FPSCounter()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            # Update FPS
            fps = fps_counter.update()
            
            # Detect objects
            detections, inference_time = detector.detect(frame)
            
            # Get performance metrics
            metrics = detector.get_performance_metrics()
            metrics['fps'] = fps
            
            # Visualize results
            vis_frame = visualizer.draw_detections(frame, detections)
            vis_frame = visualizer.draw_performance_metrics(vis_frame, metrics)
            vis_frame = visualizer.draw_targeting_overlay(vis_frame, detections, 
                                                        priority_classes=[0, 2])  # Person and car
            
            cv2.imshow("UAV Target Detection", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="UAV Target Detection Demo")
    parser.add_argument("--mode", type=str, default="webcam", choices=["image", "webcam"],
                      help="Demo mode: 'image' for single image or 'webcam' for live feed")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                      help="Path to YOLOv8 model weights")
    parser.add_argument("--image", type=str, default=None,
                      help="Path to input image (for image mode)")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save output image (for image mode)")
    parser.add_argument("--camera", type=int, default=0,
                      help="Camera device ID (for webcam mode)")
    parser.add_argument("--conf", type=float, default=0.5,
                      help="Confidence threshold for detections")
    args = parser.parse_args()
    
    print(f"Initializing detector with model {args.model}...")
    detector = TargetDetector(model_path=args.model, conf_threshold=args.conf)
    
    visualizer = DetectionVisualizer()
    
    if args.mode == "image":
        if args.image is None:
            print("Error: Image path must be specified in image mode")
            return
        
        print(f"Running detection on image {args.image}...")
        run_image_demo(detector, visualizer, args.image, args.output)
    else:
        print(f"Running live detection from camera {args.camera}...")
        run_webcam_demo(detector, visualizer, args.camera)

if __name__ == "__main__":
    main()