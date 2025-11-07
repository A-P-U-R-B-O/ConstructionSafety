"""
YOLO Detector Core Module
Wrapper for YOLOv8 model with construction safety-specific detection logic
Handles object detection, danger classification, and safety assessment
"""

from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
import json
import os
from pathlib import Path
import time
from datetime import datetime

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConstructionSafetyDetector:
    """
    Main detector class for construction site safety monitoring
    Uses pre-trained YOLOv8 models to detect objects and assess dangers
    """
    
    # COCO dataset classes that are relevant for construction safety
    CONSTRUCTION_RELEVANT_CLASSES = {
        # People and vehicles
        "person": {"danger_level": "high", "category": "people", "requires_ppe": True},
        "bicycle": {"danger_level": "medium", "category": "vehicle", "requires_ppe": False},
        "car": {"danger_level": "high", "category": "vehicle", "requires_ppe": False},
        "motorcycle": {"danger_level": "high", "category": "vehicle", "requires_ppe": False},
        "bus": {"danger_level": "high", "category": "vehicle", "requires_ppe": False},
        "truck": {"danger_level": "critical", "category": "vehicle", "requires_ppe": False},
        
        # Sharp/dangerous objects
        "knife": {"danger_level": "critical", "category": "sharp_object", "requires_ppe": True},
        "scissors": {"danger_level": "medium", "category": "sharp_object", "requires_ppe": False},
        
        # Equipment and obstacles
        "chair": {"danger_level": "low", "category": "obstacle", "requires_ppe": False},
        "bench": {"danger_level": "low", "category": "obstacle", "requires_ppe": False},
        "backpack": {"danger_level": "low", "category": "equipment", "requires_ppe": False},
        "handbag": {"danger_level": "low", "category": "equipment", "requires_ppe": False},
        "suitcase": {"danger_level": "low", "category": "obstacle", "requires_ppe": False},
        
        # Potential hazards
        "bottle": {"danger_level": "low", "category": "container", "requires_ppe": False},
        "cup": {"danger_level": "low", "category": "container", "requires_ppe": False},
        "fire hydrant": {"danger_level": "medium", "category": "equipment", "requires_ppe": False},
        "stop sign": {"danger_level": "low", "category": "sign", "requires_ppe": False},
        
        # Animals (unexpected on construction sites)
        "bird": {"danger_level": "low", "category": "animal", "requires_ppe": False},
        "cat": {"danger_level": "low", "category": "animal", "requires_ppe": False},
        "dog": {"danger_level": "medium", "category": "animal", "requires_ppe": False},
        "horse": {"danger_level": "medium", "category": "animal", "requires_ppe": False},
        
        # Tools and equipment
        "laptop": {"danger_level": "low", "category": "equipment", "requires_ppe": False},
        "cell phone": {"danger_level": "low", "category": "equipment", "requires_ppe": False},
        "book": {"danger_level": "low", "category": "equipment", "requires_ppe": False},
        "clock": {"danger_level": "low", "category": "equipment", "requires_ppe": False},
        "umbrella": {"danger_level": "low", "category": "equipment", "requires_ppe": False},
    }
    
    # Danger level scoring
    DANGER_SCORES = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
        "safe": 0
    }
    
    # Color coding for visualization (BGR format for OpenCV)
    DANGER_COLORS = {
        "critical": (0, 0, 255),      # Red
        "high": (0, 69, 255),          # Orange-Red
        "medium": (0, 165, 255),       # Orange
        "low": (0, 255, 255),          # Yellow
        "safe": (0, 255, 0)            # Green
    }
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
        config_path: Optional[str] = None
    ):
        """
        Initialize the Construction Safety Detector
        
        Args:
            model_name: YOLOv8 model to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            config_path: Optional path to custom danger classification config
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.class_names = {}
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.detection_history = []
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            self._load_custom_config(config_path)
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"ConstructionSafetyDetector initialized with model: {model_name} on device: {device}")
    
    def _initialize_model(self):
        """
        Load and initialize the YOLO model
        """
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_name}")
            
            # Load pre-trained YOLO model
            self.model = YOLO(self.model_name)
            
            # Move to specified device
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Get class names from model
            self.class_names = self.model.names
            
            logger.info(f"Model loaded successfully. Supports {len(self.class_names)} classes")
            logger.info(f"Tracking {len(self.CONSTRUCTION_RELEVANT_CLASSES)} construction-relevant classes")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def _load_custom_config(self, config_path: str):
        """
        Load custom danger classification configuration
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                
            if "danger_classes" in custom_config:
                self.CONSTRUCTION_RELEVANT_CLASSES.update(custom_config["danger_classes"])
                logger.info(f"Loaded custom danger classification from {config_path}")
                
        except Exception as e:
            logger.warning(f"Failed to load custom config: {str(e)}")
    
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_detections: int = 100
    ) -> Dict[str, Any]:
        """
        Perform object detection on a single frame
        
        Args:
            frame: Input image as numpy array (BGR format)
            conf_threshold: Confidence threshold for detections (0.0 - 1.0)
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to return
            
        Returns:
            Dictionary containing detections, metadata, and timing information
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")
        
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                verbose=False
            )
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Process results
            detections = self._process_results(results, frame.shape)
            
            # Calculate overall danger score
            danger_score = self._calculate_danger_score(detections)
            
            # Update statistics
            self.total_detections += len(detections)
            self.total_inference_time += inference_time
            
            # Store in history (keep last 100)
            self.detection_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "num_detections": len(detections),
                "danger_score": danger_score
            })
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            
            return {
                "success": True,
                "detections": detections,
                "total_objects": len(detections),
                "danger_score": danger_score,
                "danger_level": self._get_overall_danger_level(danger_score),
                "processing_time_ms": round(inference_time, 2),
                "frame_dimensions": {
                    "height": frame.shape[0],
                    "width": frame.shape[1],
                    "channels": frame.shape[2] if len(frame.shape) > 2 else 1
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "total_objects": 0
            }
    
    def _process_results(
        self,
        results: List[Any],
        frame_shape: Tuple[int, int, int]
    ) -> List[Dict[str, Any]]:
        """
        Process YOLO results into structured detection data
        
        Args:
            results: YOLO model results
            frame_shape: Shape of input frame (height, width, channels)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Extract box data
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                # Get class name
                class_name = self.class_names.get(cls_id, "unknown")
                
                # Get danger classification
                danger_info = self.CONSTRUCTION_RELEVANT_CLASSES.get(
                    class_name,
                    {"danger_level": "low", "category": "unknown", "requires_ppe": False}
                )
                
                # Calculate box center and area
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Calculate relative position in frame
                frame_height, frame_width = frame_shape[:2]
                relative_position = {
                    "center_x_percent": round((center_x / frame_width) * 100, 2),
                    "center_y_percent": round((center_y / frame_height) * 100, 2),
                    "area_percent": round((area / (frame_width * frame_height)) * 100, 2)
                }
                
                detection = {
                    "class_id": cls_id,
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x1": round(float(x1), 2),
                        "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2),
                        "y2": round(float(y2), 2),
                        "width": round(float(width), 2),
                        "height": round(float(height), 2),
                        "center_x": round(float(center_x), 2),
                        "center_y": round(float(center_y), 2)
                    },
                    "danger_level": danger_info["danger_level"],
                    "category": danger_info["category"],
                    "requires_ppe": danger_info["requires_ppe"],
                    "relative_position": relative_position,
                    "danger_score": self.DANGER_SCORES[danger_info["danger_level"]]
                }
                
                detections.append(detection)
        
        # Sort by danger score (highest first)
        detections.sort(key=lambda x: x["danger_score"], reverse=True)
        
        return detections
    
    def _calculate_danger_score(self, detections: List[Dict[str, Any]]) -> float:
        """
        Calculate overall danger score for the frame
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Float danger score (0.0 - 10.0)
        """
        if not detections:
            return 0.0
        
        # Weight by confidence and danger level
        total_score = sum(
            d["danger_score"] * d["confidence"] 
            for d in detections
        )
        
        # Normalize to 0-10 scale
        max_possible = len(detections) * 4  # 4 is max danger score (critical)
        normalized_score = (total_score / max_possible) * 10 if max_possible > 0 else 0
        
        return round(normalized_score, 2)
    
    def _get_overall_danger_level(self, danger_score: float) -> str:
        """
        Convert danger score to categorical danger level
        
        Args:
            danger_score: Numeric danger score (0-10)
            
        Returns:
            String danger level
        """
        if danger_score >= 7.5:
            return "critical"
        elif danger_score >= 5.0:
            return "high"
        elif danger_score >= 2.5:
            return "medium"
        elif danger_score > 0:
            return "low"
        else:
            return "safe"
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True,
        show_confidence: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame to draw on
            detections: List of detections to draw
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            box_thickness: Thickness of bounding box lines
            font_scale: Scale of text font
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            
            # Get color based on danger level
            color = self.DANGER_COLORS.get(detection["danger_level"], (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)
            
            # Prepare label text
            if show_labels or show_confidence:
                label_parts = []
                
                if show_labels:
                    label_parts.append(detection["class"])
                
                if show_confidence:
                    label_parts.append(f"{detection['confidence']:.2f}")
                
                label = " ".join(label_parts)
                
                # Add danger level indicator
                danger_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                    "safe": "⚪"
                }
                label = f"{danger_emoji.get(detection['danger_level'], '')} {label}"
                
                # Calculate label size
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        return annotated_frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detector statistics
        
        Returns:
            Dictionary with statistics
        """
        avg_inference_time = (
            self.total_inference_time / self.total_detections 
            if self.total_detections > 0 else 0
        )
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "total_detections": self.total_detections,
            "average_processing_time_ms": round(avg_inference_time, 2),
            "supported_classes": len(self.class_names),
            "tracked_danger_classes": len(self.CONSTRUCTION_RELEVANT_CLASSES),
            "detection_history_size": len(self.detection_history),
            "all_classes": list(self.class_names.values()),
            "danger_classes": list(self.CONSTRUCTION_RELEVANT_CLASSES.keys())
        }
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is properly loaded
        
        Returns:
            Boolean indicating model status
        """
        return self.model is not None
    
    def get_class_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific class
        
        Args:
            class_name: Name of the class
            
        Returns:
            Dictionary with class information or None
        """
        return self.CONSTRUCTION_RELEVANT_CLASSES.get(class_name)
    
    def reset_statistics(self):
        """
        Reset detection statistics
        """
        self.total_detections = 0
        self.total_inference_time = 0.0
        self.detection_history.clear()
        logger.info("Detector statistics reset")
    
    def __repr__(self) -> str:
        return f"ConstructionSafetyDetector(model={self.model_name}, device={self.device}, detections={self.total_detections})"
