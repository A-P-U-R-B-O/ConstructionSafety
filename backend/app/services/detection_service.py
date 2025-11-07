"""
Detection Service Module
Business logic layer for object detection and safety assessment
Handles frame processing, result formatting, and detection management
"""

from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque

from app.core.yolo_detector import ConstructionSafetyDetector
from app.core.model_config import get_config, DangerLevel, ObjectCategory
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionCache:
    """
    Cache for storing recent detections and statistics
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize detection cache
        
        Args:
            max_size: Maximum number of detections to cache
        """
        self.max_size = max_size
        self.detections = deque(maxlen=max_size)
        self.detection_counts = defaultdict(int)
        self.danger_level_counts = defaultdict(int)
        self.category_counts = defaultdict(int)
        self.total_processing_time = 0.0
        self.frame_count = 0
    
    def add(self, detections: List[Dict[str, Any]], processing_time: float):
        """
        Add detections to cache
        
        Args:
            detections: List of detection dictionaries
            processing_time: Time taken to process (in milliseconds)
        """
        timestamp = datetime.utcnow()
        
        cache_entry = {
            "timestamp": timestamp.isoformat(),
            "detections": detections,
            "processing_time_ms": processing_time,
            "count": len(detections)
        }
        
        self.detections.append(cache_entry)
        self.total_processing_time += processing_time
        self.frame_count += 1
        
        # Update counts
        for detection in detections:
            self.detection_counts[detection["class"]] += 1
            self.danger_level_counts[detection["danger_level"]] += 1
            self.category_counts[detection["category"]] += 1
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent detections
        
        Args:
            count: Number of recent entries to return
            
        Returns:
            List of recent detection entries
        """
        return list(self.detections)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with statistics
        """
        avg_processing_time = (
            self.total_processing_time / self.frame_count 
            if self.frame_count > 0 else 0
        )
        
        return {
            "total_frames": self.frame_count,
            "cached_entries": len(self.detections),
            "average_processing_time_ms": round(avg_processing_time, 2),
            "detection_counts": dict(self.detection_counts),
            "danger_level_counts": dict(self.danger_level_counts),
            "category_counts": dict(self.category_counts)
        }
    
    def clear(self):
        """Clear all cached data"""
        self.detections.clear()
        self.detection_counts.clear()
        self.danger_level_counts.clear()
        self.category_counts.clear()
        self.total_processing_time = 0.0
        self.frame_count = 0


class DetectionService:
    """
    Main service for managing object detection operations
    Provides high-level interface for detection processing
    """
    
    def __init__(
        self,
        detector: ConstructionSafetyDetector,
        enable_cache: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize detection service
        
        Args:
            detector: ConstructionSafetyDetector instance
            enable_cache: Whether to enable detection caching
            cache_size: Maximum cache size
        """
        self.detector = detector
        self.config = get_config()
        self.enable_cache = enable_cache
        self.cache = DetectionCache(max_size=cache_size) if enable_cache else None
        
        # Performance tracking
        self.total_detections = 0
        self.total_frames_processed = 0
        self.service_start_time = datetime.utcnow()
        
        # Detection tracking for temporal analysis
        self.recent_dangerous_objects = deque(maxlen=100)
        self.alert_cooldown: Dict[str, datetime] = {}
        
        logger.info("DetectionService initialized")
    
    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        include_landmarks: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single frame for object detection
        
        Args:
            frame: Input image as numpy array (BGR format)
            conf_threshold: Optional confidence threshold override
            iou_threshold: Optional IoU threshold override
            include_landmarks: Whether to include additional landmark info
            
        Returns:
            Dictionary with detections and metadata
        """
        start_time = time.time()
        
        try:
            # Use config defaults if not provided
            if conf_threshold is None:
                conf_threshold = self.config.model_config.confidence_threshold
            if iou_threshold is None:
                iou_threshold = self.config.model_config.iou_threshold
            
            # Run detection
            result = self.detector.detect(
                frame=frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=self.config.model_config.max_detections
            )
            
            if not result["success"]:
                logger.error(f"Detection failed: {result.get('error', 'Unknown error')}")
                return result
            
            # Enhance detections with config data
            enhanced_detections = self._enhance_detections(result["detections"])
            
            # Update statistics
            self.total_detections += len(enhanced_detections)
            self.total_frames_processed += 1
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Add to cache
            if self.enable_cache and self.cache:
                self.cache.add(enhanced_detections, processing_time)
            
            # Track dangerous objects
            dangerous_objects = [
                d for d in enhanced_detections 
                if d["danger_level"] in ["critical", "high"]
            ]
            if dangerous_objects:
                self.recent_dangerous_objects.extend(dangerous_objects)
            
            return {
                "success": True,
                "detections": enhanced_detections,
                "total_objects": len(enhanced_detections),
                "dangerous_objects": len(dangerous_objects),
                "processing_time_ms": round(processing_time, 2),
                "frame_dimensions": result["frame_dimensions"],
                "danger_score": result["danger_score"],
                "danger_level": result["danger_level"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "total_objects": 0
            }
    
    def process_frame_fast(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Fast frame processing for real-time streaming
        Optimized with minimal overhead
        
        Args:
            frame: Input image as numpy array
            conf_threshold: Confidence threshold
            
        Returns:
            Lightweight detection result
        """
        try:
            result = self.detector.detect(
                frame=frame,
                conf_threshold=conf_threshold,
                iou_threshold=0.5,
                max_detections=50  # Limit for performance
            )
            
            if not result["success"]:
                return result
            
            # Minimal enhancement
            detections = result["detections"]
            
            self.total_frames_processed += 1
            
            return {
                "success": True,
                "detections": detections,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fast processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "detections": []
            }
    
    def _enhance_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance detections with additional config-based information
        
        Args:
            detections: List of raw detections
            
        Returns:
            List of enhanced detections
        """
        enhanced = []
        
        for detection in detections:
            class_name = detection["class"]
            class_config = self.config.get_class_config(class_name)
            
            # Add config-based enhancements
            if class_config:
                detection["alert_message"] = class_config.alert_message
                detection["priority"] = class_config.priority
                detection["description"] = class_config.description
                detection["min_confidence"] = class_config.min_confidence
                
                # Flag low confidence detections
                if detection["confidence"] < class_config.min_confidence:
                    detection["low_confidence_warning"] = True
            else:
                detection["alert_message"] = f"{class_name} detected"
                detection["priority"] = 5
                detection["description"] = "Unknown object"
                detection["low_confidence_warning"] = False
            
            enhanced.append(detection)
        
        return enhanced
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        include_labels: bool = True,
        include_confidence: bool = True,
        box_thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        Wrapper around detector's draw method with additional features
        
        Args:
            frame: Input frame to annotate
            detections: List of detections to draw
            include_labels: Whether to show labels
            include_confidence: Whether to show confidence scores
            box_thickness: Thickness of bounding box lines
            font_scale: Scale of text labels
            
        Returns:
            Annotated frame
        """
        annotated = self.detector.draw_detections(
            frame=frame,
            detections=detections,
            show_labels=include_labels,
            show_confidence=include_confidence,
            box_thickness=box_thickness,
            font_scale=font_scale
        )
        
        # Add frame-level information overlay
        annotated = self._add_info_overlay(annotated, detections)
        
        return annotated
    
    def _add_info_overlay(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Add informational overlay to frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with overlay
        """
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Create semi-transparent overlay panel
        panel_height = 80
        panel_color = (0, 0, 0)  # Black
        cv2.rectangle(
            overlay,
            (0, 0),
            (width, panel_height),
            panel_color,
            -1
        )
        
        # Blend with original
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        
        # Total objects
        text = f"Objects: {len(detections)}"
        cv2.putText(frame, text, (10, 25), font, font_scale, text_color, font_thickness)
        
        # Dangerous objects count
        dangerous = len([d for d in detections if d["danger_level"] in ["critical", "high"]])
        danger_color = (0, 0, 255) if dangerous > 0 else (0, 255, 0)
        text = f"Dangerous: {dangerous}"
        cv2.putText(frame, text, (10, 55), font, font_scale, danger_color, font_thickness)
        
        # Timestamp
        timestamp_text = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        text_size = cv2.getTextSize(timestamp_text, font, font_scale * 0.7, 1)[0]
        text_x = width - text_size[0] - 10
        cv2.putText(
            frame,
            timestamp_text,
            (text_x, 25),
            font,
            font_scale * 0.7,
            text_color,
            1
        )
        
        return frame
    
    def analyze_temporal_patterns(
        self,
        time_window_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze detection patterns over time
        
        Args:
            time_window_minutes: Time window for analysis
            
        Returns:
            Dictionary with temporal analysis results
        """
        if not self.enable_cache or not self.cache:
            return {"error": "Cache not enabled"}
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        # Get recent detections within time window
        recent = [
            entry for entry in self.cache.detections
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
        
        if not recent:
            return {"message": "No recent detections in time window"}
        
        # Aggregate statistics
        total_detections = sum(entry["count"] for entry in recent)
        avg_objects_per_frame = total_detections / len(recent) if recent else 0
        
        # Find most common objects
        object_counts = defaultdict(int)
        danger_counts = defaultdict(int)
        
        for entry in recent:
            for detection in entry["detections"]:
                object_counts[detection["class"]] += 1
                danger_counts[detection["danger_level"]] += 1
        
        # Sort by frequency
        most_common = sorted(
            object_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "time_window_minutes": time_window_minutes,
            "frames_analyzed": len(recent),
            "total_detections": total_detections,
            "average_objects_per_frame": round(avg_objects_per_frame, 2),
            "most_common_objects": [
                {"class": obj, "count": count} for obj, count in most_common
            ],
            "danger_level_distribution": dict(danger_counts),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive service statistics
        
        Returns:
            Dictionary with service statistics
        """
        uptime = datetime.utcnow() - self.service_start_time
        
        stats = {
            "service_uptime_seconds": uptime.total_seconds(),
            "total_frames_processed": self.total_frames_processed,
            "total_detections": self.total_detections,
            "average_detections_per_frame": (
                self.total_detections / self.total_frames_processed
                if self.total_frames_processed > 0 else 0
            ),
            "detector_stats": self.detector.get_statistics()
        }
        
        # Add cache statistics if enabled
        if self.enable_cache and self.cache:
            stats["cache_stats"] = self.cache.get_statistics()
        
        # Add recent dangerous objects count
        stats["recent_dangerous_objects"] = len(self.recent_dangerous_objects)
        
        return stats
    
    def get_all_classes(self) -> List[Dict[str, Any]]:
        """
        Get all supported classes with their configurations
        
        Returns:
            List of class dictionaries
        """
        classes = []
        
        for class_name, class_config in self.config.danger_classes.items():
            classes.append({
                "name": class_name,
                "danger_level": class_config.danger_level.value,
                "category": class_config.category.value,
                "requires_ppe": class_config.requires_ppe,
                "priority": class_config.priority,
                "description": class_config.description,
                "alert_message": class_config.alert_message
            })
        
        # Sort by priority (highest first)
        classes.sort(key=lambda x: x["priority"], reverse=True)
        
        return classes
    
    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        danger_levels: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        min_priority: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter detections based on criteria
        
        Args:
            detections: List of detections to filter
            danger_levels: Filter by danger levels
            categories: Filter by categories
            min_confidence: Minimum confidence threshold
            min_priority: Minimum priority level
            
        Returns:
            Filtered list of detections
        """
        filtered = detections
        
        if danger_levels:
            filtered = [d for d in filtered if d["danger_level"] in danger_levels]
        
        if categories:
            filtered = [d for d in filtered if d["category"] in categories]
        
        if min_confidence is not None:
            filtered = [d for d in filtered if d["confidence"] >= min_confidence]
        
        if min_priority is not None:
            filtered = [d for d in filtered if d.get("priority", 0) >= min_priority]
        
        return filtered
    
    def check_proximity_alert(
        self,
        detections: List[Dict[str, Any]],
        max_distance_pixels: float = 100
    ) -> List[Dict[str, Any]]:
        """
        Check for dangerous proximity between objects
        
        Args:
            detections: List of detections
            max_distance_pixels: Maximum distance for proximity alert
            
        Returns:
            List of proximity alerts
        """
        alerts = []
        
        # Find people and vehicles
        people = [d for d in detections if d["category"] == "people"]
        vehicles = [d for d in detections if d["category"] == "vehicle"]
        
        # Check each person against each vehicle
        for person in people:
            person_center = (
                person["bbox"]["center_x"],
                person["bbox"]["center_y"]
            )
            
            for vehicle in vehicles:
                vehicle_center = (
                    vehicle["bbox"]["center_x"],
                    vehicle["bbox"]
)
                
                # Calculate distance
                distance = np.sqrt(
                    (person_center[0] - vehicle_center[0]) ** 2 +
                    (person_center[1] - vehicle_center[1]) ** 2
                )
                
                if distance <= max_distance_pixels:
                    alerts.append({
                        "type": "proximity_alert",
                        "severity": "critical",
                        "message": f"⚠️ DANGER: Person detected near {vehicle['class']}!",
                        "distance_pixels": round(distance, 2),
                        "person": person,
                        "vehicle": vehicle,
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        return alerts
    
    def clear_cache(self):
        """Clear detection cache and reset statistics"""
        if self.cache:
            self.cache.clear()
        self.total_detections = 0
        self.total_frames_processed = 0
        self.recent_dangerous_objects.clear()
        self.alert_cooldown.clear()
        logger.info("Detection service cache cleared")
    
    def __repr__(self) -> str:
        return (
            f"DetectionService("
            f"frames_processed={self.total_frames_processed}, "
            f"total_detections={self.total_detections}, "
            f"cache_enabled={self.enable_cache})"
        )
