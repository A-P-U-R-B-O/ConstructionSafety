"""
Detection Schemas Module
Pydantic models for request/response validation and serialization
Ensures type safety and data validation across the API
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class DangerLevelEnum(str, Enum):
    """Danger level enumeration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


class ObjectCategoryEnum(str, Enum):
    """Object category enumeration"""
    PEOPLE = "people"
    VEHICLE = "vehicle"
    SHARP_OBJECT = "sharp_object"
    OBSTACLE = "obstacle"
    EQUIPMENT = "equipment"
    CONTAINER = "container"
    ANIMAL = "animal"
    SIGN = "sign"
    TOOL = "tool"
    MACHINERY = "machinery"
    HAZARD = "hazard"
    UNKNOWN = "unknown"


class AlertSeverityEnum(str, Enum):
    """Alert severity enumeration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertTypeEnum(str, Enum):
    """Alert type enumeration"""
    DANGER_DETECTION = "danger_detection"
    PROXIMITY_WARNING = "proximity_warning"
    PPE_VIOLATION = "ppe_violation"
    MULTIPLE_HAZARDS = "multiple_hazards"
    VEHICLE_WARNING = "vehicle_warning"
    SHARP_OBJECT = "sharp_object"
    ANIMAL_DETECTED = "animal_detected"
    UNUSUAL_ACTIVITY = "unusual_activity"
    LOW_CONFIDENCE = "low_confidence"
    PATTERN_ALERT = "pattern_alert"


# ============================================================================
# Base Schemas
# ============================================================================

class BoundingBox(BaseModel):
    """Bounding box coordinates and dimensions"""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")
    width: float = Field(..., description="Box width")
    height: float = Field(..., description="Box height")
    center_x: float = Field(..., description="Center X coordinate")
    center_y: float = Field(..., description="Center Y coordinate")
    
    class Config:
        schema_extra = {
            "example": {
                "x1": 100.5,
                "y1": 150.3,
                "x2": 250.8,
                "y2": 350.6,
                "width": 150.3,
                "height": 200.3,
                "center_x": 175.65,
                "center_y": 250.45
            }
        }


class RelativePosition(BaseModel):
    """Relative position in frame (percentage)"""
    center_x_percent: float = Field(..., ge=0, le=100, description="Center X as percentage")
    center_y_percent: float = Field(..., ge=0, le=100, description="Center Y as percentage")
    area_percent: float = Field(..., ge=0, le=100, description="Area as percentage of frame")
    
    class Config:
        schema_extra = {
            "example": {
                "center_x_percent": 52.3,
                "center_y_percent": 48.7,
                "area_percent": 12.5
            }
        }


class FrameDimensions(BaseModel):
    """Frame dimensions"""
    height: int = Field(..., gt=0, description="Frame height in pixels")
    width: int = Field(..., gt=0, description="Frame width in pixels")
    channels: int = Field(default=3, ge=1, le=4, description="Number of color channels")
    
    class Config:
        schema_extra = {
            "example": {
                "height": 480,
                "width": 640,
                "channels": 3
            }
        }


# ============================================================================
# Detection Schemas
# ============================================================================

class Detection(BaseModel):
    """Single object detection"""
    class_id: int = Field(..., description="Class ID from model")
    class_name: str = Field(..., alias="class", description="Object class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    danger_level: DangerLevelEnum = Field(..., description="Danger level classification")
    category: ObjectCategoryEnum = Field(..., description="Object category")
    requires_ppe: bool = Field(..., description="Whether PPE is required")
    relative_position: RelativePosition = Field(..., description="Position relative to frame")
    danger_score: int = Field(..., ge=0, le=10, description="Numerical danger score")
    
    # Optional enhanced fields
    alert_message: Optional[str] = Field(None, description="Alert message for this object")
    priority: Optional[int] = Field(None, ge=1, le=10, description="Priority level")
    description: Optional[str] = Field(None, description="Object description")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    low_confidence_warning: Optional[bool] = Field(False, description="Low confidence flag")
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "class_id": 0,
                "class": "person",
                "confidence": 0.92,
                "bbox": {
                    "x1": 100.0,
                    "y1": 150.0,
                    "x2": 250.0,
                    "y2": 400.0,
                    "width": 150.0,
                    "height": 250.0,
                    "center_x": 175.0,
                    "center_y": 275.0
                },
                "danger_level": "high",
                "category": "people",
                "requires_ppe": True,
                "relative_position": {
                    "center_x_percent": 27.34,
                    "center_y_percent": 57.29,
                    "area_percent": 12.24
                },
                "danger_score": 7,
                "alert_message": "Worker detected - verify PPE",
                "priority": 10,
                "description": "Human worker on construction site"
            }
        }


class Alert(BaseModel):
    """Safety alert"""
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: AlertTypeEnum = Field(..., description="Type of alert")
    severity: AlertSeverityEnum = Field(..., description="Alert severity level")
    message: str = Field(..., description="Alert message")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional alert details")
    timestamp: str = Field(..., description="Alert timestamp (ISO format)")
    acknowledged: bool = Field(default=False, description="Whether alert is acknowledged")
    resolved: bool = Field(default=False, description="Whether alert is resolved")
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4e5f6",
                "alert_type": "proximity_warning",
                "severity": "critical",
                "message": "🚨 DANGER: Worker too close to truck!",
                "details": {
                    "person": "person",
                    "vehicle": "truck",
                    "distance_pixels": 85.3
                },
                "timestamp": "2025-11-07T12:50:21Z",
                "acknowledged": False,
                "resolved": False
            }
        }


# ============================================================================
# Request Schemas
# ============================================================================

class DetectionRequest(BaseModel):
    """Request for detection from base64 image"""
    image_base64: str = Field(..., description="Base64 encoded image (without data:image prefix)")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    resize: bool = Field(default=True, description="Whether to resize image for performance")
    include_landmarks: bool = Field(default=False, description="Include landmark detection")
    alert_severity: Optional[str] = Field(None, description="Filter alerts by minimum severity")
    
    @validator('image_base64')
    def validate_base64(cls, v):
        """Validate base64 string"""
        if not v or len(v) < 100:
            raise ValueError("Invalid or too short base64 string")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "image_base64": "/9j/4AAQSkZJRgABAQEAYABgAAD...",
                "confidence_threshold": 0.5,
                "resize": True,
                "include_landmarks": False,
                "alert_severity": "high"
            }
        }


class BatchDetectionRequest(BaseModel):
    """Request for batch detection"""
    images: List[str] = Field(..., min_items=1, max_items=10, description="List of base64 images")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @validator('images')
    def validate_images(cls, v):
        """Validate image list"""
        if len(v) > 10:
            raise ValueError("Maximum 10 images per batch")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "images": [
                    "/9j/4AAQSkZJRgABAQEAYABgAAD...",
                    "/9j/4AAQSkZJRgABAQEAYABgAAD..."
                ],
                "confidence_threshold": 0.5
            }
        }


# ============================================================================
# Response Schemas
# ============================================================================

class DetectionResponse(BaseModel):
    """Response for single detection"""
    success: bool = Field(..., description="Whether detection succeeded")
    detections: List[Detection] = Field(default_factory=list, description="List of detections")
    alerts: List[Alert] = Field(default_factory=list, description="Generated alerts")
    total_objects: int = Field(..., description="Total objects detected")
    dangerous_objects: int = Field(..., description="Number of dangerous objects")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    frame_dimensions: FrameDimensions = Field(..., description="Frame dimensions")
    danger_score: Optional[float] = Field(None, ge=0, le=10, description="Overall danger score")
    danger_level: Optional[str] = Field(None, description="Overall danger level")
    timestamp: Optional[str] = Field(None, description="Processing timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "detections": [],
                "alerts": [],
                "total_objects": 5,
                "dangerous_objects": 2,
                "processing_time_ms": 45.23,
                "frame_dimensions": {
                    "height": 480,
                    "width": 640,
                    "channels": 3
                },
                "danger_score": 7.5,
                "danger_level": "high",
                "timestamp": "2025-11-07T12:50:21Z"
            }
        }


class ImageDetectionResponse(BaseModel):
    """Response for image upload detection"""
    success: bool = Field(..., description="Whether detection succeeded")
    filename: str = Field(..., description="Original filename")
    detections: List[Detection] = Field(default_factory=list, description="List of detections")
    alerts: List[Alert] = Field(default_factory=list, description="Generated alerts")
    total_objects: int = Field(..., description="Total objects detected")
    dangerous_objects: int = Field(..., description="Number of dangerous objects")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    annotated_image_base64: Optional[str] = Field(None, description="Annotated image (base64)")
    frame_dimensions: FrameDimensions = Field(..., description="Frame dimensions")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "filename": "construction_site.jpg",
                "detections": [],
                "alerts": [],
                "total_objects": 3,
                "dangerous_objects": 1,
                "processing_time_ms": 52.18,
                "annotated_image_base64": "/9j/4AAQSkZJRgABAQEAYABgAAD...",
                "frame_dimensions": {
                    "height": 1080,
                    "width": 1920,
                    "channels": 3
                }
            }
        }


class BatchDetectionResult(BaseModel):
    """Single result in batch detection"""
    filename: str = Field(..., description="Image identifier")
    success: bool = Field(..., description="Whether detection succeeded")
    detections: Optional[List[Detection]] = Field(None, description="List of detections")
    alerts: Optional[List[Alert]] = Field(None, description="Generated alerts")
    total_objects: Optional[int] = Field(None, description="Total objects detected")
    dangerous_objects: Optional[int] = Field(None, description="Dangerous objects count")
    processing_time_ms: Optional[float] = Field(None, description="Processing time")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "image_001.jpg",
                "success": True,
                "detections": [],
                "alerts": [],
                "total_objects": 4,
                "dangerous_objects": 1,
                "processing_time_ms": 48.5,
                "error": None
            }
        }


class BatchDetectionResponse(BaseModel):
    """Response for batch detection"""
    success: bool = Field(..., description="Whether batch processing succeeded")
    total_images: int = Field(..., description="Total images in batch")
    successful_detections: int = Field(..., description="Successfully processed images")
    failed_detections: int = Field(..., description="Failed image processing count")
    results: List[BatchDetectionResult] = Field(..., description="Individual results")
    total_objects_detected: int = Field(..., description="Total objects across all images")
    total_dangerous_objects: int = Field(..., description="Total dangerous objects")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_images": 5,
                "successful_detections": 5,
                "failed_detections": 0,
                "results": [],
                "total_objects_detected": 23,
                "total_dangerous_objects": 7
            }
        }


class DetectionStatistics(BaseModel):
    """Detection service statistics"""
    model_name: str = Field(..., description="Model name being used")
    total_detections: int = Field(..., description="Total detections processed")
    total_alerts_generated: int = Field(..., description="Total alerts generated")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    supported_classes: int = Field(..., description="Number of supported classes")
    danger_classes: List[str] = Field(..., description="List of danger classes")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "yolov8n.pt",
                "total_detections": 1523,
                "total_alerts_generated": 342,
                "average_processing_time_ms": 47.8,
                "supported_classes": 80,
                "danger_classes": ["person", "truck", "car", "knife"],
                "uptime_seconds": 3600.5
            }
        }


# ============================================================================
# WebSocket Schemas
# ============================================================================

class WebSocketFrameRequest(BaseModel):
    """WebSocket frame request"""
    frame: str = Field(..., description="Base64 encoded frame")
    timestamp: Optional[str] = Field(None, description="Client timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "frame": "/9j/4AAQSkZJRgABAQEAYABgAAD...",
                "timestamp": "2025-11-07T12:50:21Z",
                "metadata": {
                    "camera_id": "cam_01",
                    "location": "Zone A"
                }
            }
        }


class WebSocketDetectionResponse(BaseModel):
    """WebSocket detection response"""
    type: str = Field(default="detection_result", description="Message type")
    detections: List[Detection] = Field(default_factory=list, description="Detections")
    alerts: List[Alert] = Field(default_factory=list, description="Alerts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "detection_result",
                "detections": [],
                "alerts": [],
                "metadata": {
                    "total_objects": 5,
                    "dangerous_objects": 2,
                    "processing_time_ms": 42.3,
                    "frame_number": 156,
                    "server_timestamp": "2025-11-07T12:50:21Z"
                }
            }
        }


# ============================================================================
# Danger Assessment Schemas
# ============================================================================

class SpatialRelationshipSchema(BaseModel):
    """Spatial relationship between objects"""
    object1: str = Field(..., description="First object class")
    object2: str = Field(..., description="Second object class")
    distance: float = Field(..., description="Distance in pixels")
    relative_position: str = Field(..., description="Relative position")
    interaction_type: str = Field(..., description="Interaction type")
    risk_score: float = Field(..., ge=0, le=10, description="Risk score")
    
    class Config:
        schema_extra = {
            "example": {
                "object1": "person",
                "object2": "truck",
                "distance": 85.3,
                "relative_position": "left",
                "interaction_type": "proximity_warning",
                "risk_score": 8.7
            }
        }


class DangerZoneSchema(BaseModel):
    """Danger zone information"""
    zone_id: str = Field(..., description="Zone identifier")
    center: Dict[str, float] = Field(..., description="Zone center coordinates")
    radius: float = Field(..., description="Zone radius")
    risk_level: str = Field(..., description="Risk level")
    object_count: int = Field(..., description="Objects in zone")
    objects: List[str] = Field(..., description="Object classes in zone")
    risk_score: float = Field(..., ge=0, le=10, description="Zone risk score")
    description: str = Field(..., description="Zone description")
    
    class Config:
        schema_extra = {
            "example": {
                "zone_id": "critical_0",
                "center": {"x": 320.5, "y": 240.8},
                "radius": 100.0,
                "risk_level": "critical_zone",
                "object_count": 2,
                "objects": ["truck", "person"],
                "risk_score": 10.0,
                "description": "Critical danger zone around truck"
            }
        }


class DangerousInteraction(BaseModel):
    """Dangerous object interaction"""
    object1: str = Field(..., description="First object")
    object2: str = Field(..., description="Second object")
    distance: float = Field(..., description="Distance between objects")
    base_risk: float = Field(..., description="Base risk score")
    actual_risk: float = Field(..., description="Actual risk score")
    warning: str = Field(..., description="Warning message")
    
    class Config:
        schema_extra = {
            "example": {
                "object1": "person",
                "object2": "truck",
                "distance": 75.5,
                "base_risk": 10.0,
                "actual_risk": 9.2,
                "warning": "Dangerous combination: person near truck"
            }
        }


class DangerAssessmentResponse(BaseModel):
    """Comprehensive danger assessment response"""
    success: bool = Field(..., description="Assessment success")
    timestamp: str = Field(..., description="Assessment timestamp")
    risk_level: str = Field(..., description="Overall risk level")
    comprehensive_score: float = Field(..., ge=0, le=10, description="Comprehensive risk score")
    base_danger_score: float = Field(..., ge=0, le=10, description="Base danger score")
    spatial_risk_score: float = Field(..., ge=0, le=10, description="Spatial risk score")
    interaction_risk_score: float = Field(..., ge=0, le=10, description="Interaction risk score")
    temporal_risk_score: float = Field(..., ge=0, le=10, description="Temporal risk score")
    danger_zones: List[DangerZoneSchema] = Field(default_factory=list, description="Identified danger zones")
    spatial_relationships: List[SpatialRelationshipSchema] = Field(default_factory=list, description="Spatial relationships")
    dangerous_interactions: List[DangerousInteraction] = Field(default_factory=list, description="Dangerous interactions")
    recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")
    statistics: Dict[str, int] = Field(default_factory=dict, description="Assessment statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2025-11-07T12:50:21Z",
                "risk_level": "high_risk_zone",
                "comprehensive_score": 7.8,
                "base_danger_score": 7.2,
                "spatial_risk_score": 8.5,
                "interaction_risk_score": 9.0,
                "temporal_risk_score": 5.5,
                "danger_zones": [],
                "spatial_relationships": [],
                "dangerous_interactions": [],
                "recommendations": [
                    "⚠️ HIGH RISK: Exercise extreme caution",
                    "⚠️ Ensure workers maintain safe distance from vehicles"
                ],
                "statistics": {
                    "total_objects": 5,
                    "critical_objects": 1,
                    "high_danger_objects": 2,
                    "danger_zone_count": 1,
                    "risky_interactions": 1
                }
            }
        }


# ============================================================================
# Class Information Schemas
# ============================================================================

class ClassInfo(BaseModel):
    """Information about a detection class"""
    name: str = Field(..., description="Class name")
    danger_level: DangerLevelEnum = Field(..., description="Danger level")
    category: ObjectCategoryEnum = Field(..., description="Category")
    requires_ppe: bool = Field(..., description="PPE requirement")
    priority: int = Field(..., ge=1, le=10, description="Priority level")
    description: Optional[str] = Field(None, description="Class description")
    alert_message: Optional[str] = Field(None, description="Default alert message")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "person",
                "danger_level": "high",
                "category": "people",
                "requires_ppe": True,
                "priority": 10,
                "description": "Human worker on construction site",
                "alert_message": "Worker detected - verify PPE"
            }
        }


class ClassListResponse(BaseModel):
    """Response with list of supported classes"""
    success: bool = Field(..., description="Request success")
    total_classes: int = Field(..., description="Total number of classes")
    classes: List[ClassInfo] = Field(..., description="List of class information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_classes": 25,
                "classes": []
            }
        }


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = Field(default=False, description="Always False for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid image format",
                "detail": "Could not decode base64 image data",
                "timestamp": "2025-11-07T12:50:21Z"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    model_loaded: bool = Field(..., description="Model load status")
    timestamp: str = Field(..., description="Check timestamp")
    error: Optional[str] = Field(None, description="Error if unhealthy")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "detection",
                "model_loaded": True,
                "timestamp": "2025-11-07T12:50:21Z",
                "error": None
            }
        }
