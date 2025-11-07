"""
Detection API Routes
Handles all detection-related endpoints for construction safety monitoring
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json

from app.core.yolo_detector import ConstructionSafetyDetector
from app.services.detection_service import DetectionService
from app.services.alert_service import AlertService
from app.schemas.detection import (
    DetectionRequest,
    DetectionResponse,
    BatchDetectionResponse,
    ImageDetectionResponse,
    DetectionStatistics
)
from app.utils.image_utils import (
    decode_base64_image,
    encode_image_to_base64,
    validate_image,
    resize_image
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/detection", tags=["detection"])

# Initialize services
detector = ConstructionSafetyDetector(model_name="yolov8n.pt")
detection_service = DetectionService(detector)
alert_service = AlertService()


@router.post("/detect", response_model=DetectionResponse)
async def detect_from_base64(request: DetectionRequest):
    """
    Detect objects and safety hazards from a base64 encoded image
    
    Args:
        request: DetectionRequest containing base64 image and optional parameters
        
    Returns:
        DetectionResponse with detections, alerts, and metadata
    """
    try:
        logger.info(f"Processing detection request with confidence threshold: {request.confidence_threshold}")
        
        # Decode base64 image
        frame = decode_base64_image(request.image_base64)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Validate image
        if not validate_image(frame):
            raise HTTPException(status_code=400, detail="Image validation failed")
        
        # Resize if needed for performance
        if request.resize:
            frame = resize_image(frame, max_width=640, max_height=480)
        
        # Run detection
        result = detection_service.process_frame(
            frame=frame,
            conf_threshold=request.confidence_threshold,
            include_landmarks=request.include_landmarks
        )
        
        # Generate alerts
        alerts = alert_service.generate_alerts(
            detections=result["detections"],
            severity_filter=request.alert_severity
        )
        
        # Prepare response
        response = DetectionResponse(
            success=True,
            detections=result["detections"],
            alerts=alerts,
            total_objects=len(result["detections"]),
            dangerous_objects=len([d for d in result["detections"] if d["danger_level"] in ["high", "critical"]]),
            processing_time_ms=result["processing_time_ms"],
            frame_dimensions=result["frame_dimensions"]
        )
        
        logger.info(f"Detection completed: {response.total_objects} objects found, {response.dangerous_objects} dangerous")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in detect_from_base64: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/detect/upload", response_model=ImageDetectionResponse)
async def detect_from_upload(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    draw_boxes: bool = Form(True),
    include_labels: bool = Form(True)
):
    """
    Detect objects from an uploaded image file
    Supports: JPG, JPEG, PNG, BMP
    
    Args:
        file: Uploaded image file
        confidence_threshold: Minimum confidence for detections (0.0 - 1.0)
        draw_boxes: Whether to return annotated image with bounding boxes
        include_labels: Whether to include labels on annotated image
        
    Returns:
        ImageDetectionResponse with detections, alerts, and optionally annotated image
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/bmp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported: JPG, PNG, BMP"
            )
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Run detection
        result = detection_service.process_frame(
            frame=frame,
            conf_threshold=confidence_threshold
        )
        
        # Generate alerts
        alerts = alert_service.generate_alerts(result["detections"])
        
        # Draw bounding boxes if requested
        annotated_image_base64 = None
        if draw_boxes:
            annotated_frame = detection_service.draw_detections(
                frame=frame,
                detections=result["detections"],
                include_labels=include_labels
            )
            annotated_image_base64 = encode_image_to_base64(annotated_frame)
        
        response = ImageDetectionResponse(
            success=True,
            filename=file.filename,
            detections=result["detections"],
            alerts=alerts,
            total_objects=len(result["detections"]),
            dangerous_objects=len([d for d in result["detections"] if d["danger_level"] in ["high", "critical"]]),
            processing_time_ms=result["processing_time_ms"],
            annotated_image_base64=annotated_image_base64,
            frame_dimensions=result["frame_dimensions"]
        )
        
        logger.info(f"Upload detection completed: {file.filename} - {response.total_objects} objects")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_from_upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload detection failed: {str(e)}")


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Batch detection on multiple uploaded images
    Useful for analyzing multiple construction site images at once
    
    Args:
        files: List of uploaded image files (max 10)
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        BatchDetectionResponse with results for each image
    """
    try:
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images allowed per batch request"
            )
        
        logger.info(f"Processing batch detection for {len(files)} files")
        
        results = []
        total_detections = 0
        total_dangerous = 0
        
        for idx, file in enumerate(files):
            try:
                # Read image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning(f"Failed to decode image: {file.filename}")
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Failed to decode image"
                    })
                    continue
                
                # Run detection
                result = detection_service.process_frame(
                    frame=frame,
                    conf_threshold=confidence_threshold
                )
                
                # Generate alerts
                alerts = alert_service.generate_alerts(result["detections"])
                
                dangerous_count = len([d for d in result["detections"] if d["danger_level"] in ["high", "critical"]])
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "detections": result["detections"],
                    "alerts": alerts,
                    "total_objects": len(result["detections"]),
                    "dangerous_objects": dangerous_count,
                    "processing_time_ms": result["processing_time_ms"]
                })
                
                total_detections += len(result["detections"])
                total_dangerous += dangerous_count
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        response = BatchDetectionResponse(
            success=True,
            total_images=len(files),
            successful_detections=len([r for r in results if r["success"]]),
            failed_detections=len([r for r in results if not r["success"]]),
            results=results,
            total_objects_detected=total_detections,
            total_dangerous_objects=total_dangerous
        )
        
        logger.info(f"Batch detection completed: {response.successful_detections}/{response.total_images} successful")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_batch: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@router.get("/stats", response_model=DetectionStatistics)
async def get_detection_statistics():
    """
    Get detection statistics and system information
    
    Returns:
        DetectionStatistics with model info and performance metrics
    """
    try:
        stats = detection_service.get_statistics()
        
        return DetectionStatistics(
            model_name=stats["model_name"],
            total_detections=stats["total_detections"],
            total_alerts_generated=alert_service.get_total_alerts(),
            average_processing_time_ms=stats["average_processing_time_ms"],
            supported_classes=stats["supported_classes"],
            danger_classes=stats["danger_classes"]
        )
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/detect/annotate")
async def detect_and_annotate(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    box_thickness: int = Form(2),
    font_scale: float = Form(0.5)
):
    """
    Detect objects and return annotated image as downloadable file
    
    Args:
        file: Uploaded image file
        confidence_threshold: Minimum confidence for detections
        box_thickness: Thickness of bounding box lines
        font_scale: Scale of text labels
        
    Returns:
        Annotated image as StreamingResponse (downloadable)
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Run detection
        result = detection_service.process_frame(
            frame=frame,
            conf_threshold=confidence_threshold
        )
        
        # Draw detections
        annotated_frame = detection_service.draw_detections(
            frame=frame,
            detections=result["detections"],
            box_thickness=box_thickness,
            font_scale=font_scale
        )
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        io_buf = io.BytesIO(buffer)
        
        return StreamingResponse(
            io_buf,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename=annotated_{file.filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_and_annotate: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for detection service
    
    Returns:
        Service health status
    """
    try:
        # Check if model is loaded
        model_status = detector.is_model_loaded()
        
        return {
            "status": "healthy" if model_status else "degraded",
            "service": "detection",
            "model_loaded": model_status,
            "timestamp": "2025-11-07 11:01:28"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "detection",
            "error": str(e)
        }


@router.delete("/cache/clear")
async def clear_detection_cache():
    """
    Clear detection cache and statistics
    Admin endpoint for resetting detection history
    
    Returns:
        Success message
    """
    try:
        detection_service.clear_cache()
        alert_service.clear_cache()
        
        logger.info("Detection cache cleared successfully")
        
        return {
            "success": True,
            "message": "Detection cache and statistics cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/classes")
async def get_supported_classes():
    """
    Get list of all supported detection classes and their danger levels
    
    Returns:
        Dictionary of classes with their properties
    """
    try:
        classes = detection_service.get_all_classes()
        
        return {
            "success": True,
            "total_classes": len(classes),
            "classes": classes
        }
    except Exception as e:
        logger.error(f"Error getting classes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get classes: {str(e)}")


@router.post("/detect/stream-frame")
async def detect_stream_frame(request: DetectionRequest):
    """
    Optimized endpoint for video stream frame detection
    Lower latency, optimized for real-time streaming
    
    Args:
        request: DetectionRequest with base64 frame
        
    Returns:
        Lightweight detection response
    """
    try:
        # Decode image
        frame = decode_base64_image(request.image_base64)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Resize for performance (streaming optimization)
        frame = resize_image(frame, max_width=416, max_height=416)
        
        # Quick detection with lower confidence for speed
        result = detection_service.process_frame_fast(
            frame=frame,
            conf_threshold=max(request.confidence_threshold, 0.3)
        )
        
        # Generate only critical alerts
        alerts = alert_service.generate_critical_alerts(result["detections"])
        
        # Lightweight response
        return {
            "success": True,
            "detections": result["detections"],
            "alerts": alerts,
            "count": len(result["detections"]),
            "dangerous_count": len([d for d in result["detections"] if d["danger_level"] == "high"]),
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Stream frame detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
