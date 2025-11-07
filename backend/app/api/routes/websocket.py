"""
WebSocket API Routes
Handles real-time video streaming and detection for construction safety monitoring
Supports multiple concurrent connections with optimized performance
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Set, Optional
import asyncio
import cv2
import numpy as np
import base64
import json
import time
from datetime import datetime

from app.core.yolo_detector import ConstructionSafetyDetector
from app.services.detection_service import DetectionService
from app.services.alert_service import AlertService
from app.utils.image_utils import decode_base64_image, resize_image
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ws", tags=["websocket"])

# Initialize services
detector = ConstructionSafetyDetector(model_name="yolov8n.pt")
detection_service = DetectionService(detector)
alert_service = AlertService()


class ConnectionManager:
    """
    Manages WebSocket connections for multiple clients
    Handles broadcasting and individual client communication
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, dict] = {}
        self.frame_counts: Dict[str, int] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str, metadata: dict = None):
        """
        Accept and register a new WebSocket connection
        
        Args:
            websocket: WebSocket connection object
            client_id: Unique identifier for the client
            metadata: Optional metadata about the connection
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = metadata or {}
        self.frame_counts[client_id] = 0
        
        logger.info(f"Client connected: {client_id} | Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_message(client_id, {
            "type": "connection_established",
            "client_id": client_id,
            "message": "Connected to Construction Safety Detection Service",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, client_id: str):
        """
        Remove a client connection
        
        Args:
            client_id: Client identifier to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            if client_id in self.frame_counts:
                del self.frame_counts[client_id]
            
            logger.info(f"Client disconnected: {client_id} | Remaining connections: {len(self.active_connections)}")
    
    async def send_message(self, client_id: str, message: dict):
        """
        Send message to a specific client
        
        Args:
            client_id: Target client identifier
            message: Dictionary message to send
        """
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {str(e)}")
    
    async def broadcast(self, message: dict, exclude: Optional[Set[str]] = None):
        """
        Broadcast message to all connected clients
        
        Args:
            message: Dictionary message to broadcast
            exclude: Set of client IDs to exclude from broadcast
        """
        exclude = exclude or set()
        disconnected = []
        
        for client_id, connection in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {str(e)}")
                    disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
    
    def get_connection_stats(self) -> dict:
        """
        Get statistics about active connections
        
        Returns:
            Dictionary with connection statistics
        """
        return {
            "active_connections": len(self.active_connections),
            "total_frames_processed": sum(self.frame_counts.values()),
            "clients": [
                {
                    "client_id": client_id,
                    "frames_processed": self.frame_counts.get(client_id, 0),
                    "metadata": self.connection_metadata.get(client_id, {})
                }
                for client_id in self.active_connections.keys()
            ]
        }


# Initialize connection manager
manager = ConnectionManager()


@router.websocket("/detect")
async def websocket_detection_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(default=None),
    confidence: float = Query(default=0.5, ge=0.0, le=1.0),
    fps_limit: int = Query(default=10, ge=1, le=30)
):
    """
    Main WebSocket endpoint for real-time object detection
    
    Args:
        websocket: WebSocket connection
        client_id: Optional client identifier (auto-generated if not provided)
        confidence: Confidence threshold for detections (0.0 - 1.0)
        fps_limit: Maximum frames per second to process (1-30)
    
    Protocol:
        Client sends: {"frame": "base64_encoded_image", "timestamp": "ISO_timestamp"}
        Server responds: {"detections": [...], "alerts": [...], "metadata": {...}}
    """
    
    # Generate client ID if not provided
    if not client_id:
        client_id = f"client_{int(time.time() * 1000)}"
    
    try:
        # Connect client
        await manager.connect(websocket, client_id, {
            "confidence_threshold": confidence,
            "fps_limit": fps_limit,
            "connected_at": datetime.utcnow().isoformat()
        })
        
        # Frame processing control
        frame_delay = 1.0 / fps_limit
        last_process_time = 0
        
        while True:
            try:
                # Receive frame data from client
                data = await websocket.receive_text()
                frame_data = json.loads(data)
                
                # FPS limiting
                current_time = time.time()
                if current_time - last_process_time < frame_delay:
                    await asyncio.sleep(frame_delay - (current_time - last_process_time))
                
                # Extract frame
                if "frame" not in frame_data:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "No frame data provided",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
                
                # Decode frame
                frame = decode_base64_image(frame_data["frame"])
                
                if frame is None:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Failed to decode frame",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue
                
                # Resize for performance
                frame = resize_image(frame, max_width=640, max_height=480)
                
                # Process detection
                start_time = time.time()
                result = detection_service.process_frame_fast(
                    frame=frame,
                    conf_threshold=confidence
                )
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Generate alerts
                alerts = alert_service.generate_alerts(result["detections"])
                
                # Update frame count
                manager.frame_counts[client_id] = manager.frame_counts.get(client_id, 0) + 1
                
                # Prepare response
                response = {
                    "type": "detection_result",
                    "detections": result["detections"],
                    "alerts": alerts,
                    "metadata": {
                        "total_objects": len(result["detections"]),
                        "dangerous_objects": len([d for d in result["detections"] if d["danger_level"] in ["high", "critical"]]),
                        "processing_time_ms": round(processing_time, 2),
                        "frame_number": manager.frame_counts[client_id],
                        "client_timestamp": frame_data.get("timestamp"),
                        "server_timestamp": datetime.utcnow().isoformat()
                    }
                }
                
                # Send response
                await manager.send_message(client_id, response)
                
                last_process_time = time.time()
                
                # Log periodic stats (every 100 frames)
                if manager.frame_counts[client_id] % 100 == 0:
                    logger.info(f"Client {client_id}: Processed {manager.frame_counts[client_id]} frames")
                
            except json.JSONDecodeError:
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error processing frame for {client_id}: {str(e)}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected normally")
    
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {str(e)}", exc_info=True)
        manager.disconnect(client_id)


@router.websocket("/detect/stream")
async def websocket_stream_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(default=None),
    confidence: float = Query(default=0.5),
    alert_only: bool = Query(default=False)
):
    """
    Optimized streaming endpoint for high-performance detection
    Returns only essential data for minimal latency
    
    Args:
        websocket: WebSocket connection
        client_id: Optional client identifier
        confidence: Confidence threshold for detections
        alert_only: If True, only send responses when alerts are detected
    """
    
    if not client_id:
        client_id = f"stream_{int(time.time() * 1000)}"
    
    try:
        await manager.connect(websocket, client_id, {
            "mode": "stream",
            "alert_only": alert_only,
            "confidence_threshold": confidence
        })
        
        while True:
            try:
                # Receive frame
                data = await websocket.receive_text()
                frame_data = json.loads(data)
                
                # Decode and resize
                frame = decode_base64_image(frame_data["frame"])
                if frame is None:
                    continue
                
                frame = resize_image(frame, max_width=416, max_height=416)
                
                # Fast detection
                result = detection_service.process_frame_fast(frame, conf_threshold=confidence)
                
                # Check for dangerous objects
                dangerous = [d for d in result["detections"] if d["danger_level"] in ["high", "critical"]]
                
                # Skip response if alert_only and no dangers
                if alert_only and len(dangerous) == 0:
                    continue
                
                # Minimal response
                response = {
                    "type": "stream",
                    "count": len(result["detections"]),
                    "danger_count": len(dangerous),
                    "dangers": [
                        {
                            "class": d["class"],
                            "confidence": d["confidence"],
                            "bbox": d["bbox"]
                        }
                        for d in dangerous
                    ],
                    "has_alert": len(dangerous) > 0
                }
                
                await manager.send_message(client_id, response)
                
            except Exception as e:
                logger.error(f"Stream error for {client_id}: {str(e)}")
                continue
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@router.websocket("/detect/annotated")
async def websocket_annotated_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(default=None),
    confidence: float = Query(default=0.5)
):
    """
    WebSocket endpoint that returns annotated frames with bounding boxes
    Useful for displaying real-time detection overlay
    
    Args:
        websocket: WebSocket connection
        client_id: Optional client identifier
        confidence: Confidence threshold for detections
    
    Returns frames with drawn bounding boxes as base64
    """
    
    if not client_id:
        client_id = f"annotated_{int(time.time() * 1000)}"
    
    try:
        await manager.connect(websocket, client_id, {"mode": "annotated"})
        
        while True:
            try:
                # Receive frame
                data = await websocket.receive_text()
                frame_data = json.loads(data)
                
                # Decode frame
                frame = decode_base64_image(frame_data["frame"])
                if frame is None:
                    continue
                
                # Run detection
                result = detection_service.process_frame(frame, conf_threshold=confidence)
                
                # Draw detections on frame
                annotated_frame = detection_service.draw_detections(
                    frame=frame,
                    detections=result["detections"]
                )
                
                # Encode back to base64
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                annotated_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Generate alerts
                alerts = alert_service.generate_alerts(result["detections"])
                
                # Send response with annotated frame
                response = {
                    "type": "annotated_frame",
                    "frame": annotated_base64,
                    "detections": result["detections"],
                    "alerts": alerts,
                    "metadata": {
                        "total_objects": len(result["detections"]),
                        "dangerous_objects": len([d for d in result["detections"] if d["danger_level"] in ["high", "critical"]])
                    }
                }
                
                await manager.send_message(client_id, response)
                
            except Exception as e:
                logger.error(f"Annotated stream error for {client_id}: {str(e)}")
                continue
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)


@router.websocket("/monitor")
async def websocket_monitor_endpoint(websocket: WebSocket):
    """
    Monitoring endpoint for observing all active connections and system stats
    Admin/dashboard use only
    
    Broadcasts system statistics every 2 seconds
    """
    
    monitor_id = f"monitor_{int(time.time() * 1000)}"
    
    try:
        await manager.connect(websocket, monitor_id, {"role": "monitor"})
        
        while True:
            try:
                # Get connection stats
                stats = manager.get_connection_stats()
                
                # Get detection service stats
                detection_stats = detection_service.get_statistics()
                
                # Combine stats
                monitor_data = {
                    "type": "monitor_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "connections": stats,
                    "detection_stats": {
                        "total_detections": detection_stats["total_detections"],
                        "average_processing_time_ms": detection_stats["average_processing_time_ms"]
                    },
                    "alert_stats": {
                        "total_alerts": alert_service.get_total_alerts()
                    }
                }
                
                await manager.send_message(monitor_id, monitor_data)
                
                # Wait 2 seconds before next update
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Monitor error: {str(e)}")
                await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        manager.disconnect(monitor_id)


@router.websocket("/ping")
async def websocket_ping_endpoint(websocket: WebSocket):
    """
    Simple ping/pong endpoint for connection health checking
    """
    
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
            else:
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        pass


# Helper function to get connection manager stats (can be called from other modules)
def get_active_connections_count() -> int:
    """
    Get the number of active WebSocket connections
    
    Returns:
        Integer count of active connections
    """
    return len(manager.active_connections)


def get_connection_manager() -> ConnectionManager:
    """
    Get the connection manager instance
    
    Returns:
        ConnectionManager instance
    """
    return manager
