"""
Main FastAPI Application Entry Point
Construction Safety AI Detection System
Author: A-P-U-R-B-O
Created: 2025-11-07 13:09:37 UTC
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
from typing import Dict, Any
import os

# Import routers
from app.api.routes import detection, websocket, health

# Import core components
from app.core.yolo_detector import ConstructionSafetyDetector
from app.core.model_config import get_config
from app.services.detection_service import DetectionService
from app.services.alert_service import AlertService
from app.services.danger_classifier import DangerClassifier
from app.utils.logger import get_logger, setup_logging

# Initialize logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs"),
    enable_json=os.getenv("ENABLE_JSON_LOGS", "false").lower() == "true"
)

logger = get_logger(__name__)


# ============================================================================
# Application State Management
# ============================================================================

class AppState:
    """
    Global application state container
    """
    def __init__(self):
        self.detector: ConstructionSafetyDetector = None
        self.detection_service: DetectionService = None
        self.alert_service: AlertService = None
        self.danger_classifier: DangerClassifier = None
        self.config = None
        self.startup_time = None
        self.request_count = 0
        self.total_detections = 0
        self.total_alerts = 0


# Global app state
app_state = AppState()


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    
    Args:
        app: FastAPI application instance
    """
    # ==================== STARTUP ====================
    logger.info("="*70)
    logger.info("🚀 Starting Construction Safety AI Detection System")
    logger.info("="*70)
    logger.info(f"👤 User: A-P-U-R-B-O")
    logger.info(f"🕐 Startup Time: 2025-11-07 13:09:37 UTC")
    logger.info(f"🐍 Python Environment: {os.getenv('PYTHON_ENV', 'production')}")
    logger.info(f"📊 Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    
    try:
        # Load configuration
        logger.info("📋 Loading configuration...")
        app_state.config = get_config()
        logger.info(f"✅ Configuration loaded: {app_state.config.get_config_summary()['total_classes']} classes configured")
        
        # Initialize YOLO detector
        logger.info("🤖 Initializing YOLO detector...")
        model_name = os.getenv("MODEL_NAME", "yolov8n.pt")
        device = os.getenv("DEVICE", "cpu")
        
        app_state.detector = ConstructionSafetyDetector(
            model_name=model_name,
            device=device
        )
        logger.info(f"✅ YOLO detector initialized: {model_name} on {device}")
        
        # Initialize services
        logger.info("🔧 Initializing services...")
        
        app_state.detection_service = DetectionService(
            detector=app_state.detector,
            enable_cache=True,
            cache_size=1000
        )
        logger.info("✅ Detection service initialized")
        
        app_state.alert_service = AlertService(
            enable_history=True,
            cooldown_seconds=int(os.getenv("ALERT_COOLDOWN", "5")),
            max_alerts_per_frame=int(os.getenv("MAX_ALERTS_PER_FRAME", "5"))
        )
        logger.info("✅ Alert service initialized")
        
        app_state.danger_classifier = DangerClassifier(
            history_size=100
        )
        logger.info("✅ Danger classifier initialized")
        
        # Store startup time
        app_state.startup_time = time.time()
        
        logger.info("="*70)
        logger.info("✨ Application startup complete!")
        logger.info("📡 Server is ready to accept requests")
        logger.info("="*70)
        
    except Exception as e:
        logger.critical(f"❌ Failed to start application: {str(e)}", exc_info=True)
        raise
    
    # Yield control to the application
    yield
    
    # ==================== SHUTDOWN ====================
    logger.info("="*70)
    logger.info("🛑 Shutting down Construction Safety AI Detection System")
    logger.info("="*70)
    
    try:
        # Log final statistics
        if app_state.detection_service:
            stats = app_state.detection_service.get_statistics()
            logger.info(f"📊 Total frames processed: {stats['total_frames_processed']}")
            logger.info(f"📊 Total detections: {stats['total_detections']}")
        
        if app_state.alert_service:
            alert_stats = app_state.alert_service.get_alert_statistics()
            logger.info(f"📊 Total alerts generated: {alert_stats.get('total_alerts', 0)}")
        
        uptime = time.time() - app_state.startup_time
        logger.info(f"⏱️  Total uptime: {uptime:.2f} seconds")
        logger.info(f"📈 Total requests: {app_state.request_count}")
        
        # Cleanup
        logger.info("🧹 Cleaning up resources...")
        
        if app_state.detection_service:
            app_state.detection_service.clear_cache()
        
        if app_state.alert_service:
            app_state.alert_service.clear_cache()
        
        if app_state.danger_classifier:
            app_state.danger_classifier.clear_history()
        
        logger.info("✅ Cleanup complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {str(e)}", exc_info=True)
    
    logger.info("="*70)
    logger.info("👋 Shutdown complete. Goodbye!")
    logger.info("="*70)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Construction Safety AI Detection System",
    description="""
    🏗️ **Advanced AI-powered safety detection system for construction sites**
    
    This API provides real-time object detection, danger assessment, and safety alerts
    for construction site monitoring using YOLOv8 computer vision models.
    
    ## Features
    
    * 🎯 **Real-time Object Detection** - Detect workers, vehicles, equipment, and hazards
    * ⚠️ **Intelligent Alert System** - Context-aware safety alerts with severity levels
    * 📊 **Danger Assessment** - Multi-dimensional risk analysis and scoring
    * 🎥 **Video Stream Processing** - WebSocket support for live camera feeds
    * 📈 **Statistics & Analytics** - Comprehensive detection and alert metrics
    * 🔧 **Configurable Rules** - Custom danger classifications and alert rules
    
    ## Author
    
    * **Developer**: A-P-U-R-B-O
    * **Created**: 2025-11-07 13:09:37 UTC
    * **Version**: 1.0.0
    
    ## Technology Stack
    
    * **Framework**: FastAPI
    * **AI Model**: YOLOv8 (Ultralytics)
    * **Image Processing**: OpenCV, PIL
    * **Real-time Communication**: WebSockets
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "A-P-U-R-B-O",
        "email": "contact@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)


# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Processing-Time"]
)

# GZip Compression Middleware
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000  # Only compress responses larger than 1KB
)


# ============================================================================
# Custom Middleware
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time header to all responses
    """
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000  # Convert to ms
    response.headers["X-Processing-Time"] = f"{process_time:.2f}ms"
    
    return response


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Log all incoming requests
    """
    app_state.request_count += 1
    
    logger.info(
        f"📨 Request: {request.method} {request.url.path} | "
        f"Client: {request.client.host} | "
        f"User-Agent: {request.headers.get('user-agent', 'Unknown')}"
    )
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"✅ Response: {response.status_code} | "
            f"Time: {process_time:.2f}ms | "
            f"Path: {request.url.path}"
        )
        
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        
        logger.error(
            f"❌ Request failed: {request.method} {request.url.path} | "
            f"Error: {str(e)} | "
            f"Time: {process_time:.2f}ms",
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(e),
                "path": request.url.path
            }
        )


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors
    """
    logger.warning(f"⚠️ Validation error: {exc.errors()}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "detail": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions
    """
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


# ============================================================================
# Include Routers
# ============================================================================

# Detection routes
app.include_router(
    detection.router,
    tags=["Detection"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

# WebSocket routes
app.include_router(
    websocket.router,
    tags=["WebSocket"],
    responses={
        404: {"description": "Not found"}
    }
)

# Health check routes
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
    responses={
        200: {"description": "Service healthy"}
    }
)


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get(
    "/",
    summary="Root Endpoint",
    description="Redirect to API documentation",
    response_class=RedirectResponse,
    status_code=status.HTTP_307_TEMPORARY_REDIRECT
)
async def root():
    """
    Root endpoint - redirects to API documentation
    """
    return RedirectResponse(url="/docs")


@app.get(
    "/info",
    summary="System Information",
    description="Get system and application information",
    response_model=Dict[str, Any],
    tags=["System"]
)
async def system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information
    
    Returns:
        Dictionary with system information
    """
    uptime = time.time() - app_state.startup_time if app_state.startup_time else 0
    
    info = {
        "application": {
            "name": "Construction Safety AI Detection System",
            "version": "1.0.0",
            "author": "A-P-U-R-B-O",
            "created": "2025-11-07 13:09:37 UTC",
            "status": "operational"
        },
        "system": {
            "uptime_seconds": round(uptime, 2),
            "uptime_formatted": format_uptime(uptime),
            "request_count": app_state.request_count,
            "python_version": os.sys.version,
            "environment": os.getenv("PYTHON_ENV", "production")
        },
        "model": {
            "name": os.getenv("MODEL_NAME", "yolov8n.pt"),
            "device": os.getenv("DEVICE", "cpu"),
            "loaded": app_state.detector is not None
        },
        "services": {
            "detection_service": app_state.detection_service is not None,
            "alert_service": app_state.alert_service is not None,
            "danger_classifier": app_state.danger_classifier is not None
        },
        "statistics": {}
    }
    
    # Add statistics if available
    if app_state.detection_service:
        stats = app_state.detection_service.get_statistics()
        info["statistics"]["detection"] = {
            "total_frames_processed": stats["total_frames_processed"],
            "total_detections": stats["total_detections"],
            "average_detections_per_frame": stats["average_detections_per_frame"]
        }
    
    if app_state.alert_service:
        alert_stats = app_state.alert_service.get_alert_statistics()
        info["statistics"]["alerts"] = {
            "total_alerts": alert_stats.get("total_alerts", 0),
            "severity_distribution": alert_stats.get("severity_distribution", {})
        }
    
    if app_state.danger_classifier:
        classifier_stats = app_state.danger_classifier.get_statistics()
        info["statistics"]["danger_assessment"] = classifier_stats
    
    return info


@app.get(
    "/status",
    summary="Service Status",
    description="Get current service status",
    tags=["System"]
)
async def service_status() -> Dict[str, Any]:
    """
    Get current service status
    
    Returns:
        Service status information
    """
    return {
        "status": "healthy",
        "timestamp": "2025-11-07 13:09:37 UTC",
        "user": "A-P-U-R-B-O",
        "services": {
            "detector": "operational" if app_state.detector else "not_initialized",
            "detection_service": "operational" if app_state.detection_service else "not_initialized",
            "alert_service": "operational" if app_state.alert_service else "not_initialized",
            "danger_classifier": "operational" if app_state.danger_classifier else "not_initialized"
        },
        "uptime_seconds": round(time.time() - app_state.startup_time, 2) if app_state.startup_time else 0
    }


@app.post(
    "/reset",
    summary="Reset System",
    description="Reset caches and statistics (admin only)",
    tags=["System"]
)
async def reset_system() -> Dict[str, Any]:
    """
    Reset system caches and statistics
    
    Returns:
        Reset confirmation
    """
    logger.warning("🔄 System reset requested by A-P-U-R-B-O")
    
    try:
        if app_state.detection_service:
            app_state.detection_service.clear_cache()
            logger.info("✅ Detection service cache cleared")
        
        if app_state.alert_service:
            app_state.alert_service.clear_cache()
            logger.info("✅ Alert service cache cleared")
        
        if app_state.danger_classifier:
            app_state.danger_classifier.clear_history()
            logger.info("✅ Danger classifier history cleared")
        
        # Reset counters
        app_state.total_detections = 0
        app_state.total_alerts = 0
        
        return {
            "success": True,
            "message": "System reset successfully",
            "timestamp": "2025-11-07 13:09:37 UTC",
            "reset_by": "A-P-U-R-B-O"
        }
        
    except Exception as e:
        logger.error(f"❌ System reset failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# Utility Functions
# ============================================================================

def format_uptime(seconds: float) -> str:
    """
    Format uptime in human-readable format
    
    Args:
        seconds: Uptime in seconds
        
    Returns:
        Formatted uptime string
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*70)
    logger.info("🚀 Starting server directly via main.py")
    logger.info("="*70)
    
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", "1")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )