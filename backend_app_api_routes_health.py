"""
Health Check API Routes
Comprehensive health monitoring endpoints for the construction safety detection system
Author: A-P-U-R-B-O
Created: 2025-11-07 13:12:40 UTC
"""

from fastapi import APIRouter, status, Response
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import time
import psutil
import platform
import sys
from datetime import datetime, timedelta

from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ============================================================================
# Global Health State
# ============================================================================

class HealthMetrics:
    """Track health metrics over time"""
    
    def __init__(self):
        self.startup_time = time.time()
        self.last_health_check = None
        self.health_check_count = 0
        self.failed_checks = 0
        self.last_error = None
        self.consecutive_failures = 0
    
    def record_success(self):
        """Record successful health check"""
        self.last_health_check = datetime.utcnow()
        self.health_check_count += 1
        self.consecutive_failures = 0
    
    def record_failure(self, error: str):
        """Record failed health check"""
        self.last_health_check = datetime.utcnow()
        self.health_check_count += 1
        self.failed_checks += 1
        self.consecutive_failures += 1
        self.last_error = error
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds"""
        return time.time() - self.startup_time


# Initialize metrics
health_metrics = HealthMetrics()


# ============================================================================
# Basic Health Check
# ============================================================================

@router.get(
    "/",
    summary="Basic Health Check",
    description="Simple health check endpoint for load balancers",
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint
    
    Returns a simple status indicating the service is alive.
    This endpoint is designed for load balancers and monitoring tools.
    
    Returns:
        Dict with status and timestamp
    """
    try:
        health_metrics.record_success()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "user": "A-P-U-R-B-O"
        }
    
    except Exception as e:
        health_metrics.record_failure(str(e))
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        )


# ============================================================================
# Liveness Probe
# ============================================================================

@router.get(
    "/live",
    summary="Liveness Probe",
    description="Kubernetes/Docker liveness probe endpoint",
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def liveness_probe() -> Dict[str, Any]:
    """
    Liveness probe for container orchestration
    
    Indicates if the application is running and can accept requests.
    Used by Kubernetes to determine if a pod should be restarted.
    
    Returns:
        Dict with liveness status
    """
    try:
        return {
            "alive": True,
            "status": "running",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "uptime_seconds": round(health_metrics.get_uptime(), 2)
        }
    
    except Exception as e:
        logger.error(f"Liveness probe failed: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "alive": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        )


# ============================================================================
# Readiness Probe
# ============================================================================

@router.get(
    "/ready",
    summary="Readiness Probe",
    description="Kubernetes/Docker readiness probe endpoint",
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def readiness_probe() -> Dict[str, Any]:
    """
    Readiness probe for container orchestration
    
    Indicates if the application is ready to accept traffic.
    Used by Kubernetes to determine if a pod should receive requests.
    
    Checks:
    - Services are initialized
    - Model is loaded
    - System resources are available
    
    Returns:
        Dict with readiness status
    """
    try:
        # Import app state
        from app.main import app_state
        
        # Check if services are initialized
        services_ready = all([
            app_state.detector is not None,
            app_state.detection_service is not None,
            app_state.alert_service is not None,
            app_state.danger_classifier is not None
        ])
        
        # Check model status
        model_loaded = app_state.detector.is_model_loaded() if app_state.detector else False
        
        # Check system resources
        memory = psutil.virtual_memory()
        memory_available = memory.available > 100 * 1024 * 1024  # At least 100MB available
        
        # Determine readiness
        is_ready = services_ready and model_loaded and memory_available
        
        response = {
            "ready": is_ready,
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "checks": {
                "services_initialized": services_ready,
                "model_loaded": model_loaded,
                "memory_available": memory_available
            },
            "details": {
                "detector": app_state.detector is not None,
                "detection_service": app_state.detection_service is not None,
                "alert_service": app_state.alert_service is not None,
                "danger_classifier": app_state.danger_classifier is not None,
                "memory_available_mb": round(memory.available / (1024 * 1024), 2)
            }
        }
        
        if is_ready:
            return response
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response
            )
    
    except Exception as e:
        logger.error(f"Readiness probe failed: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        )


# ============================================================================
# Detailed Health Check
# ============================================================================

@router.get(
    "/detailed",
    summary="Detailed Health Check",
    description="Comprehensive health check with system metrics",
    status_code=status.HTTP_200_OK,
    tags=["Health"]
)
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with comprehensive system information
    
    Provides extensive information about:
    - Application status
    - Service status
    - System resources
    - Performance metrics
    - Health history
    
    Returns:
        Dict with detailed health information
    """
    try:
        from app.main import app_state
        
        # Application info
        uptime = health_metrics.get_uptime()
        
        application_info = {
            "name": "Construction Safety AI Detection System",
            "version": "1.0.0",
            "author": "A-P-U-R-B-O",
            "status": "operational",
            "uptime_seconds": round(uptime, 2),
            "uptime_formatted": format_uptime(uptime),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "startup_time": datetime.fromtimestamp(health_metrics.startup_time).strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        # Service status
        service_status = {
            "detector": {
                "initialized": app_state.detector is not None,
                "model_loaded": app_state.detector.is_model_loaded() if app_state.detector else False,
                "model_name": app_state.detector.model_name if app_state.detector else None,
                "device": app_state.detector.device if app_state.detector else None
            },
            "detection_service": {
                "initialized": app_state.detection_service is not None,
                "cache_enabled": app_state.detection_service.enable_cache if app_state.detection_service else False
            },
            "alert_service": {
                "initialized": app_state.alert_service is not None,
                "history_enabled": app_state.alert_service.enable_history if app_state.alert_service else False
            },
            "danger_classifier": {
                "initialized": app_state.danger_classifier is not None
            }
        }
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_resources = {
            "cpu": {
                "usage_percent": round(cpu_percent, 2),
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total_mb": round(memory.total / (1024 * 1024), 2),
                "available_mb": round(memory.available / (1024 * 1024), 2),
                "used_mb": round(memory.used / (1024 * 1024), 2),
                "percent": round(memory.percent, 2)
            },
            "disk": {
                "total_gb": round(disk.total / (1024 * 1024 * 1024), 2),
                "used_gb": round(disk.used / (1024 * 1024 * 1024), 2),
                "free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
                "percent": round(disk.percent, 2)
            }
        }
        
        # Platform info
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_implementation": platform.python_implementation()
        }
        
        # Performance metrics
        performance_metrics = {}
        
        if app_state.detection_service:
            detection_stats = app_state.detection_service.get_statistics()
            performance_metrics["detection"] = {
                "total_frames_processed": detection_stats["total_frames_processed"],
                "total_detections": detection_stats["total_detections"],
                "average_detections_per_frame": round(detection_stats["average_detections_per_frame"], 2)
            }
            
            if "detector_stats" in detection_stats:
                performance_metrics["detector"] = {
                    "average_processing_time_ms": round(
                        detection_stats["detector_stats"]["average_processing_time_ms"], 2
                    )
                }
        
        if app_state.alert_service:
            alert_stats = app_state.alert_service.get_alert_statistics()
            performance_metrics["alerts"] = {
                "total_alerts": alert_stats.get("total_alerts", 0),
                "severity_distribution": alert_stats.get("severity_distribution", {})
            }
        
        # Health check history
        health_history = {
            "total_checks": health_metrics.health_check_count,
            "failed_checks": health_metrics.failed_checks,
            "consecutive_failures": health_metrics.consecutive_failures,
            "success_rate": round(
                (health_metrics.health_check_count - health_metrics.failed_checks) / 
                health_metrics.health_check_count * 100, 2
            ) if health_metrics.health_check_count > 0 else 100.0,
            "last_check": health_metrics.last_health_check.strftime("%Y-%m-%d %H:%M:%S UTC") 
                if health_metrics.last_health_check else None,
            "last_error": health_metrics.last_error
        }
        
        # Determine overall health status
        health_score = calculate_health_score(
            service_status,
            system_resources,
            health_history
        )
        
        overall_status = determine_health_status(health_score)
        
        health_metrics.record_success()
        
        return {
            "status": overall_status,
            "health_score": health_score,
            "application": application_info,
            "services": service_status,
            "system_resources": system_resources,
            "platform": platform_info,
            "performance": performance_metrics,
            "health_history": health_history,
            "recommendations": generate_health_recommendations(
                system_resources,
                service_status,
                health_history
            )
        }
    
    except Exception as e:
        health_metrics.record_failure(str(e))
        logger.error(f"Detailed health check failed: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        )


# ============================================================================
# Service-Specific Health Checks
# ============================================================================

@router.get(
    "/detection",
    summary="Detection Service Health",
    description="Health check for detection service",
    tags=["Health"]
)
async def detection_service_health() -> Dict[str, Any]:
    """
    Health check specifically for detection service
    
    Returns:
        Detection service health status
    """
    try:
        from app.main import app_state
        
        if not app_state.detection_service:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_initialized",
                    "service": "detection",
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                }
            )
        
        stats = app_state.detection_service.get_statistics()
        
        return {
            "status": "healthy",
            "service": "detection",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(f"Detection service health check failed: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "detection",
                "error": str(e),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        )


@router.get(
    "/alert",
    summary="Alert Service Health",
    description="Health check for alert service",
    tags=["Health"]
)
async def alert_service_health() -> Dict[str, Any]:
    """
    Health check specifically for alert service
    
    Returns:
        Alert service health status
    """
    try:
        from app.main import app_state
        
        if not app_state.alert_service:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_initialized",
                    "service": "alert",
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                }
            )
        
        stats = app_state.alert_service.get_alert_statistics()
        
        return {
            "status": "healthy",
            "service": "alert",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(f"Alert service health check failed: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "alert",
                "error": str(e),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        )


@router.get(
    "/model",
    summary="Model Health Check",
    description="Health check for YOLO model",
    tags=["Health"]
)
async def model_health() -> Dict[str, Any]:
    """
    Health check specifically for YOLO model
    
    Returns:
        Model health status
    """
    try:
        from app.main import app_state
        
        if not app_state.detector:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_initialized",
                    "service": "model",
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                }
            )
        
        stats = app_state.detector.get_statistics()
        model_loaded = app_state.detector.is_model_loaded()
        
        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "service": "model",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "model_loaded": model_loaded,
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(f"Model health check failed: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "service": "model",
                "error": str(e),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        )


# ============================================================================
# System Metrics
# ============================================================================

@router.get(
    "/metrics",
    summary="System Metrics",
    description="Get system performance metrics",
    tags=["Health"]
)
async def system_metrics() -> Dict[str, Any]:
    """
    Get detailed system performance metrics
    
    Returns:
        System metrics including CPU, memory, disk, and network
    """
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        # Process info
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "cpu": {
                "usage_percent": round(sum(cpu_percent) / len(cpu_percent), 2),
                "per_core": [round(p, 2) for p in cpu_percent],
                "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total_mb": round(memory.total / (1024 * 1024), 2),
                "available_mb": round(memory.available / (1024 * 1024), 2),
                "used_mb": round(memory.used / (1024 * 1024), 2),
                "percent": round(memory.percent, 2),
                "swap_total_mb": round(swap.total / (1024 * 1024), 2),
                "swap_used_mb": round(swap.used / (1024 * 1024), 2),
                "swap_percent": round(swap.percent, 2)
            },
            "disk": {
                "total_gb": round(disk.total / (1024 * 1024 * 1024), 2),
                "used_gb": round(disk.used / (1024 * 1024 * 1024), 2),
                "free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
                "percent": round(disk.percent, 2),
                "read_mb": round(disk_io.read_bytes / (1024 * 1024), 2) if disk_io else None,
                "write_mb": round(disk_io.write_bytes / (1024 * 1024), 2) if disk_io else None
            },
            "network": {
                "bytes_sent_mb": round(net_io.bytes_sent / (1024 * 1024), 2),
                "bytes_recv_mb": round(net_io.bytes_recv / (1024 * 1024), 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            },
            "process": {
                "memory_rss_mb": round(process_memory.rss / (1024 * 1024), 2),
                "memory_vms_mb": round(process_memory.vms / (1024 * 1024), 2),
                "cpu_percent": round(process.cpu_percent(interval=0.1), 2),
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": str(e),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            }
        )


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


def calculate_health_score(
    service_status: Dict[str, Any],
    system_resources: Dict[str, Any],
    health_history: Dict[str, Any]
) -> float:
    """
    Calculate overall health score (0-100)
    
    Args:
        service_status: Service status information
        system_resources: System resource metrics
        health_history: Health check history
        
    Returns:
        Health score between 0 and 100
    """
    score = 100.0
    
    # Service initialization (-20 points per uninitialized service)
    for service, status in service_status.items():
        if not status.get("initialized", False):
            score -= 20
    
    # Model loading (-15 points if not loaded)
    if not service_status.get("detector", {}).get("model_loaded", False):
        score -= 15
    
    # CPU usage (-10 points if > 90%)
    cpu_percent = system_resources.get("cpu", {}).get("usage_percent", 0)
    if cpu_percent > 90:
        score -= 10
    elif cpu_percent > 75:
        score -= 5
    
    # Memory usage (-15 points if > 90%)
    memory_percent = system_resources.get("memory", {}).get("percent", 0)
    if memory_percent > 90:
        score -= 15
    elif memory_percent > 75:
        score -= 7
    
    # Disk usage (-10 points if > 90%)
    disk_percent = system_resources.get("disk", {}).get("percent", 0)
    if disk_percent > 90:
        score -= 10
    elif disk_percent > 80:
        score -= 5
    
    # Health check failures (-5 points per consecutive failure)
    consecutive_failures = health_history.get("consecutive_failures", 0)
    score -= min(consecutive_failures * 5, 25)  # Max -25 points
    
    return max(0.0, min(100.0, score))


def determine_health_status(health_score: float) -> str:
    """
    Determine health status from score
    
    Args:
        health_score: Health score (0-100)
        
    Returns:
        Status string
    """
    if health_score >= 90:
        return "excellent"
    elif health_score >= 75:
        return "good"
    elif health_score >= 60:
        return "fair"
    elif health_score >= 40:
        return "degraded"
    else:
        return "critical"


def generate_health_recommendations(
    system_resources: Dict[str, Any],
    service_status: Dict[str, Any],
    health_history: Dict[str, Any]
) -> List[str]:
    """
    Generate health recommendations based on metrics
    
    Args:
        system_resources: System resource metrics
        service_status: Service status information
        health_history: Health check history
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # CPU recommendations
    cpu_percent = system_resources.get("cpu", {}).get("usage_percent", 0)
    if cpu_percent > 90:
        recommendations.append("⚠️ CPU usage is very high (>90%). Consider scaling or optimizing workload.")
    elif cpu_percent > 75:
        recommendations.append("ℹ️ CPU usage is elevated (>75%). Monitor for sustained high usage.")
    
    # Memory recommendations
    memory_percent = system_resources.get("memory", {}).get("percent", 0)
    if memory_percent > 90:
        recommendations.append("⚠️ Memory usage is critical (>90%). Increase memory or restart service.")
    elif memory_percent > 75:
        recommendations.append("ℹ️ Memory usage is high (>75%). Consider increasing available memory.")
    
    # Disk recommendations
    disk_percent = system_resources.get("disk", {}).get("percent", 0)
    if disk_percent > 90:
        recommendations.append("⚠️ Disk space is critically low (>90%). Clean up logs or increase storage.")
    elif disk_percent > 80:
        recommendations.append("ℹ️ Disk space is running low (>80%). Plan for cleanup or expansion.")
    
    # Service recommendations
    for service_name, status in service_status.items():
        if not status.get("initialized", False):
            recommendations.append(f"❌ {service_name} is not initialized. Check startup logs.")
    
    # Model recommendations
    if not service_status.get("detector", {}).get("model_loaded", False):
        recommendations.append("❌ Detection model is not loaded. Verify model file exists.")
    
    # Health check recommendations
    if health_history.get("consecutive_failures", 0) > 3:
        recommendations.append("⚠️ Multiple consecutive health check failures detected. Investigate immediately.")
    
    if not recommendations:
        recommendations.append("✅ All systems operating normally. No recommendations at this time.")
    
    return recommendations