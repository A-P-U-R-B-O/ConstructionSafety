"""
Application Configuration Module
Centralized configuration management for the construction safety detection system
Author: A-P-U-R-B-O
Created: 2025-11-07 13:18:18 UTC
"""

from pydantic import BaseSettings, Field, validator, HttpUrl
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
from functools import lru_cache

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Base Configuration
# ============================================================================

class Settings(BaseSettings):
    """
    Main application settings
    Loads configuration from environment variables and .env file
    """
    
    # ========================================
    # Application Settings
    # ========================================
    
    APP_NAME: str = Field(
        default="Construction Safety AI Detection System",
        description="Application name"
    )
    
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    APP_AUTHOR: str = Field(
        default="A-P-U-R-B-O",
        description="Application author"
    )
    
    APP_DESCRIPTION: str = Field(
        default="AI-powered safety detection system for construction sites",
        description="Application description"
    )
    
    ENVIRONMENT: str = Field(
        default="production",
        description="Environment (development, staging, production)"
    )
    
    DEBUG: bool = Field(
        default=False,
        description="Debug mode"
    )
    
    # ========================================
    # Server Settings
    # ========================================
    
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    
    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    
    WORKERS: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of worker processes"
    )
    
    RELOAD: bool = Field(
        default=False,
        description="Auto-reload on code changes"
    )
    
    BASE_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL for the application"
    )
    
    # ========================================
    # CORS Settings
    # ========================================
    
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    CORS_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS"
    )
    
    CORS_METHODS: List[str] = Field(
        default=["*"],
        description="Allowed CORS methods"
    )
    
    CORS_HEADERS: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    
    # ========================================
    # Model Settings
    # ========================================
    
    MODEL_NAME: str = Field(
        default="yolov8n.pt",
        description="YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)"
    )
    
    MODEL_DEVICE: str = Field(
        default="cpu",
        description="Device for model inference (cpu, cuda, mps)"
    )
    
    MODEL_CONFIDENCE_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default confidence threshold for detections"
    )
    
    MODEL_IOU_THRESHOLD: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS"
    )
    
    MODEL_MAX_DETECTIONS: int = Field(
        default=100,
        ge=1,
        le=300,
        description="Maximum detections per image"
    )
    
    MODEL_INPUT_SIZE: int = Field(
        default=640,
        ge=320,
        le=1280,
        description="Model input size (must be multiple of 32)"
    )
    
    @validator('MODEL_INPUT_SIZE')
    def validate_input_size(cls, v):
        """Ensure input size is multiple of 32"""
        if v % 32 != 0:
            raise ValueError("MODEL_INPUT_SIZE must be multiple of 32")
        return v
    
    # ========================================
    # Detection Service Settings
    # ========================================
    
    DETECTION_CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable detection result caching"
    )
    
    DETECTION_CACHE_SIZE: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Detection cache size"
    )
    
    DETECTION_RESIZE_ENABLED: bool = Field(
        default=True,
        description="Auto-resize images for performance"
    )
    
    DETECTION_MAX_WIDTH: int = Field(
        default=640,
        ge=320,
        le=1920,
        description="Maximum image width for processing"
    )
    
    DETECTION_MAX_HEIGHT: int = Field(
        default=480,
        ge=240,
        le=1080,
        description="Maximum image height for processing"
    )
    
    # ========================================
    # Alert Settings
    # ========================================
    
    ALERT_COOLDOWN_SECONDS: int = Field(
        default=5,
        ge=0,
        le=300,
        description="Cooldown period between duplicate alerts (seconds)"
    )
    
    ALERT_MAX_PER_FRAME: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum alerts to generate per frame"
    )
    
    ALERT_HISTORY_ENABLED: bool = Field(
        default=True,
        description="Enable alert history tracking"
    )
    
    ALERT_HISTORY_SIZE: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Alert history size"
    )
    
    ALERT_MIN_SEVERITY: str = Field(
        default="low",
        description="Minimum alert severity to track (low, medium, high, critical)"
    )
    
    # ========================================
    # Danger Classification Settings
    # ========================================
    
    DANGER_CLASSIFIER_HISTORY_SIZE: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Danger classifier temporal history size"
    )
    
    DANGER_CRITICAL_DISTANCE: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Critical danger distance threshold (pixels)"
    )
    
    DANGER_HIGH_RISK_DISTANCE: int = Field(
        default=100,
        ge=20,
        le=300,
        description="High risk distance threshold (pixels)"
    )
    
    DANGER_MODERATE_DISTANCE: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Moderate danger distance threshold (pixels)"
    )
    
    # ========================================
    # WebSocket Settings
    # ========================================
    
    WS_HEARTBEAT_INTERVAL: int = Field(
        default=30,
        ge=5,
        le=300,
        description="WebSocket heartbeat interval (seconds)"
    )
    
    WS_MAX_CONNECTIONS: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum concurrent WebSocket connections"
    )
    
    WS_MESSAGE_QUEUE_SIZE: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="WebSocket message queue size"
    )
    
    WS_DEFAULT_FPS: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Default FPS for WebSocket video streaming"
    )
    
    # ========================================
    # Logging Settings
    # ========================================
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    LOG_DIR: Path = Field(
        default=Path("logs"),
        description="Log directory path"
    )
    
    LOG_FORMAT: str = Field(
        default="detailed",
        description="Log format (simple, standard, detailed)"
    )
    
    LOG_JSON_ENABLED: bool = Field(
        default=False,
        description="Enable JSON formatted logs"
    )
    
    LOG_FILE_ENABLED: bool = Field(
        default=True,
        description="Enable file logging"
    )
    
    LOG_CONSOLE_ENABLED: bool = Field(
        default=True,
        description="Enable console logging"
    )
    
    LOG_ROTATION_ENABLED: bool = Field(
        default=True,
        description="Enable log rotation"
    )
    
    LOG_MAX_SIZE_MB: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum log file size in MB"
    )
    
    LOG_BACKUP_COUNT: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Number of log backup files"
    )
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    # ========================================
    # File Upload Settings
    # ========================================
    
    UPLOAD_MAX_SIZE_MB: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum upload file size in MB"
    )
    
    UPLOAD_ALLOWED_EXTENSIONS: List[str] = Field(
        default=["jpg", "jpeg", "png", "bmp", "webp"],
        description="Allowed image upload extensions"
    )
    
    UPLOAD_DIR: Path = Field(
        default=Path("uploads"),
        description="Upload directory path"
    )
    
    # ========================================
    # Video Processing Settings
    # ========================================
    
    VIDEO_PROCESSING_ENABLED: bool = Field(
        default=True,
        description="Enable video file processing"
    )
    
    VIDEO_MAX_SIZE_MB: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum video file size in MB"
    )
    
    VIDEO_ALLOWED_FORMATS: List[str] = Field(
        default=["mp4", "avi", "mov", "mkv"],
        description="Allowed video formats"
    )
    
    VIDEO_OUTPUT_DIR: Path = Field(
        default=Path("output_videos"),
        description="Video output directory"
    )
    
    VIDEO_DEFAULT_FPS: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Default FPS for video processing"
    )
    
    # ========================================
    # Performance Settings
    # ========================================
    
    PERFORMANCE_MONITORING_ENABLED: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    
    PERFORMANCE_LOG_INTERVAL: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Performance logging interval (seconds)"
    )
    
    RATE_LIMIT_ENABLED: bool = Field(
        default=False,
        description="Enable rate limiting"
    )
    
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Maximum requests per minute per IP"
    )
    
    # ========================================
    # Cache Settings
    # ========================================
    
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable caching"
    )
    
    CACHE_TTL_SECONDS: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Cache TTL in seconds"
    )
    
    CACHE_MAX_SIZE: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum cache entries"
    )
    
    # ========================================
    # Security Settings
    # ========================================
    
    SECRET_KEY: str = Field(
        default="changeme-in-production-A-P-U-R-B-O-2025-11-07",
        description="Secret key for signing/encryption"
    )
    
    API_KEY_ENABLED: bool = Field(
        default=False,
        description="Enable API key authentication"
    )
    
    API_KEYS: List[str] = Field(
        default=[],
        description="Valid API keys"
    )
    
    ALLOWED_IPS: List[str] = Field(
        default=["*"],
        description="Allowed IP addresses (* for all)"
    )
    
    # ========================================
    # Database Settings (Future Use)
    # ========================================
    
    DATABASE_ENABLED: bool = Field(
        default=False,
        description="Enable database connection"
    )
    
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="Database connection URL"
    )
    
    DATABASE_POOL_SIZE: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Database connection pool size"
    )
    
    # ========================================
    # External Services (Future Use)
    # ========================================
    
    NOTIFICATION_ENABLED: bool = Field(
        default=False,
        description="Enable notification service"
    )
    
    EMAIL_ENABLED: bool = Field(
        default=False,
        description="Enable email notifications"
    )
    
    EMAIL_SMTP_HOST: Optional[str] = Field(
        default=None,
        description="SMTP host"
    )
    
    EMAIL_SMTP_PORT: int = Field(
        default=587,
        ge=1,
        le=65535,
        description="SMTP port"
    )
    
    EMAIL_FROM: Optional[str] = Field(
        default=None,
        description="Email from address"
    )
    
    WEBHOOK_ENABLED: bool = Field(
        default=False,
        description="Enable webhook notifications"
    )
    
    WEBHOOK_URL: Optional[HttpUrl] = Field(
        default=None,
        description="Webhook URL for notifications"
    )
    
    # ========================================
    # Feature Flags
    # ========================================
    
    FEATURE_BATCH_PROCESSING: bool = Field(
        default=True,
        description="Enable batch image processing"
    )
    
    FEATURE_VIDEO_PROCESSING: bool = Field(
        default=True,
        description="Enable video file processing"
    )
    
    FEATURE_REAL_TIME_STREAMING: bool = Field(
        default=True,
        description="Enable real-time video streaming"
    )
    
    FEATURE_DANGER_ASSESSMENT: bool = Field(
        default=True,
        description="Enable advanced danger assessment"
    )
    
    FEATURE_ALERT_SYSTEM: bool = Field(
        default=True,
        description="Enable alert generation system"
    )
    
    FEATURE_STATISTICS: bool = Field(
        default=True,
        description="Enable statistics tracking"
    )
    
    # ========================================
    # Metadata
    # ========================================
    
    CREATED_DATE: str = Field(
        default="2025-11-07 13:18:18 UTC",
        description="Configuration created date"
    )
    
    CREATED_BY: str = Field(
        default="A-P-U-R-B-O",
        description="Configuration created by"
    )
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    def __init__(self, **kwargs):
        """Initialize settings"""
        super().__init__(**kwargs)
        
        # Create necessary directories
        self._create_directories()
        
        # Log configuration summary
        logger.info(f"Configuration loaded by {self.CREATED_BY}")
        logger.info(f"Environment: {self.ENVIRONMENT}")
        logger.info(f"Debug Mode: {self.DEBUG}")
        logger.info(f"Model: {self.MODEL_NAME} on {self.MODEL_DEVICE}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.LOG_DIR,
            self.UPLOAD_DIR,
            self.VIDEO_OUTPUT_DIR
        ]
        
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "application": {
                "name": self.APP_NAME,
                "version": self.APP_VERSION,
                "author": self.APP_AUTHOR,
                "environment": self.ENVIRONMENT,
                "debug": self.DEBUG
            },
            "server": {
                "host": self.HOST,
                "port": self.PORT,
                "workers": self.WORKERS,
                "base_url": self.BASE_URL
            },
            "model": {
                "name": self.MODEL_NAME,
                "device": self.MODEL_DEVICE,
                "confidence_threshold": self.MODEL_CONFIDENCE_THRESHOLD,
                "max_detections": self.MODEL_MAX_DETECTIONS
            },
            "services": {
                "detection_cache": self.DETECTION_CACHE_ENABLED,
                "alert_system": self.FEATURE_ALERT_SYSTEM,
                "danger_assessment": self.FEATURE_DANGER_ASSESSMENT,
                "video_processing": self.FEATURE_VIDEO_PROCESSING
            },
            "logging": {
                "level": self.LOG_LEVEL,
                "file_enabled": self.LOG_FILE_ENABLED,
                "json_enabled": self.LOG_JSON_ENABLED
            },
            "features": {
                "batch_processing": self.FEATURE_BATCH_PROCESSING,
                "real_time_streaming": self.FEATURE_REAL_TIME_STREAMING,
                "statistics": self.FEATURE_STATISTICS
            },
            "metadata": {
                "created_date": self.CREATED_DATE,
                "created_by": self.CREATED_BY
            }
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration settings
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check critical settings
        if self.ENVIRONMENT == "production" and self.DEBUG:
            warnings.append("Debug mode is enabled in production environment")
        
        if self.SECRET_KEY == "changeme-in-production-A-P-U-R-B-O-2025-11-07":
            if self.ENVIRONMENT == "production":
                issues.append("Default SECRET_KEY is being used in production")
            else:
                warnings.append("Using default SECRET_KEY")
        
        if self.MODEL_DEVICE == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    issues.append("CUDA device specified but not available")
            except ImportError:
                issues.append("CUDA device specified but PyTorch not available")
        
        if self.CORS_ORIGINS == ["*"] and self.ENVIRONMENT == "production":
            warnings.append("CORS is open to all origins in production")
        
        if self.WORKERS > 1 and self.RELOAD:
            warnings.append("Auto-reload is enabled with multiple workers")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "timestamp": "2025-11-07 13:18:18 UTC",
            "validated_by": "A-P-U-R-B-O"
        }


# ============================================================================
# Configuration Instance
# ============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get settings instance (cached)
    
    Returns:
        Settings instance
    """
    logger.info("Loading application settings...")
    settings = Settings()
    
    # Validate configuration
    validation = settings.validate_configuration()
    
    if not validation["valid"]:
        logger.error(f"Configuration validation failed: {validation['issues']}")
        for issue in validation["issues"]:
            logger.error(f"  - {issue}")
    
    if validation["warnings"]:
        logger.warning(f"Configuration warnings: {len(validation['warnings'])}")
        for warning in validation["warnings"]:
            logger.warning(f"  - {warning}")
    
    logger.info("✅ Configuration loaded successfully")
    logger.info(f"📋 Configuration Summary:")
    summary = settings.get_summary()
    logger.info(f"   - Environment: {summary['application']['environment']}")
    logger.info(f"   - Model: {summary['model']['name']} on {summary['model']['device']}")
    logger.info(f"   - Server: {summary['server']['host']}:{summary['server']['port']}")
    logger.info(f"   - Created by: {summary['metadata']['created_by']}")
    
    return settings


# ============================================================================
# Configuration Helpers
# ============================================================================

def reload_settings():
    """
    Reload settings (clear cache)
    """
    get_settings.cache_clear()
    logger.info("Settings cache cleared and will be reloaded on next access")


def export_settings_to_file(filepath: str = "config_export.json"):
    """
    Export current settings to JSON file
    
    Args:
        filepath: Output file path
    """
    import json
    
    settings = get_settings()
    summary = settings.get_summary()
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Settings exported to {filepath}")


def print_settings_summary():
    """
    Print settings summary to console
    """
    settings = get_settings()
    summary = settings.get_summary()
    
    print("\n" + "="*70)
    print(f"CONFIGURATION SUMMARY - {settings.APP_NAME}")
    print("="*70)
    print(f"Author: {settings.APP_AUTHOR}")
    print(f"Created: {settings.CREATED_DATE}")
    print(f"Version: {settings.APP_VERSION}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print("-"*70)
    print(f"Server: {settings.HOST}:{settings.PORT} ({settings.WORKERS} workers)")
    print(f"Model: {settings.MODEL_NAME} on {settings.MODEL_DEVICE}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"Cache Enabled: {settings.CACHE_ENABLED}")
    print("="*70 + "\n")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Get settings
    settings = get_settings()
    
    # Print summary
    print_settings_summary()
    
    # Validate configuration
    validation = settings.validate_configuration()
    print(f"\n Configuration Validation: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
    
    if validation["issues"]:
        print("\nIssues:")
        for issue in validation["issues"]:
            print(f"  ❌ {issue}")
    
    if validation["warnings"]:
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  ⚠️  {warning}")
    
    # Export settings
    export_settings_to_file("config_export.json")
    print("\n✅ Configuration exported to config_export.json")