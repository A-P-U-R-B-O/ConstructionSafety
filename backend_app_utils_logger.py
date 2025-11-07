"""
Logger Utilities Module
Comprehensive logging configuration for the construction safety detection system
Provides structured logging with multiple handlers, formatters, and log levels
"""

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Optional
import json
from pythonjsonlogger import jsonlogger


# ============================================================================
# Constants
# ============================================================================

# Log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Default log directory
DEFAULT_LOG_DIR = Path("logs")

# Log file names
APP_LOG_FILE = "app.log"
ERROR_LOG_FILE = "error.log"
ACCESS_LOG_FILE = "access.log"
DETECTION_LOG_FILE = "detection.log"
ALERT_LOG_FILE = "alert.log"

# Log format templates
STANDARD_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"
SIMPLE_FORMAT = "%(levelname)s - %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File rotation settings
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


# ============================================================================
# Custom Formatters
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for console output
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors"""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the message
        formatted = super().format(record)
        
        # Add color to level name
        colored_level = f"{log_color}{record.levelname}{reset}"
        formatted = formatted.replace(record.levelname, colored_level, 1)
        
        return formatted


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging
    """
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to JSON log"""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add level
        log_record['level'] = record.levelname
        
        # Add logger name
        log_record['logger'] = record.name
        
        # Add user context (if available)
        if hasattr(record, 'user'):
            log_record['user'] = record.user
        else:
            log_record['user'] = 'A-P-U-R-B-O'
        
        # Add module info
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


# ============================================================================
# Logger Configuration
# ============================================================================

class LoggerConfig:
    """
    Logger configuration manager
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = False,
        enable_rotation: bool = True
    ):
        """
        Initialize logger configuration
        
        Args:
            log_dir: Directory for log files
            log_level: Default log level
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_json: Enable JSON formatted logs
            enable_rotation: Enable log rotation
        """
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.log_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.enable_rotation = enable_rotation
        
        # Create log directory
        if self.enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Track configured loggers
        self.configured_loggers = set()
    
    def get_console_handler(self, use_colors: bool = True) -> logging.StreamHandler:
        """
        Create console handler
        
        Args:
            use_colors: Whether to use colored output
            
        Returns:
            Console handler
        """
        console_handler = logging.StreamHandler(sys.stdout)
        
        if use_colors:
            formatter = ColoredFormatter(
                DETAILED_FORMAT,
                datefmt=DATE_FORMAT
            )
        else:
            formatter = logging.Formatter(
                DETAILED_FORMAT,
                datefmt=DATE_FORMAT
            )
        
        console_handler.setFormatter(formatter)
        return console_handler
    
    def get_file_handler(
        self,
        filename: str,
        level: int = logging.INFO
    ) -> logging.Handler:
        """
        Create file handler with rotation
        
        Args:
            filename: Log file name
            level: Log level for this handler
            
        Returns:
            File handler
        """
        file_path = self.log_dir / filename
        
        if self.enable_rotation:
            # Rotating file handler (size-based)
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding='utf-8'
            )
        else:
            # Standard file handler
            file_handler = logging.FileHandler(
                file_path,
                encoding='utf-8'
            )
        
        file_handler.setLevel(level)
        
        if self.enable_json:
            formatter = CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            formatter = logging.Formatter(
                DETAILED_FORMAT,
                datefmt=DATE_FORMAT
            )
        
        file_handler.setFormatter(formatter)
        return file_handler
    
    def get_timed_rotating_handler(
        self,
        filename: str,
        when: str = 'midnight',
        interval: int = 1,
        backup_count: int = 7
    ) -> TimedRotatingFileHandler:
        """
        Create timed rotating file handler
        
        Args:
            filename: Log file name
            when: When to rotate ('midnight', 'H', 'D', 'W0'-'W6')
            interval: Rotation interval
            backup_count: Number of backups to keep
            
        Returns:
            Timed rotating file handler
        """
        file_path = self.log_dir / filename
        
        handler = TimedRotatingFileHandler(
            file_path,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        if self.enable_json:
            formatter = CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            formatter = logging.Formatter(
                DETAILED_FORMAT,
                datefmt=DATE_FORMAT
            )
        
        handler.setFormatter(formatter)
        return handler
    
    def configure_logger(
        self,
        name: str,
        level: Optional[int] = None,
        handlers: Optional[list] = None
    ) -> logging.Logger:
        """
        Configure a logger with specified handlers
        
        Args:
            name: Logger name
            level: Log level (uses default if None)
            handlers: List of handlers (creates default if None)
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        
        # Only configure if not already configured
        if name in self.configured_loggers:
            return logger
        
        logger.setLevel(level or self.log_level)
        logger.propagate = False
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add handlers
        if handlers is None:
            handlers = []
            
            # Add console handler
            if self.enable_console:
                handlers.append(self.get_console_handler())
            
            # Add file handler
            if self.enable_file:
                handlers.append(self.get_file_handler(APP_LOG_FILE))
        
        for handler in handlers:
            logger.addHandler(handler)
        
        self.configured_loggers.add(name)
        
        return logger


# ============================================================================
# Specialized Logger Functions
# ============================================================================

def get_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Get or create a logger with default configuration
    
    Args:
        name: Logger name (usually __name__)
        level: Log level ('DEBUG', 'INFO', etc.)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Get environment settings
    env_level = os.getenv('LOG_LEVEL', 'INFO')
    log_level = level or env_level
    
    # Create config
    config = LoggerConfig(
        log_level=log_level,
        enable_console=log_to_console,
        enable_file=log_to_file
    )
    
    return config.configure_logger(name)


def get_detection_logger() -> logging.Logger:
    """
    Get specialized logger for detection operations
    
    Returns:
        Detection logger
    """
    config = LoggerConfig(log_level="INFO")
    
    handlers = [
        config.get_console_handler(),
        config.get_file_handler(DETECTION_LOG_FILE)
    ]
    
    return config.configure_logger('detection', handlers=handlers)


def get_alert_logger() -> logging.Logger:
    """
    Get specialized logger for alert operations
    
    Returns:
        Alert logger
    """
    config = LoggerConfig(log_level="INFO")
    
    handlers = [
        config.get_console_handler(),
        config.get_file_handler(ALERT_LOG_FILE)
    ]
    
    return config.configure_logger('alert', handlers=handlers)


def get_access_logger() -> logging.Logger:
    """
    Get specialized logger for access logs (API requests)
    
    Returns:
        Access logger
    """
    config = LoggerConfig(log_level="INFO", enable_json=True)
    
    handlers = [
        config.get_file_handler(ACCESS_LOG_FILE)
    ]
    
    return config.configure_logger('access', handlers=handlers)


def get_error_logger() -> logging.Logger:
    """
    Get specialized logger for errors only
    
    Returns:
        Error logger
    """
    config = LoggerConfig(log_level="ERROR")
    
    handlers = [
        config.get_console_handler(),
        config.get_file_handler(ERROR_LOG_FILE, level=logging.ERROR)
    ]
    
    return config.configure_logger('error', handlers=handlers)


# ============================================================================
# Logging Utilities
# ============================================================================

class LogContext:
    """
    Context manager for temporary log level changes
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize log context
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.level = level
        self.original_level = logger.level
    
    def __enter__(self):
        """Set temporary log level"""
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log level"""
        self.logger.setLevel(self.original_level)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls
    
    Args:
        logger: Logger to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}() with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__}() returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__}() raised exception: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time
    
    Args:
        logger: Logger to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # ms
                logger.info(f"{func.__name__}() executed in {execution_time:.2f}ms")
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000  # ms
                logger.error(
                    f"{func.__name__}() failed after {execution_time:.2f}ms: {str(e)}",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


class StructuredLogger:
    """
    Wrapper for structured logging with context
    """
    
    def __init__(self, logger: logging.Logger, context: Optional[dict] = None):
        """
        Initialize structured logger
        
        Args:
            logger: Base logger
            context: Default context to include in all logs
        """
        self.logger = logger
        self.context = context or {}
    
    def _add_context(self, message: str, extra_context: Optional[dict] = None) -> str:
        """
        Add context to log message
        
        Args:
            message: Log message
            extra_context: Additional context
            
        Returns:
            Message with context
        """
        context = {**self.context, **(extra_context or {})}
        if context:
            context_str = json.dumps(context)
            return f"{message} | Context: {context_str}"
        return message
    
    def debug(self, message: str, **context):
        """Log debug message with context"""
        self.logger.debug(self._add_context(message, context))
    
    def info(self, message: str, **context):
        """Log info message with context"""
        self.logger.info(self._add_context(message, context))
    
    def warning(self, message: str, **context):
        """Log warning message with context"""
        self.logger.warning(self._add_context(message, context))
    
    def error(self, message: str, exc_info: bool = False, **context):
        """Log error message with context"""
        self.logger.error(self._add_context(message, context), exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False, **context):
        """Log critical message with context"""
        self.logger.critical(self._add_context(message, context), exc_info=exc_info)


# ============================================================================
# Performance Logging
# ============================================================================

class PerformanceLogger:
    """
    Logger for performance metrics
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger
        
        Args:
            logger: Base logger
        """
        self.logger = logger
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """
        Start timing an operation
        
        Args:
            operation: Operation name
        """
        import time
        self.metrics[operation] = {
            'start_time': time.time(),
            'end_time': None,
            'duration_ms': None
        }
    
    def end_timer(self, operation: str):
        """
        End timing an operation and log result
        
        Args:
            operation: Operation name
        """
        import time
        if operation not in self.metrics:
            self.logger.warning(f"No timer started for operation: {operation}")
            return
        
        end_time = time.time()
        start_time = self.metrics[operation]['start_time']
        duration_ms = (end_time - start_time) * 1000
        
        self.metrics[operation]['end_time'] = end_time
        self.metrics[operation]['duration_ms'] = duration_ms
        
        self.logger.info(f"Performance: {operation} took {duration_ms:.2f}ms")
    
    def get_metrics(self) -> dict:
        """
        Get all recorded metrics
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()


# ============================================================================
# Global Logger Initialization
# ============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_json: bool = False
):
    """
    Setup global logging configuration
    
    Args:
        log_level: Default log level
        log_dir: Log directory path
        enable_json: Enable JSON logging
    """
    log_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    
    config = LoggerConfig(
        log_dir=log_path,
        log_level=log_level,
        enable_json=enable_json
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))
    
    # Add handlers
    if config.enable_console:
        root_logger.addHandler(config.get_console_handler())
    
    if config.enable_file:
        root_logger.addHandler(config.get_file_handler(APP_LOG_FILE))
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized - Level: {log_level}, Directory: {log_path}")
    logging.info(f"Current User: A-P-U-R-B-O")
    logging.info(f"Timestamp: 2025-11-07 13:06:10 UTC")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="DEBUG")
    
    # Get loggers
    app_logger = get_logger(__name__)
    detection_logger = get_detection_logger()
    alert_logger = get_alert_logger()
    
    # Test logging
    app_logger.debug("This is a debug message")
    app_logger.info("This is an info message")
    app_logger.warning("This is a warning message")
    app_logger.error("This is an error message")
    app_logger.critical("This is a critical message")
    
    # Test structured logging
    structured = StructuredLogger(app_logger, context={"user": "A-P-U-R-B-O"})
    structured.info("Detection completed", objects_detected=5, processing_time_ms=45.2)
    
    # Test performance logging
    perf = PerformanceLogger(app_logger)
    perf.start_timer("image_processing")
    import time
    time.sleep(0.1)
    perf.end_timer("image_processing")
    
    print("\n" + "="*50)
    print("Logging test completed!")
    print(f"Check logs directory: {DEFAULT_LOG_DIR}")
    print("="*50)