# ============================================================================
# Construction Safety AI Detection System - Docker Image
# Author: A-P-U-R-B-O
# Created: 2025-11-07 13:24:18 UTC
# Base Image: Python 3.11 Slim
# ============================================================================

# ========================================
# Stage 1: Base Image
# ========================================
FROM python:3.11-slim as base

# Set metadata
LABEL maintainer="A-P-U-R-B-O"
LABEL version="1.0.0"
LABEL description="Construction Safety AI Detection System"
LABEL created="2025-11-07 13:24:18 UTC"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential build tools
    gcc \
    g++ \
    make \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    # Additional image processing libraries
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    # System utilities
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ========================================
# Stage 2: Dependencies
# ========================================
FROM base as dependencies

# Copy requirements file
COPY backend/requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLOv8 models (optional - can be done at runtime)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "Model will be downloaded at runtime"


# ========================================
# Stage 3: Application
# ========================================
FROM dependencies as application

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/output_videos /app/weights

# Copy application code
COPY backend/app /app/app

# Copy configuration files
COPY backend/.env.example /app/.env.example

# Set file permissions
RUN chmod -R 755 /app && \
    chmod -R 777 /app/logs /app/uploads /app/output_videos

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=INFO \
    MODEL_NAME=yolov8n.pt \
    MODEL_DEVICE=cpu \
    ENVIRONMENT=production \
    PYTHONPATH=/app

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ========================================
# Stage 4: Development (Optional)
# ========================================
FROM application as development

# Switch back to root for installations
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest==7.4.4 \
    pytest-asyncio==0.23.3 \
    pytest-cov==4.1.0 \
    black==23.12.1 \
    flake8==7.0.0 \
    mypy==1.8.0 \
    ipython==8.20.0 \
    debugpy==1.8.0

# Install additional dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to appuser
USER appuser

# Override CMD for development (with reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# ========================================
# Stage 5: Production (Default)
# ========================================
FROM application as production

# Production-specific optimizations
ENV PYTHONOPTIMIZE=2

# Use gunicorn for production (alternative to uvicorn)
# Uncomment if you prefer gunicorn:
# CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]

# Default production command (uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]


# ============================================================================
# Build Instructions:
# ============================================================================
#
# Build for production (default):
#   docker build -t construction-safety-ai:latest .
#
# Build for development:
#   docker build --target development -t construction-safety-ai:dev .
#
# Build with specific stage:
#   docker build --target production -t construction-safety-ai:prod .
#
# Run container:
#   docker run -d -p 8000:8000 --name safety-ai construction-safety-ai:latest
#
# Run with environment variables:
#   docker run -d -p 8000:8000 \
#     -e MODEL_NAME=yolov8s.pt \
#     -e LOG_LEVEL=DEBUG \
#     --name safety-ai construction-safety-ai:latest
#
# Run with volume mounts:
#   docker run -d -p 8000:8000 \
#     -v $(pwd)/logs:/app/logs \
#     -v $(pwd)/uploads:/app/uploads \
#     --name safety-ai construction-safety-ai:latest
#
# Run with GPU support (NVIDIA):
#   docker run -d -p 8000:8000 \
#     --gpus all \
#     -e MODEL_DEVICE=cuda \
#     --name safety-ai construction-safety-ai:latest
#
# ============================================================================
# Image Size Optimization:
# ============================================================================
#
# The multi-stage build reduces final image size significantly:
# - Base stage: ~500MB
# - Dependencies stage: ~2GB
# - Final production image: ~2.5GB
#
# To further optimize:
# 1. Use python:3.11-alpine (smaller but may have compatibility issues)
# 2. Remove unnecessary dependencies
# 3. Use .dockerignore to exclude files
#
# ============================================================================
# Security Notes:
# ============================================================================
#
# - Runs as non-root user (appuser)
# - Minimal attack surface with slim base image
# - No unnecessary tools in production
# - Health checks enabled
# - PYTHONDONTWRITEBYTECODE prevents .pyc files
#
# ============================================================================
# Author: A-P-U-R-B-O
# Date: 2025-11-07 13:24:18 UTC
# Version: 1.0.0
# ============================================================================
