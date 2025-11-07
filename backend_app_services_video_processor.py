"""
Video Processor Service Module
Handles video stream processing, frame extraction, buffering, and optimization
Supports real-time video streams, recorded videos, and batch processing
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import threading
import queue
import time
from pathlib import Path
import tempfile

from app.core.yolo_detector import ConstructionSafetyDetector
from app.services.detection_service import DetectionService
from app.services.alert_service import AlertService
from app.utils.logger import get_logger
from app.utils.image_utils import resize_image, validate_image

logger = get_logger(__name__)


class FrameBuffer:
    """
    Thread-safe frame buffer for video processing
    """
    
    def __init__(self, max_size: int = 30):
        """
        Initialize frame buffer
        
        Args:
            max_size: Maximum number of frames to buffer
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frame_count = 0
    
    def add(self, frame: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Add frame to buffer
        
        Args:
            frame: Frame as numpy array
            metadata: Optional metadata about the frame
        """
        with self.lock:
            self.frame_count += 1
            self.buffer.append({
                "frame": frame,
                "frame_number": self.frame_count,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            })
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get latest frame from buffer
        
        Returns:
            Latest frame dictionary or None
        """
        with self.lock:
            return self.buffer[-1] if self.buffer else None
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all frames in buffer
        
        Returns:
            List of frame dictionaries
        """
        with self.lock:
            return list(self.buffer)
    
    def clear(self):
        """Clear all frames from buffer"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self.lock:
            return len(self.buffer) == 0


class VideoStreamProcessor:
    """
    Processes video streams in real-time with detection
    """
    
    def __init__(
        self,
        detector: ConstructionSafetyDetector,
        detection_service: DetectionService,
        alert_service: AlertService,
        target_fps: int = 10,
        buffer_size: int = 30
    ):
        """
        Initialize video stream processor
        
        Args:
            detector: ConstructionSafetyDetector instance
            detection_service: DetectionService instance
            alert_service: AlertService instance
            target_fps: Target frames per second for processing
            buffer_size: Frame buffer size
        """
        self.detector = detector
        self.detection_service = detection_service
        self.alert_service = alert_service
        self.target_fps = target_fps
        self.buffer = FrameBuffer(max_size=buffer_size)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)
        
        # Statistics
        self.total_frames_processed = 0
        self.total_detections = 0
        self.total_alerts = 0
        self.processing_start_time = None
        self.dropped_frames = 0
        
        logger.info(f"VideoStreamProcessor initialized (target_fps={target_fps})")
    
    def start_processing(self):
        """Start the video processing thread"""
        if self.is_processing:
            logger.warning("Processing already started")
            return
        
        self.is_processing = True
        self.processing_start_time = datetime.utcnow()
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Video processing started")
    
    def stop_processing(self):
        """Stop the video processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Video processing stopped")
    
    def add_frame(self, frame: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Add frame to processing queue
        
        Args:
            frame: Frame as numpy array
            metadata: Optional frame metadata
        """
        try:
            self.frame_queue.put({
                "frame": frame,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }, block=False)
        except queue.Full:
            self.dropped_frames += 1
            logger.warning(f"Frame dropped (queue full). Total dropped: {self.dropped_frames}")
    
    def get_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get processing result from queue
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Result dictionary or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop (runs in separate thread)"""
        frame_delay = 1.0 / self.target_fps
        
        while self.is_processing:
            try:
                loop_start = time.time()
                
                # Get frame from queue
                try:
                    frame_data = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                frame = frame_data["frame"]
                metadata = frame_data["metadata"]
                
                # Process frame
                result = self._process_single_frame(frame, metadata)
                
                # Add to buffer
                self.buffer.add(frame, metadata)
                
                # Put result in output queue
                try:
                    self.result_queue.put(result, block=False)
                except queue.Full:
                    logger.warning("Result queue full, dropping result")
                
                # Update statistics
                self.total_frames_processed += 1
                self.total_detections += result.get("total_objects", 0)
                self.total_alerts += len(result.get("alerts", []))
                
                # FPS limiting
                elapsed = time.time() - loop_start
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}", exc_info=True)
    
    def _process_single_frame(
        self,
        frame: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single frame
        
        Args:
            frame: Frame to process
            metadata: Frame metadata
            
        Returns:
            Processing result dictionary
        """
        try:
            # Validate and resize frame
            if not validate_image(frame):
                return {"success": False, "error": "Invalid frame"}
            
            # Resize for performance
            processed_frame = resize_image(frame, max_width=640, max_height=480)
            
            # Run detection
            detection_result = self.detection_service.process_frame_fast(
                frame=processed_frame,
                conf_threshold=0.5
            )
            
            if not detection_result["success"]:
                return detection_result
            
            # Generate alerts
            alerts = self.alert_service.generate_alerts(
                detections=detection_result["detections"]
            )
            
            return {
                "success": True,
                "frame_number": self.total_frames_processed,
                "detections": detection_result["detections"],
                "alerts": alerts,
                "total_objects": len(detection_result["detections"]),
                "dangerous_objects": len([
                    d for d in detection_result["detections"]
                    if d["danger_level"] in ["critical", "high"]
                ]),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics
        
        Returns:
            Statistics dictionary
        """
        uptime = (
            (datetime.utcnow() - self.processing_start_time).total_seconds()
            if self.processing_start_time else 0
        )
        
        actual_fps = (
            self.total_frames_processed / uptime
            if uptime > 0 else 0
        )
        
        return {
            "is_processing": self.is_processing,
            "total_frames_processed": self.total_frames_processed,
            "total_detections": self.total_detections,
            "total_alerts": self.total_alerts,
            "dropped_frames": self.dropped_frames,
            "uptime_seconds": round(uptime, 2),
            "target_fps": self.target_fps,
            "actual_fps": round(actual_fps, 2),
            "buffer_size": self.buffer.size(),
            "queue_size": self.frame_queue.qsize(),
            "result_queue_size": self.result_queue.qsize()
        }


class VideoFileProcessor:
    """
    Processes video files with detection and saves results
    """
    
    def __init__(
        self,
        detector: ConstructionSafetyDetector,
        detection_service: DetectionService,
        alert_service: AlertService
    ):
        """
        Initialize video file processor
        
        Args:
            detector: ConstructionSafetyDetector instance
            detection_service: DetectionService instance
            alert_service: AlertService instance
        """
        self.detector = detector
        self.detection_service = detection_service
        self.alert_service = alert_service
        
        logger.info("VideoFileProcessor initialized")
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        draw_detections: bool = True,
        skip_frames: int = 0,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Process entire video file
        
        Args:
            video_path: Path to input video file
            output_path: Optional path for output video with annotations
            draw_detections: Whether to draw detections on output video
            skip_frames: Number of frames to skip between processing
            max_frames: Maximum number of frames to process
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Processing results dictionary
        """
        if not Path(video_path).exists():
            return {"success": False, "error": f"Video file not found: {video_path}"}
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"success": False, "error": "Failed to open video file"}
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {video_path}")
            logger.info(f"Properties: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
            
            # Initialize video writer if output path provided
            writer = None
            if output_path and draw_detections:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            # Processing statistics
            results = {
                "success": True,
                "video_path": video_path,
                "output_path": output_path,
                "properties": {
                    "fps": fps,
                    "width": frame_width,
                    "height": frame_height,
                    "total_frames": total_frames
                },
                "frames_processed": 0,
                "frames_skipped": 0,
                "total_detections": 0,
                "total_alerts": 0,
                "detections_by_frame": [],
                "alerts_by_frame": []
            }
            
            frame_count = 0
            processed_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Check max frames limit
                if max_frames and processed_count >= max_frames:
                    break
                
                # Skip frames if configured
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    results["frames_skipped"] += 1
                    continue
                
                # Process frame
                detection_result = self.detection_service.process_frame(
                    frame=frame,
                    conf_threshold=0.5
                )
                
                if detection_result["success"]:
                    detections = detection_result["detections"]
                    
                    # Generate alerts
                    alerts = self.alert_service.generate_alerts(detections)
                    
                    # Update statistics
                    results["frames_processed"] += 1
                    results["total_detections"] += len(detections)
                    results["total_alerts"] += len(alerts)
                    
                    # Store frame results
                    results["detections_by_frame"].append({
                        "frame_number": frame_count,
                        "detections": detections,
                        "alerts": alerts
                    })
                    
                    # Draw detections if enabled
                    if draw_detections and writer:
                        annotated_frame = self.detection_service.draw_detections(
                            frame=frame,
                            detections=detections
                        )
                        writer.write(annotated_frame)
                    elif writer:
                        writer.write(frame)
                
                processed_count += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(processed_count, total_frames)
                
                # Log progress periodically
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{total_frames} frames")
            
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            
            processing_time = time.time() - start_time
            results["processing_time_seconds"] = round(processing_time, 2)
            results["avg_fps"] = round(processed_count / processing_time, 2) if processing_time > 0 else 0
            
            logger.info(f"Video processing completed: {processed_count} frames in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval_seconds: float = 1.0,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            interval_seconds: Interval between extracted frames
            max_frames: Maximum number of frames to extract
            
        Returns:
            Extraction results
        """
        if not Path(video_path).exists():
            return {"success": False, "error": f"Video file not found: {video_path}"}
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"success": False, "error": "Failed to open video file"}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval_seconds)
            
            extracted_frames = []
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check if we should extract this frame
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = output_path / frame_filename
                    
                    cv2.imwrite(str(frame_path), frame)
                    
                    extracted_frames.append({
                        "frame_number": frame_count,
                        "timestamp_seconds": frame_count / fps,
                        "file_path": str(frame_path)
                    })
                    
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {extracted_count} frames from {video_path}")
            
            return {
                "success": True,
                "video_path": video_path,
                "output_directory": output_dir,
                "total_frames": frame_count,
                "extracted_frames": extracted_count,
                "frames": extracted_frames
            }
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def create_summary_video(
        self,
        video_path: str,
        detections_data: List[Dict[str, Any]],
        output_path: str,
        highlight_dangerous: bool = True
    ) -> Dict[str, Any]:
        """
        Create summary video with only frames containing detections
        
        Args:
            video_path: Path to input video
            detections_data: List of detection results by frame
            output_path: Path for output summary video
            highlight_dangerous: Whether to highlight only dangerous detections
            
        Returns:
            Summary creation results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"success": False, "error": "Failed to open video"}
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frames_included = 0
            
            for detection_data in detections_data:
                frame_number = detection_data["frame_number"]
                detections = detection_data.get("detections", [])
                
                # Filter for dangerous detections if enabled
                if highlight_dangerous:
                    detections = [
                        d for d in detections
                        if d["danger_level"] in ["critical", "high"]
                    ]
                
                if not detections:
                    continue
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # Draw detections
                    annotated = self.detection_service.draw_detections(
                        frame=frame,
                        detections=detections
                    )
                    writer.write(annotated)
                    frames_included += 1
            
            cap.release()
            writer.release()
            
            logger.info(f"Summary video created: {frames_included} frames")
            
            return {
                "success": True,
                "output_path": output_path,
                "frames_included": frames_included,
                "total_input_frames": len(detections_data)
            }
            
        except Exception as e:
            logger.error(f"Error creating summary video: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}


class BatchVideoProcessor:
    """
    Process multiple videos in batch
    """
    
    def __init__(
        self,
        detector: ConstructionSafetyDetector,
        detection_service: DetectionService,
        alert_service: AlertService
    ):
        """
        Initialize batch processor
        
        Args:
            detector: ConstructionSafetyDetector instance
            detection_service: DetectionService instance
            alert_service: AlertService instance
        """
        self.file_processor = VideoFileProcessor(
            detector=detector,
            detection_service=detection_service,
            alert_service=alert_service
        )
        
        logger.info("BatchVideoProcessor initialized")
    
    def process_batch(
        self,
        video_paths: List[str],
        output_dir: str,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple videos
        
        Args:
            video_paths: List of video file paths
            output_dir: Output directory for processed videos
            parallel: Whether to process in parallel (not implemented yet)
            
        Returns:
            Batch processing results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            "success": True,
            "total_videos": len(video_paths),
            "processed_videos": 0,
            "failed_videos": 0,
            "results": []
        }
        
        for idx, video_path in enumerate(video_paths):
            logger.info(f"Processing video {idx + 1}/{len(video_paths)}: {video_path}")
            
            video_name = Path(video_path).stem
            output_video_path = str(output_path / f"{video_name}_processed.mp4")
            
            result = self.file_processor.process_video(
                video_path=video_path,
                output_path=output_video_path,
                draw_detections=True
            )
            
            if result["success"]:
                results["processed_videos"] += 1
            else:
                results["failed_videos"] += 1
            
            results["results"].append(result)
        
        logger.info(f"Batch processing completed: {results['processed_videos']}/{results['total_videos']} successful")
        
        return results


def create_video_processor(
    detector: ConstructionSafetyDetector,
    detection_service: DetectionService,
    alert_service: AlertService,
    processor_type: str = "stream"
) -> Any:
    """
    Factory function to create video processors
    
    Args:
        detector: ConstructionSafetyDetector instance
        detection_service: DetectionService instance
        alert_service: AlertService instance
        processor_type: Type of processor ("stream", "file", "batch")
        
    Returns:
        Video processor instance
    """
    if processor_type == "stream":
        return VideoStreamProcessor(detector, detection_service, alert_service)
    elif processor_type == "file":
        return VideoFileProcessor(detector, detection_service, alert_service)
    elif processor_type == "batch":
        return BatchVideoProcessor(detector, detection_service, alert_service)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")