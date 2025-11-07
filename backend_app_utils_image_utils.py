"""
Image Utilities Module
Comprehensive image processing utilities for the construction safety detection system
Handles image encoding/decoding, validation, transformation, and optimization
"""

import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple, List, Union, Any
import imghdr
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Supported image formats
SUPPORTED_FORMATS = ['jpeg', 'jpg', 'png', 'bmp', 'webp']

# Maximum image dimensions
MAX_WIDTH = 4096
MAX_HEIGHT = 4096
MIN_WIDTH = 32
MIN_HEIGHT = 32

# Default processing dimensions
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# Image quality settings
JPEG_QUALITY = 85
PNG_COMPRESSION = 6


# ============================================================================
# Base64 Encoding/Decoding
# ============================================================================

def encode_image_to_base64(
    image: np.ndarray,
    format: str = 'jpeg',
    quality: int = JPEG_QUALITY
) -> str:
    """
    Encode numpy array image to base64 string
    
    Args:
        image: Image as numpy array (BGR or RGB)
        format: Output format ('jpeg', 'png')
        quality: JPEG quality (0-100) or PNG compression (0-9)
        
    Returns:
        Base64 encoded string
    """
    try:
        # Convert format to extension
        if format.lower() in ['jpg', 'jpeg']:
            ext = '.jpg'
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format.lower() == 'png':
            ext = '.png'
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, quality]
        else:
            ext = '.jpg'
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        
        # Encode image
        success, buffer = cv2.imencode(ext, image, encode_param)
        
        if not success:
            logger.error("Failed to encode image")
            return ""
        
        # Convert to base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return ""


def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Decode base64 string to numpy array image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    try:
        # Remove data URI prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode base64 image")
            return None
        
        return image
        
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        return None


def encode_pil_to_base64(
    pil_image: Image.Image,
    format: str = 'JPEG',
    quality: int = JPEG_QUALITY
) -> str:
    """
    Encode PIL Image to base64 string
    
    Args:
        pil_image: PIL Image object
        format: Output format ('JPEG', 'PNG')
        quality: Image quality
        
    Returns:
        Base64 encoded string
    """
    try:
        buffer = BytesIO()
        
        if format.upper() == 'JPEG':
            pil_image.save(buffer, format='JPEG', quality=quality)
        elif format.upper() == 'PNG':
            pil_image.save(buffer, format='PNG', compress_level=quality)
        else:
            pil_image.save(buffer, format=format)
        
        buffer.seek(0)
        base64_str = base64.b64encode(buffer.read()).decode('utf-8')
        
        return base64_str
        
    except Exception as e:
        logger.error(f"Error encoding PIL image to base64: {str(e)}")
        return ""


def decode_base64_to_pil(base64_string: str) -> Optional[Image.Image]:
    """
    Decode base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object or None if failed
    """
    try:
        # Remove data URI prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(img_bytes))
        
        return pil_image
        
    except Exception as e:
        logger.error(f"Error decoding base64 to PIL image: {str(e)}")
        return None


# ============================================================================
# Image Validation
# ============================================================================

def validate_image(image: np.ndarray) -> bool:
    """
    Validate numpy array image
    
    Args:
        image: Image as numpy array
        
    Returns:
        Boolean indicating if image is valid
    """
    try:
        if image is None:
            logger.warning("Image is None")
            return False
        
        if not isinstance(image, np.ndarray):
            logger.warning(f"Image is not numpy array: {type(image)}")
            return False
        
        if image.size == 0:
            logger.warning("Image is empty")
            return False
        
        if len(image.shape) < 2:
            logger.warning(f"Invalid image shape: {image.shape}")
            return False
        
        height, width = image.shape[:2]
        
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            logger.warning(f"Image too small: {width}x{height}")
            return False
        
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            logger.warning(f"Image too large: {width}x{height}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False


def validate_image_file(file_path: str) -> bool:
    """
    Validate image file
    
    Args:
        file_path: Path to image file
        
    Returns:
        Boolean indicating if file is valid image
    """
    try:
        if not Path(file_path).exists():
            logger.warning(f"File does not exist: {file_path}")
            return False
        
        # Check file format
        img_format = imghdr.what(file_path)
        
        if img_format not in SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format: {img_format}")
            return False
        
        # Try to load image
        image = cv2.imread(file_path)
        
        return validate_image(image)
        
    except Exception as e:
        logger.error(f"Error validating image file: {str(e)}")
        return False


def get_image_info(image: np.ndarray) -> dict:
    """
    Get detailed information about image
    
    Args:
        image: Image as numpy array
        
    Returns:
        Dictionary with image information
    """
    try:
        if not validate_image(image):
            return {"valid": False}
        
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        return {
            "valid": True,
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(image.dtype),
            "size_bytes": image.nbytes,
            "aspect_ratio": round(width / height, 2),
            "is_grayscale": channels == 1,
            "is_color": channels >= 3
        }
        
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return {"valid": False, "error": str(e)}


# ============================================================================
# Image Resizing and Transformation
# ============================================================================

def resize_image(
    image: np.ndarray,
    max_width: int = DEFAULT_WIDTH,
    max_height: int = DEFAULT_HEIGHT,
    maintain_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to fit within maximum dimensions
    
    Args:
        image: Image as numpy array
        max_width: Maximum width
        max_height: Maximum height
        maintain_aspect_ratio: Whether to maintain aspect ratio
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    try:
        if not validate_image(image):
            return image
        
        height, width = image.shape[:2]
        
        # Check if resize needed
        if width <= max_width and height <= max_height:
            return image
        
        if maintain_aspect_ratio:
            # Calculate scaling factor
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width = max_width
            new_height = max_height
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return resized
        
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image


def resize_to_fixed(
    image: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to fixed dimensions (may distort aspect ratio)
    
    Args:
        image: Image as numpy array
        width: Target width
        height: Target height
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    try:
        if not validate_image(image):
            return image
        
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        
        return resized
        
    except Exception as e:
        logger.error(f"Error resizing to fixed dimensions: {str(e)}")
        return image


def crop_image(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> Optional[np.ndarray]:
    """
    Crop image to specified region
    
    Args:
        image: Image as numpy array
        x: Top-left X coordinate
        y: Top-left Y coordinate
        width: Crop width
        height: Crop height
        
    Returns:
        Cropped image or None if invalid
    """
    try:
        if not validate_image(image):
            return None
        
        img_height, img_width = image.shape[:2]
        
        # Validate coordinates
        if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
            logger.warning(f"Invalid crop coordinates: ({x}, {y}, {width}, {height})")
            return None
        
        cropped = image[y:y+height, x:x+width]
        
        return cropped
        
    except Exception as e:
        logger.error(f"Error cropping image: {str(e)}")
        return None


def crop_to_bbox(
    image: np.ndarray,
    bbox: dict,
    padding: int = 0
) -> Optional[np.ndarray]:
    """
    Crop image to bounding box with optional padding
    
    Args:
        image: Image as numpy array
        bbox: Bounding box dict with x1, y1, x2, y2
        padding: Padding in pixels
        
    Returns:
        Cropped image or None if invalid
    """
    try:
        if not validate_image(image):
            return None
        
        img_height, img_width = image.shape[:2]
        
        # Extract coordinates with padding
        x1 = max(0, int(bbox['x1']) - padding)
        y1 = max(0, int(bbox['y1']) - padding)
        x2 = min(img_width, int(bbox['x2']) + padding)
        y2 = min(img_height, int(bbox['y2']) + padding)
        
        cropped = image[y1:y2, x1:x2]
        
        return cropped
        
    except Exception as e:
        logger.error(f"Error cropping to bbox: {str(e)}")
        return None


def rotate_image(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    scale: float = 1.0
) -> np.ndarray:
    """
    Rotate image by specified angle
    
    Args:
        image: Image as numpy array
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Center of rotation (default: image center)
        scale: Scale factor
        
    Returns:
        Rotated image
    """
    try:
        if not validate_image(image):
            return image
        
        height, width = image.shape[:2]
        
        if center is None:
            center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
        
    except Exception as e:
        logger.error(f"Error rotating image: {str(e)}")
        return image


def flip_image(
    image: np.ndarray,
    flip_code: int = 1
) -> np.ndarray:
    """
    Flip image horizontally, vertically, or both
    
    Args:
        image: Image as numpy array
        flip_code: 0=vertical, 1=horizontal, -1=both
        
    Returns:
        Flipped image
    """
    try:
        if not validate_image(image):
            return image
        
        flipped = cv2.flip(image, flip_code)
        
        return flipped
        
    except Exception as e:
        logger.error(f"Error flipping image: {str(e)}")
        return image


# ============================================================================
# Image Enhancement
# ============================================================================

def adjust_brightness(
    image: np.ndarray,
    value: int = 30
) -> np.ndarray:
    """
    Adjust image brightness
    
    Args:
        image: Image as numpy array
        value: Brightness adjustment (-255 to 255)
        
    Returns:
        Adjusted image
    """
    try:
        if not validate_image(image):
            return image
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Adjust value channel
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        hsv = cv2.merge([h, s, v])
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return adjusted
        
    except Exception as e:
        logger.error(f"Error adjusting brightness: {str(e)}")
        return image


def adjust_contrast(
    image: np.ndarray,
    alpha: float = 1.5
) -> np.ndarray:
    """
    Adjust image contrast
    
    Args:
        image: Image as numpy array
        alpha: Contrast factor (1.0 = no change, >1 = increase, <1 = decrease)
        
    Returns:
        Adjusted image
    """
    try:
        if not validate_image(image):
            return image
        
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        return adjusted
        
    except Exception as e:
        logger.error(f"Error adjusting contrast: {str(e)}")
        return image


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur to image
    
    Args:
        image: Image as numpy array
        kernel_size: Kernel size (must be odd)
        sigma: Standard deviation
        
    Returns:
        Blurred image
    """
    try:
        if not validate_image(image):
            return image
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        return blurred
        
    except Exception as e:
        logger.error(f"Error applying Gaussian blur: {str(e)}")
        return image


def sharpen_image(
    image: np.ndarray,
    amount: float = 1.0
) -> np.ndarray:
    """
    Sharpen image using unsharp mask
    
    Args:
        image: Image as numpy array
        amount: Sharpening amount (0.0 to 2.0)
        
    Returns:
        Sharpened image
    """
    try:
        if not validate_image(image):
            return image
        
        # Create blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Sharpen using weighted addition
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        
        return sharpened
        
    except Exception as e:
        logger.error(f"Error sharpening image: {str(e)}")
        return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range
    
    Args:
        image: Image as numpy array
        
    Returns:
        Normalized image
    """
    try:
        if not validate_image(image):
            return image
        
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized.astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        return image


# ============================================================================
# Color Conversion
# ============================================================================

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB
    
    Args:
        image: Image in BGR format
        
    Returns:
        Image in RGB format
    """
    try:
        if not validate_image(image):
            return image
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        logger.error(f"Error converting BGR to RGB: {str(e)}")
        return image


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR
    
    Args:
        image: Image in RGB format
        
    Returns:
        Image in BGR format
    """
    try:
        if not validate_image(image):
            return image
        
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        logger.error(f"Error converting RGB to BGR: {str(e)}")
        return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale
    
    Args:
        image: Color image
        
    Returns:
        Grayscale image
    """
    try:
        if not validate_image(image):
            return image
        
        if len(image.shape) == 2:
            return image  # Already grayscale
        
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return grayscale
        
    except Exception as e:
        logger.error(f"Error converting to grayscale: {str(e)}")
        return image


# ============================================================================
# Image Drawing Utilities
# ============================================================================

def draw_rectangle(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw rectangle on image
    
    Args:
        image: Image as numpy array
        x1, y1: Top-left coordinates
        x2, y2: Bottom-right coordinates
        color: Rectangle color (BGR)
        thickness: Line thickness (-1 for filled)
        
    Returns:
        Image with rectangle drawn
    """
    try:
        result = image.copy()
        cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return result
        
    except Exception as e:
        logger.error(f"Error drawing rectangle: {str(e)}")
        return image


def draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    bg_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Draw text on image with optional background
    
    Args:
        image: Image as numpy array
        text: Text to draw
        x, y: Text position (bottom-left corner)
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness
        bg_color: Background color (BGR) or None
        
    Returns:
        Image with text drawn
    """
    try:
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background if specified
        if bg_color is not None:
            cv2.rectangle(
                result,
                (x, y - text_height - baseline),
                (x + text_width, y + baseline),
                bg_color,
                -1
            )
        
        # Draw text
        cv2.putText(
            result,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error drawing text: {str(e)}")
        return image


def draw_circle(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw circle on image
    
    Args:
        image: Image as numpy array
        center_x, center_y: Circle center coordinates
        radius: Circle radius
        color: Circle color (BGR)
        thickness: Line thickness (-1 for filled)
        
    Returns:
        Image with circle drawn
    """
    try:
        result = image.copy()
        cv2.circle(result, (int(center_x), int(center_y)), int(radius), color, thickness)
        return result
        
    except Exception as e:
        logger.error(f"Error drawing circle: {str(e)}")
        return image


# ============================================================================
# File I/O
# ============================================================================

def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load image from file
    
    Args:
        file_path: Path to image file
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        image = cv2.imread(file_path)
        
        if image is None:
            logger.error(f"Failed to load image: {file_path}")
            return None
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None


def save_image(
    image: np.ndarray,
    file_path: str,
    quality: int = JPEG_QUALITY
) -> bool:
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        file_path: Output file path
        quality: JPEG quality (0-100)
        
    Returns:
        Boolean indicating success
    """
    try:
        if not validate_image(image):
            return False
        
        # Create directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Set encoding parameters
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif file_path.lower().endswith('.png'):
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
        else:
            encode_param = []
        
        success = cv2.imwrite(file_path, image, encode_param)
        
        if success:
            logger.debug(f"Image saved: {file_path}")
        else:
            logger.error(f"Failed to save image: {file_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return False


# ============================================================================
# Batch Processing
# ============================================================================

def process_images_batch(
    images: List[np.ndarray],
    operation: callable,
    **kwargs
) -> List[np.ndarray]:
    """
    Apply operation to batch of images
    
    Args:
        images: List of images
        operation: Function to apply to each image
        **kwargs: Additional arguments for operation
        
    Returns:
        List of processed images
    """
    try:
        processed = []
        
        for image in images:
            result = operation(image, **kwargs)
            processed.append(result)
        
        return processed
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return images