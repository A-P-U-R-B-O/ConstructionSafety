"""
Model Configuration Module
Centralized configuration for YOLO models, danger classifications, and detection settings
Supports loading custom configurations and dynamic updates
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import json
import os
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DangerLevel(str, Enum):
    """Enumeration of danger levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


class ObjectCategory(str, Enum):
    """Enumeration of object categories"""
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


class ModelType(str, Enum):
    """YOLOv8 model types"""
    NANO = "yolov8n.pt"
    SMALL = "yolov8s.pt"
    MEDIUM = "yolov8m.pt"
    LARGE = "yolov8l.pt"
    XLARGE = "yolov8x.pt"


class DeviceType(str, Enum):
    """Inference device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class ClassConfig(BaseModel):
    """
    Configuration for a single detection class
    """
    danger_level: DangerLevel
    category: ObjectCategory
    requires_ppe: bool = False
    alert_message: Optional[str] = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    priority: int = Field(default=5, ge=1, le=10)
    description: Optional[str] = None
    
    class Config:
        use_enum_values = True


class AlertRule(BaseModel):
    """
    Configuration for alert generation rules
    """
    rule_name: str
    conditions: Dict[str, Any]
    alert_message: str
    severity: DangerLevel
    enabled: bool = True
    
    class Config:
        use_enum_values = True


class ModelConfiguration(BaseModel):
    """
    Main model configuration settings
    """
    model_type: ModelType = ModelType.NANO
    device: DeviceType = DeviceType.CPU
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=300)
    input_size: int = Field(default=640, ge=320, le=1280)
    half_precision: bool = False  # FP16 for faster inference
    augment: bool = False  # Test-time augmentation
    
    @validator('input_size')
    def validate_input_size(cls, v):
        """Ensure input size is multiple of 32"""
        if v % 32 != 0:
            raise ValueError("Input size must be multiple of 32")
        return v
    
    class Config:
        use_enum_values = True


class ConstructionSafetyConfig:
    """
    Main configuration manager for construction safety detection
    Handles loading, saving, and managing all detection configurations
    """
    
    # Default danger classifications for COCO classes
    DEFAULT_DANGER_CLASSES: Dict[str, ClassConfig] = {
        # Critical danger objects
        "person": ClassConfig(
            danger_level=DangerLevel.HIGH,
            category=ObjectCategory.PEOPLE,
            requires_ppe=True,
            alert_message="Worker detected without PPE verification",
            min_confidence=0.6,
            priority=10,
            description="Human worker on construction site"
        ),
        "truck": ClassConfig(
            danger_level=DangerLevel.CRITICAL,
            category=ObjectCategory.VEHICLE,
            requires_ppe=False,
            alert_message="Heavy vehicle detected in area",
            min_confidence=0.7,
            priority=10,
            description="Large construction vehicle"
        ),
        "car": ClassConfig(
            danger_level=DangerLevel.HIGH,
            category=ObjectCategory.VEHICLE,
            requires_ppe=False,
            alert_message="Vehicle detected in construction zone",
            min_confidence=0.6,
            priority=8,
            description="Passenger vehicle"
        ),
        "motorcycle": ClassConfig(
            danger_level=DangerLevel.HIGH,
            category=ObjectCategory.VEHICLE,
            requires_ppe=False,
            alert_message="Motorcycle detected",
            min_confidence=0.6,
            priority=7,
            description="Two-wheeled vehicle"
        ),
        "bus": ClassConfig(
            danger_level=DangerLevel.HIGH,
            category=ObjectCategory.VEHICLE,
            requires_ppe=False,
            alert_message="Large vehicle in area",
            min_confidence=0.7,
            priority=9,
            description="Bus or large passenger vehicle"
        ),
        
        # Sharp and dangerous objects
        "knife": ClassConfig(
            danger_level=DangerLevel.CRITICAL,
            category=ObjectCategory.SHARP_OBJECT,
            requires_ppe=True,
            alert_message="Sharp object detected - exercise caution",
            min_confidence=0.7,
            priority=9,
            description="Knife or blade"
        ),
        "scissors": ClassConfig(
            danger_level=DangerLevel.MEDIUM,
            category=ObjectCategory.SHARP_OBJECT,
            requires_ppe=False,
            alert_message="Scissors detected",
            min_confidence=0.5,
            priority=5,
            description="Cutting tool"
        ),
        
        # Obstacles and equipment
        "chair": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.OBSTACLE,
            requires_ppe=False,
            alert_message="Obstacle in path",
            min_confidence=0.5,
            priority=3,
            description="Seating furniture"
        ),
        "bench": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.OBSTACLE,
            requires_ppe=False,
            alert_message="Bench obstacle detected",
            min_confidence=0.5,
            priority=3,
            description="Bench seating"
        ),
        "couch": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.OBSTACLE,
            requires_ppe=False,
            alert_message="Large obstacle in area",
            min_confidence=0.5,
            priority=4,
            description="Large furniture"
        ),
        
        # Containers and bottles
        "bottle": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.CONTAINER,
            requires_ppe=False,
            alert_message="Container detected",
            min_confidence=0.4,
            priority=2,
            description="Bottle container"
        ),
        "cup": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.CONTAINER,
            requires_ppe=False,
            alert_message="Cup detected",
            min_confidence=0.4,
            priority=2,
            description="Drinking cup"
        ),
        "wine glass": ClassConfig(
            danger_level=DangerLevel.MEDIUM,
            category=ObjectCategory.CONTAINER,
            requires_ppe=False,
            alert_message="Glass container - breakage hazard",
            min_confidence=0.5,
            priority=4,
            description="Glass container"
        ),
        
        # Equipment
        "backpack": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Equipment bag detected",
            min_confidence=0.5,
            priority=3,
            description="Personal bag"
        ),
        "handbag": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Personal item detected",
            min_confidence=0.5,
            priority=2,
            description="Handbag or purse"
        ),
        "suitcase": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.OBSTACLE,
            requires_ppe=False,
            alert_message="Large bag obstacle",
            min_confidence=0.5,
            priority=3,
            description="Luggage container"
        ),
        "umbrella": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Umbrella detected",
            min_confidence=0.4,
            priority=2,
            description="Rain protection"
        ),
        
        # Electronics
        "laptop": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Electronic equipment present",
            min_confidence=0.6,
            priority=3,
            description="Laptop computer"
        ),
        "cell phone": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Mobile device detected",
            min_confidence=0.5,
            priority=2,
            description="Mobile phone"
        ),
        "keyboard": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Computer equipment present",
            min_confidence=0.5,
            priority=2,
            description="Computer keyboard"
        ),
        "mouse": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Computer peripheral detected",
            min_confidence=0.4,
            priority=2,
            description="Computer mouse"
        ),
        
        # Animals (unexpected on construction sites)
        "bird": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.ANIMAL,
            requires_ppe=False,
            alert_message="Bird in area",
            min_confidence=0.5,
            priority=2,
            description="Avian animal"
        ),
        "cat": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.ANIMAL,
            requires_ppe=False,
            alert_message="Cat detected on site",
            min_confidence=0.6,
            priority=3,
            description="Feline animal"
        ),
        "dog": ClassConfig(
            danger_level=DangerLevel.MEDIUM,
            category=ObjectCategory.ANIMAL,
            requires_ppe=False,
            alert_message="Dog present - possible hazard",
            min_confidence=0.6,
            priority=5,
            description="Canine animal"
        ),
        "horse": ClassConfig(
            danger_level=DangerLevel.MEDIUM,
            category=ObjectCategory.ANIMAL,
            requires_ppe=False,
            alert_message="Large animal detected",
            min_confidence=0.7,
            priority=6,
            description="Equine animal"
        ),
        
        # Signs and safety equipment
        "fire hydrant": ClassConfig(
            danger_level=DangerLevel.MEDIUM,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Fire safety equipment nearby",
            min_confidence=0.7,
            priority=6,
            description="Fire hydrant"
        ),
        "stop sign": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.SIGN,
            requires_ppe=False,
            alert_message="Traffic sign detected",
            min_confidence=0.7,
            priority=4,
            description="Stop traffic sign"
        ),
        "parking meter": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Parking equipment detected",
            min_confidence=0.6,
            priority=2,
            description="Parking meter"
        ),
        
        # Sports equipment (unusual on construction sites)
        "sports ball": ClassConfig(
            danger_level=DangerLevel.LOW,
            category=ObjectCategory.EQUIPMENT,
            requires_ppe=False,
            alert_message="Ball detected",
            min_confidence=0.5,
            priority=2,
            description="Sports ball"
        ),
        "baseball bat": ClassConfig(
            danger_level=DangerLevel.MEDIUM,
            category=ObjectCategory.TOOL,
            requires_ppe=False,
            alert_message="Bat-like object detected",
            min_confidence=0.6,
            priority=5,
            description="Baseball bat or similar"
        ),
        
        # Bicycles and skateboards
        "bicycle": ClassConfig(
            danger_level=DangerLevel.MEDIUM,
            category=ObjectCategory.VEHICLE,
            requires_ppe=False,
            alert_message="Bicycle in construction zone",
            min_confidence=0.6,
            priority=5,
            description="Two-wheeled bicycle"
        ),
        "skateboard": ClassConfig(
            danger_level=DangerLevel.MEDIUM,
            category=ObjectCategory.VEHICLE,
            requires_ppe=False,
            alert_message="Skateboard detected",
            min_confidence=0.5,
            priority=4,
            description="Skateboard"
        ),
    }
    
    # Default alert rules
    DEFAULT_ALERT_RULES: List[AlertRule] = [
        AlertRule(
            rule_name="multiple_workers",
            conditions={"min_persons": 5},
            alert_message="⚠️ ALERT: Multiple workers detected - ensure proper supervision",
            severity=DangerLevel.MEDIUM,
            enabled=True
        ),
        AlertRule(
            rule_name="vehicle_near_workers",
            conditions={"vehicle_detected": True, "person_detected": True, "max_distance": 100},
            alert_message="🚨 CRITICAL: Vehicle detected near workers!",
            severity=DangerLevel.CRITICAL,
            enabled=True
        ),
        AlertRule(
            rule_name="sharp_objects",
            conditions={"sharp_object_detected": True},
            alert_message="⚠️ WARNING: Sharp objects detected - exercise caution",
            severity=DangerLevel.HIGH,
            enabled=True
        ),
        AlertRule(
            rule_name="animal_on_site",
            conditions={"animal_detected": True},
            alert_message="🐕 NOTICE: Animal detected on construction site",
            severity=DangerLevel.MEDIUM,
            enabled=True
        ),
        AlertRule(
            rule_name="low_confidence_critical",
            conditions={"critical_object": True, "max_confidence": 0.7},
            alert_message="⚠️ WARNING: Low confidence critical object detection",
            severity=DangerLevel.MEDIUM,
            enabled=True
        ),
    ]
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        self.danger_classes: Dict[str, ClassConfig] = self.DEFAULT_DANGER_CLASSES.copy()
        self.alert_rules: List[AlertRule] = self.DEFAULT_ALERT_RULES.copy()
        self.model_config = ModelConfiguration()
        
        # Load custom configuration if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        
        logger.info(f"Configuration initialized with {len(self.danger_classes)} danger classes")
    
    def load_config(self, config_file: str):
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load danger classes
            if "danger_classes" in config_data:
                for class_name, class_data in config_data["danger_classes"].items():
                    self.danger_classes[class_name] = ClassConfig(**class_data)
            
            # Load alert rules
            if "alert_rules" in config_data:
                self.alert_rules = [AlertRule(**rule) for rule in config_data["alert_rules"]]
            
            # Load model configuration
            if "model_config" in config_data:
                self.model_config = ModelConfiguration(**config_data["model_config"])
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {str(e)}")
            raise
    
    def save_config(self, config_file: str):
        """
        Save current configuration to JSON file
        
        Args:
            config_file: Path to save configuration file
        """
        try:
            config_data = {
                "danger_classes": {
                    name: config.dict() for name, config in self.danger_classes.items()
                },
                "alert_rules": [rule.dict() for rule in self.alert_rules],
                "model_config": self.model_config.dict()
            }
            
            # Create directory if it doesn't exist
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {str(e)}")
            raise
    
    def get_class_config(self, class_name: str) -> Optional[ClassConfig]:
        """
        Get configuration for a specific class
        
        Args:
            class_name: Name of the class
            
        Returns:
            ClassConfig object or None if not found
        """
        return self.danger_classes.get(class_name)
    
    def add_class_config(self, class_name: str, config: ClassConfig):
        """
        Add or update class configuration
        
        Args:
            class_name: Name of the class
            config: ClassConfig object
        """
        self.danger_classes[class_name] = config
        logger.info(f"Added/updated configuration for class: {class_name}")
    
    def remove_class_config(self, class_name: str) -> bool:
        """
        Remove class configuration
        
        Args:
            class_name: Name of the class to remove
            
        Returns:
            Boolean indicating success
        """
        if class_name in self.danger_classes:
            del self.danger_classes[class_name]
            logger.info(f"Removed configuration for class: {class_name}")
            return True
        return False
    
    def get_classes_by_danger_level(self, danger_level: DangerLevel) -> List[str]:
        """
        Get all classes with specific danger level
        
        Args:
            danger_level: DangerLevel to filter by
            
        Returns:
            List of class names
        """
        return [
            name for name, config in self.danger_classes.items()
            if config.danger_level == danger_level
        ]
    
    def get_classes_by_category(self, category: ObjectCategory) -> List[str]:
        """
        Get all classes in specific category
        
        Args:
            category: ObjectCategory to filter by
            
        Returns:
            List of class names
        """
        return [
            name
