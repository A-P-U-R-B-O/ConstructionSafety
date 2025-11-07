"""
Danger Classifier Service Module
Advanced danger assessment and classification for construction safety
Analyzes spatial relationships, temporal patterns, and context for intelligent risk evaluation
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from app.core.model_config import get_config, DangerLevel, ObjectCategory
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RiskZone(str, Enum):
    """Risk zones in construction site"""
    CRITICAL_ZONE = "critical_zone"
    HIGH_RISK_ZONE = "high_risk_zone"
    MODERATE_ZONE = "moderate_zone"
    LOW_RISK_ZONE = "low_risk_zone"
    SAFE_ZONE = "safe_zone"


class InteractionType(str, Enum):
    """Types of object interactions"""
    COLLISION_RISK = "collision_risk"
    PROXIMITY_WARNING = "proximity_warning"
    WORKING_TOGETHER = "working_together"
    SAFE_DISTANCE = "safe_distance"
    NO_INTERACTION = "no_interaction"


@dataclass
class SpatialRelationship:
    """
    Represents spatial relationship between two objects
    """
    object1: Dict[str, Any]
    object2: Dict[str, Any]
    distance: float
    relative_position: str  # "above", "below", "left", "right", "overlapping"
    interaction_type: InteractionType
    risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object1": self.object1["class"],
            "object2": self.object2["class"],
            "distance": round(self.distance, 2),
            "relative_position": self.relative_position,
            "interaction_type": self.interaction_type.value,
            "risk_score": round(self.risk_score, 2)
        }


@dataclass
class DangerZone:
    """
    Represents a dangerous zone in the frame
    """
    zone_id: str
    center_x: float
    center_y: float
    radius: float
    risk_level: RiskZone
    objects: List[Dict[str, Any]]
    risk_score: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "center": {"x": round(self.center_x, 2), "y": round(self.center_y, 2)},
            "radius": round(self.radius, 2),
            "risk_level": self.risk_level.value,
            "object_count": len(self.objects),
            "objects": [obj["class"] for obj in self.objects],
            "risk_score": round(self.risk_score, 2),
            "description": self.description
        }


class DangerClassifier:
    """
    Main classifier for danger assessment and risk evaluation
    """
    
    # Distance thresholds (in pixels)
    CRITICAL_DISTANCE = 50
    HIGH_RISK_DISTANCE = 100
    MODERATE_DISTANCE = 200
    SAFE_DISTANCE = 300
    
    # Risk score weights
    WEIGHTS = {
        "danger_level": 0.35,
        "proximity": 0.25,
        "object_count": 0.15,
        "interaction": 0.15,
        "temporal": 0.10
    }
    
    # Dangerous combinations (higher risk when together)
    DANGEROUS_COMBINATIONS = {
        ("person", "truck"): 10.0,
        ("person", "car"): 8.0,
        ("person", "motorcycle"): 7.0,
        ("person", "knife"): 9.5,
        ("person", "scissors"): 6.0,
        ("truck", "car"): 5.0,
        ("person", "dog"): 6.5,
    }
    
    def __init__(self, history_size: int = 100):
        """
        Initialize danger classifier
        
        Args:
            history_size: Size of temporal history to maintain
        """
        self.config = get_config()
        self.history_size = history_size
        
        # Temporal tracking
        self.detection_history = deque(maxlen=history_size)
        self.danger_zone_history = deque(maxlen=history_size)
        self.interaction_history = defaultdict(list)
        
        # Statistics
        self.total_assessments = 0
        self.critical_situations = 0
        self.high_risk_situations = 0
        
        logger.info("DangerClassifier initialized")
    
    def assess_danger(
        self,
        detections: List[Dict[str, Any]],
        frame_dimensions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive danger assessment on detections
        
        Args:
            detections: List of detection dictionaries
            frame_dimensions: Optional frame dimensions for spatial analysis
            
        Returns:
            Comprehensive danger assessment results
        """
        if not detections:
            return self._empty_assessment()
        
        try:
            # 1. Calculate base danger score
            base_score = self._calculate_base_danger_score(detections)
            
            # 2. Analyze spatial relationships
            spatial_analysis = self._analyze_spatial_relationships(detections)
            
            # 3. Identify danger zones
            danger_zones = self._identify_danger_zones(detections, frame_dimensions)
            
            # 4. Analyze object interactions
            interactions = self._analyze_interactions(detections)
            
            # 5. Assess temporal patterns
            temporal_score = self._assess_temporal_patterns(detections)
            
            # 6. Calculate comprehensive risk score
            comprehensive_score = self._calculate_comprehensive_score(
                base_score=base_score,
                spatial_score=spatial_analysis["risk_score"],
                interaction_score=interactions["risk_score"],
                temporal_score=temporal_score
            )
            
            # 7. Determine overall risk level
            risk_level = self._determine_risk_level(comprehensive_score)
            
            # 8. Generate risk recommendations
            recommendations = self._generate_recommendations(
                detections=detections,
                spatial_analysis=spatial_analysis,
                danger_zones=danger_zones,
                interactions=interactions,
                risk_level=risk_level
            )
            
            # Update statistics
            self.total_assessments += 1
            if risk_level == RiskZone.CRITICAL_ZONE:
                self.critical_situations += 1
            elif risk_level == RiskZone.HIGH_RISK_ZONE:
                self.high_risk_situations += 1
            
            # Store in history
            self._update_history(detections, danger_zones)
            
            assessment = {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "risk_level": risk_level.value,
                "comprehensive_score": round(comprehensive_score, 2),
                "base_danger_score": round(base_score, 2),
                "spatial_risk_score": round(spatial_analysis["risk_score"], 2),
                "interaction_risk_score": round(interactions["risk_score"], 2),
                "temporal_risk_score": round(temporal_score, 2),
                "danger_zones": [zone.to_dict() for zone in danger_zones],
                "spatial_relationships": [rel.to_dict() for rel in spatial_analysis["relationships"]],
                "dangerous_interactions": interactions["dangerous_interactions"],
                "recommendations": recommendations,
                "statistics": {
                    "total_objects": len(detections),
                    "critical_objects": len([d for d in detections if d["danger_level"] == "critical"]),
                    "high_danger_objects": len([d for d in detections if d["danger_level"] == "high"]),
                    "danger_zone_count": len(danger_zones),
                    "risky_interactions": len(interactions["dangerous_interactions"])
                }
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in danger assessment: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _calculate_base_danger_score(self, detections: List[Dict[str, Any]]) -> float:
        """
        Calculate base danger score from object danger levels
        
        Args:
            detections: List of detections
            
        Returns:
            Base danger score (0-10)
        """
        if not detections:
            return 0.0
        
        danger_scores = {
            "critical": 10,
            "high": 7,
            "medium": 4,
            "low": 2,
            "safe": 0
        }
        
        total_score = sum(
            danger_scores.get(d["danger_level"], 0) * d["confidence"]
            for d in detections
        )
        
        # Normalize to 0-10 scale
        max_possible = len(detections) * 10
        normalized = (total_score / max_possible) * 10 if max_possible > 0 else 0
        
        return min(normalized, 10.0)
    
    def _analyze_spatial_relationships(
        self,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze spatial relationships between objects
        
        Args:
            detections: List of detections
            
        Returns:
            Spatial analysis results
        """
        relationships = []
        risk_scores = []
        
        # Compare each pair of objects
        for i, obj1 in enumerate(detections):
            for obj2 in detections[i + 1:]:
                relationship = self._calculate_spatial_relationship(obj1, obj2)
                relationships.append(relationship)
                risk_scores.append(relationship.risk_score)
        
        avg_risk = np.mean(risk_scores) if risk_scores else 0.0
        max_risk = max(risk_scores) if risk_scores else 0.0
        
        return {
            "relationships": relationships,
            "total_relationships": len(relationships),
            "risk_score": avg_risk,
            "max_risk": max_risk,
            "high_risk_pairs": len([r for r in relationships if r.risk_score > 7.0])
        }
    
    def _calculate_spatial_relationship(
        self,
        obj1: Dict[str, Any],
        obj2: Dict[str, Any]
    ) -> SpatialRelationship:
        """
        Calculate spatial relationship between two objects
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            SpatialRelationship object
        """
        # Calculate center points
        center1 = (obj1["bbox"]["center_x"], obj1["bbox"]["center_y"])
        center2 = (obj2["bbox"]["center_x"], obj2["bbox"]["center_y"])
        
        # Calculate Euclidean distance
        distance = np.sqrt(
            (center1[0] - center2[0]) ** 2 +
            (center1[1] - center2[1]) ** 2
        )
        
        # Determine relative position
        relative_position = self._get_relative_position(obj1, obj2)
        
        # Determine interaction type based on distance
        if distance < self.CRITICAL_DISTANCE:
            interaction_type = InteractionType.COLLISION_RISK
        elif distance < self.HIGH_RISK_DISTANCE:
            interaction_type = InteractionType.PROXIMITY_WARNING
        elif distance < self.MODERATE_DISTANCE:
            interaction_type = InteractionType.WORKING_TOGETHER
        elif distance < self.SAFE_DISTANCE:
            interaction_type = InteractionType.SAFE_DISTANCE
        else:
            interaction_type = InteractionType.NO_INTERACTION
        
        # Calculate risk score
        risk_score = self._calculate_relationship_risk(
            obj1, obj2, distance, interaction_type
        )
        
        return SpatialRelationship(
            object1=obj1,
            object2=obj2,
            distance=distance,
            relative_position=relative_position,
            interaction_type=interaction_type,
            risk_score=risk_score
        )
    
    def _get_relative_position(
        self,
        obj1: Dict[str, Any],
        obj2: Dict[str, Any]
    ) -> str:
        """
        Determine relative position between objects
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            Relative position string
        """
        center1 = (obj1["bbox"]["center_x"], obj1["bbox"]["center_y"])
        center2 = (obj2["bbox"]["center_x"], obj2["bbox"]["center_y"])
        
        # Check for overlap
        bbox1 = obj1["bbox"]
        bbox2 = obj2["bbox"]
        
        overlap_x = not (bbox1["x2"] < bbox2["x1"] or bbox2["x2"] < bbox1["x1"])
        overlap_y = not (bbox1["y2"] < bbox2["y1"] or bbox2["y2"] < bbox1["y1"])
        
        if overlap_x and overlap_y:
            return "overlapping"
        
        # Determine direction
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "below" if dy > 0 else "above"
    
    def _calculate_relationship_risk(
        self,
        obj1: Dict[str, Any],
        obj2: Dict[str, Any],
        distance: float,
        interaction_type: InteractionType
    ) -> float:
        """
        Calculate risk score for object relationship
        
        Args:
            obj1: First object
            obj2: Second object
            distance: Distance between objects
            interaction_type: Type of interaction
            
        Returns:
            Risk score (0-10)
        """
        # Base risk from danger levels
        danger_scores = {"critical": 10, "high": 7, "medium": 4, "low": 2, "safe": 0}
        base_risk = (
            danger_scores.get(obj1["danger_level"], 0) +
            danger_scores.get(obj2["danger_level"], 0)
        ) / 2
        
        # Distance modifier (closer = more dangerous)
        if distance < self.CRITICAL_DISTANCE:
            distance_modifier = 2.0
        elif distance < self.HIGH_RISK_DISTANCE:
            distance_modifier = 1.5
        elif distance < self.MODERATE_DISTANCE:
            distance_modifier = 1.2
        else:
            distance_modifier = 1.0
        
        # Check for dangerous combinations
        combo_key = tuple(sorted([obj1["class"], obj2["class"]]))
        combination_modifier = self.DANGEROUS_COMBINATIONS.get(combo_key, 1.0)
        
        # Calculate final risk
        risk = base_risk * distance_modifier * (combination_modifier / 10.0)
        
        return min(risk, 10.0)
    
    def _identify_danger_zones(
        self,
        detections: List[Dict[str, Any]],
        frame_dimensions: Optional[Dict[str, Any]] = None
    ) -> List[DangerZone]:
        """
        Identify danger zones in the frame
        
        Args:
            detections: List of detections
            frame_dimensions: Frame dimensions
            
        Returns:
            List of DangerZone objects
        """
        danger_zones = []
        
        # Group dangerous objects
        critical_objects = [d for d in detections if d["danger_level"] == "critical"]
        high_danger_objects = [d for d in detections if d["danger_level"] == "high"]
        
        # Create zones around critical objects
        for idx, obj in enumerate(critical_objects):
            zone = DangerZone(
                zone_id=f"critical_{idx}",
                center_x=obj["bbox"]["center_x"],
                center_y=obj["bbox"]["center_y"],
                radius=self.HIGH_RISK_DISTANCE,
                risk_level=RiskZone.CRITICAL_ZONE,
                objects=[obj],
                risk_score=10.0,
                description=f"Critical danger zone around {obj['class']}"
            )
            danger_zones.append(zone)
        
        # Create zones for clusters of high-danger objects
        if len(high_danger_objects) >= 2:
            zone = self._create_cluster_zone(high_danger_objects, "high_danger")
            if zone:
                danger_zones.append(zone)
        
        return danger_zones
    
    def _create_cluster_zone(
        self,
        objects: List[Dict[str, Any]],
        zone_type: str
    ) -> Optional[DangerZone]:
        """
        Create danger zone for object cluster
        
        Args:
            objects: List of objects in cluster
            zone_type: Type of zone
            
        Returns:
            DangerZone object or None
        """
        if not objects:
            return None
        
        # Calculate cluster center
        centers = [(obj["bbox"]["center_x"], obj["bbox"]["center_y"]) for obj in objects]
        avg_x = np.mean([c[0] for c in centers])
        avg_y = np.mean([c[1] for c in centers])
        
        # Calculate radius (max distance from center)
        distances = [
            np.sqrt((c[0] - avg_x) ** 2 + (c[1] - avg_y) ** 2)
            for c in centers
        ]
        radius = max(distances) + self.MODERATE_DISTANCE
        
        return DangerZone(
            zone_id=f"{zone_type}_cluster",
            center_x=avg_x,
            center_y=avg_y,
            radius=radius,
            risk_level=RiskZone.HIGH_RISK_ZONE,
            objects=objects,
            risk_score=7.5,
            description=f"Cluster of {len(objects)} dangerous objects"
        )
    
    def _analyze_interactions(
        self,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze dangerous interactions between objects
        
        Args:
            detections: List of detections
            
        Returns:
            Interaction analysis results
        """
        dangerous_interactions = []
        risk_scores = []
        
        # Check for dangerous combinations
        for i, obj1 in enumerate(detections):
            for obj2 in detections[i + 1:]:
                combo_key = tuple(sorted([obj1["class"], obj2["class"]]))
                
                if combo_key in self.DANGEROUS_COMBINATIONS:
                    # Calculate distance
                    center1 = (obj1["bbox"]["center_x"], obj1["bbox"]["center_y"])
                    center2 = (obj2["bbox"]["center_x"], obj2["bbox"]["center_y"])
                    distance = np.sqrt(
                        (center1[0] - center2[0]) ** 2 +
                        (center1[1] - center2[1]) ** 2
                    )
                    
                    # Calculate risk based on distance
                    base_risk = self.DANGEROUS_COMBINATIONS[combo_key]
                    if distance < self.CRITICAL_DISTANCE:
                        risk = base_risk
                    elif distance < self.HIGH_RISK_DISTANCE:
                        risk = base_risk * 0.8
                    elif distance < self.MODERATE_DISTANCE:
                        risk = base_risk * 0.5
                    else:
                        risk = base_risk * 0.3
                    
                    dangerous_interactions.append({
                        "object1": obj1["class"],
                        "object2": obj2["class"],
                        "distance": round(distance, 2),
                        "base_risk": base_risk,
                        "actual_risk": round(risk, 2),
                        "warning": f"Dangerous combination: {obj1['class']} near {obj2['class']}"
                    })
                    
                    risk_scores.append(risk)
        
        avg_risk = np.mean(risk_scores) if risk_scores else 0.0
        
        return {
            "dangerous_interactions": dangerous_interactions,
            "interaction_count": len(dangerous_interactions),
            "risk_score": avg_risk
        }
    
    def _assess_temporal_patterns(
        self,
        detections: List[Dict[str, Any]]
    ) -> float:
        """
        Assess temporal patterns and trends
        
        Args:
            detections: Current detections
            
        Returns:
            Temporal risk score (0-10)
        """
        if len(self.detection_history) < 5:
            return 0.0  # Not enough history
        
        # Count dangerous objects over time
        current_dangerous = len([
            d for d in detections
            if d["danger_level"] in ["critical", "high"]
        ])
        
        historical_dangerous = [
            len([d for d in frame if d["danger_level"] in ["critical", "high"]])
            for frame in self.detection_history
        ]
        
        # Check for increasing trend
        if len(historical_dangerous) >= 5:
            recent_avg = np.mean(historical_dangerous[-5:])
            older_avg = np.mean(historical_dangerous[-10:-5]) if len(historical_dangerous) >= 10 else recent_avg
            
            if current_dangerous > recent_avg * 1.5:
                return 8.0  # Sudden increase
            elif current_dangerous > recent_avg:
                return 5.0  # Gradual increase
        
        return 2.0  # Stable
    
    def _calculate_comprehensive_score(
        self,
        base_score: float,
        spatial_score: float,
        interaction_score: float,
        temporal_score: float
    ) -> float:
        """
        Calculate comprehensive risk score using weighted combination
        
        Args:
            base_score: Base danger score
            spatial_score: Spatial relationship score
            interaction_score: Interaction risk score
            temporal_score: Temporal pattern score
            
        Returns:
            Comprehensive score (0-10)
        """
        comprehensive = (
            base_score * self.WEIGHTS["danger_level"] +
            spatial_score * self.WEIGHTS["proximity"] +
            interaction_score * self.WEIGHTS["interaction"] +
            temporal_score * self.WEIGHTS["temporal"]
        )
        
        # Boost score if multiple high scores
        high_scores = sum([
            1 for score in [base_score, spatial_score, interaction_score]
            if score > 7.0
        ])
        
        if high_scores >= 2:
            comprehensive *= 1.2  # 20% boost
        
        return min(comprehensive, 10.0)
    
    def _determine_risk_level(self, score: float) -> RiskZone:
        """
        Determine risk level from score
        
        Args:
            score: Risk score (0-10)
            
        Returns:
            RiskZone enum
        """
        if score >= 8.5:
            return RiskZone.CRITICAL_ZONE
        elif score >= 6.5:
            return RiskZone.HIGH_RISK_ZONE
        elif score >= 4.0:
            return RiskZone.MODERATE_ZONE
        elif score >= 2.0:
            return RiskZone.LOW_RISK_ZONE
        else:
            return RiskZone.SAFE_ZONE
    
    def _generate_recommendations(
        self,
        detections: List[Dict[str, Any]],
        spatial_analysis: Dict[str, Any],
        danger_zones: List[DangerZone],
        interactions: Dict[str, Any],
        risk_level: RiskZone
    ) -> List[str]:
        """
        Generate safety recommendations
        
        Args:
            detections: List of detections
            spatial_analysis: Spatial analysis results
            danger_zones: List of danger zones
            interactions: Interaction analysis
            risk_level: Overall risk level
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Risk level specific recommendations
        if risk_level == RiskZone.CRITICAL_ZONE:
            recommendations.append("🚨 CRITICAL: Evacuate area immediately")
            recommendations.append("🚨 Stop all work activities until hazards are addressed")
        elif risk_level == RiskZone.HIGH_RISK_ZONE:
            recommendations.append("⚠️ HIGH RISK: Exercise extreme caution")
            recommendations.append("⚠️ Increase supervision and safety monitoring")
        
        # Object-specific recommendations
        critical_objects = [d for d in detections if d["danger_level"] == "critical"]
        if critical_objects:
            recommendations.append(
                f"⚠️ {len(critical_objects)} critical hazard(s) detected - immediate attention required"
            )
        
        # Proximity recommendations
        if spatial_analysis["high_risk_pairs"] > 0:
            recommendations.append(
                f"⚠️ {spatial_analysis['high_risk_pairs']} dangerous proximity situation(s) detected"
            )
        
        # Interaction recommendations
        if interactions["interaction_count"] > 0:
            recommendations.append(
                f"⚠️ {interactions['interaction_count']} dangerous object combination(s) present"
            )
        
        # Danger zone recommendations
        if len(danger_zones) > 0:
            recommendations.append(
                f"⚠️ {len(danger_zones)} danger zone(s) identified - maintain safe distance"
            )
        
        # PPE recommendations
        people = [d for d in detections if d["category"] == "people"]
        if people:
            recommendations.append(
                "✓ Verify all workers are wearing appropriate PPE (hard hat, safety vest, gloves)"
            )
        
        # Vehicle recommendations
        vehicles = [d for d in detections if d["category"] == "vehicle"]
        if vehicles and people:
            recommendations.append(
                "✓ Ensure workers maintain safe distance from vehicles"
            )
            recommendations.append(
                "✓ Use spotters for vehicle operations"
            )
        
        if not recommendations:
            recommendations.append("✓ Continue normal safety protocols")
        
        return recommendations
    
    def _update_history(
        self,
        detections: List[Dict[str, Any]],
        danger_zones: List[DangerZone]
    ):
        """
        Update temporal history
        
        Args:
            detections: Current detections
            danger_zones: Current danger zones
        """
        self.detection_history.append(detections)
        self.danger_zone_history.append(danger_zones)
    
    def _empty_assessment(self) -> Dict[str, Any]:
        """
        Return empty assessment for no detections
        
        Returns:
            Empty assessment dictionary
        """
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "risk_level": RiskZone.SAFE_ZONE.value,
            "comprehensive_score": 0.0,
            "danger_zones": [],
            "spatial_relationships": [],
            "dangerous_interactions": [],
            "recommendations": ["✓ No hazards detected - maintain standard safety protocols"],
            "statistics": {
                "total_objects": 0,
                "critical_objects": 0,
                "high_danger_objects": 0,
                "danger_zone_count": 0,
                "risky_interactions": 0
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classifier statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_assessments": self.total_assessments,
            "critical_situations": self.critical_situations,
            "high_risk_situations": self.high_risk_situations,
            "history_size": len(self.detection_history),
            "critical_rate": (
                self.critical_situations / self.total_assessments
                if self.total_assessments > 0 else 0
            )
        }
    
    def clear_history(self):
        """Clear temporal history"""
        self.detection_history.clear()
        self.danger_zone_history.clear()
        self.interaction_history.clear()
        logger.info("Danger classifier history cleared")
    
    def __repr__(self) -> str:
        return (
            f"DangerClassifier("
            f"assessments={self.total_assessments}, "
            f"critical={self.critical_situations}, "
            f"high_risk={self.high_risk_situations})"
        )