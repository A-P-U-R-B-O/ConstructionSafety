"""
Alert Service Module
Intelligent alert generation and management for construction safety
Handles alert rules, severity assessment, notification, and alert history
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json

from app.core.model_config import get_config, DangerLevel, AlertRule
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    """Types of alerts"""
    DANGER_DETECTION = "danger_detection"
    PROXIMITY_WARNING = "proximity_warning"
    PPE_VIOLATION = "ppe_violation"
    MULTIPLE_HAZARDS = "multiple_hazards"
    VEHICLE_WARNING = "vehicle_warning"
    SHARP_OBJECT = "sharp_object"
    ANIMAL_DETECTED = "animal_detected"
    UNUSUAL_ACTIVITY = "unusual_activity"
    LOW_CONFIDENCE = "low_confidence"
    PATTERN_ALERT = "pattern_alert"


class Alert:
    """
    Alert data model
    """
    
    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.alert_id = self._generate_alert_id()
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.acknowledged = False
        self.resolved = False
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid
        return f"alert_{uuid.uuid4().hex[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }
    
    def acknowledge(self):
        """Mark alert as acknowledged"""
        self.acknowledged = True
    
    def resolve(self):
        """Mark alert as resolved"""
        self.resolved = True
    
    def __repr__(self) -> str:
        return f"Alert(id={self.alert_id}, type={self.alert_type.value}, severity={self.severity.value})"


class AlertHistory:
    """
    Manages alert history and statistics
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize alert history
        
        Args:
            max_history: Maximum number of alerts to store
        """
        self.max_history = max_history
        self.alerts = deque(maxlen=max_history)
        self.alert_counts = defaultdict(int)
        self.severity_counts = defaultdict(int)
        self.type_counts = defaultdict(int)
    
    def add(self, alert: Alert):
        """
        Add alert to history
        
        Args:
            alert: Alert object to add
        """
        self.alerts.append(alert)
        self.alert_counts["total"] += 1
        self.severity_counts[alert.severity.value] += 1
        self.type_counts[alert.alert_type.value] += 1
    
    def get_recent(self, count: int = 10) -> List[Alert]:
        """
        Get recent alerts
        
        Args:
            count: Number of recent alerts to return
            
        Returns:
            List of recent alerts
        """
        return list(self.alerts)[-count:]
    
    def get_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """
        Get alerts by severity
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of matching alerts
        """
        return [alert for alert in self.alerts if alert.severity == severity]
    
    def get_unacknowledged(self) -> List[Alert]:
        """
        Get all unacknowledged alerts
        
        Returns:
            List of unacknowledged alerts
        """
        return [alert for alert in self.alerts if not alert.acknowledged]
    
    def get_unresolved(self) -> List[Alert]:
        """
        Get all unresolved alerts
        
        Returns:
            List of unresolved alerts
        """
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_alerts": self.alert_counts["total"],
            "severity_distribution": dict(self.severity_counts),
            "type_distribution": dict(self.type_counts),
            "unacknowledged_count": len(self.get_unacknowledged()),
            "unresolved_count": len(self.get_unresolved()),
            "history_size": len(self.alerts)
        }
    
    def clear(self):
        """Clear all history"""
        self.alerts.clear()
        self.alert_counts.clear()
        self.severity_counts.clear()
        self.type_counts.clear()


class AlertService:
    """
    Main service for alert generation and management
    Analyzes detections and generates intelligent alerts
    """
    
    def __init__(
        self,
        enable_history: bool = True,
        cooldown_seconds: int = 5,
        max_alerts_per_frame: int = 5
    ):
        """
        Initialize alert service
        
        Args:
            enable_history: Whether to maintain alert history
            cooldown_seconds: Cooldown period for duplicate alerts
            max_alerts_per_frame: Maximum alerts to generate per frame
        """
        self.config = get_config()
        self.enable_history = enable_history
        self.history = AlertHistory() if enable_history else None
        self.cooldown_seconds = cooldown_seconds
        self.max_alerts_per_frame = max_alerts_per_frame
        
        # Cooldown tracking
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Pattern tracking
        self.detection_patterns = defaultdict(list)
        
        logger.info(f"AlertService initialized (cooldown={cooldown_seconds}s)")
    
    def generate_alerts(
        self,
        detections: List[Dict[str, Any]],
        severity_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts based on detections
        
        Args:
            detections: List of detection dictionaries
            severity_filter: Optional severity level filter (return only this severity or higher)
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if not detections:
            return alerts
        
        try:
            # 1. Generate danger detection alerts
            danger_alerts = self._generate_danger_alerts(detections)
            alerts.extend(danger_alerts)
            
            # 2. Generate proximity alerts
            proximity_alerts = self._generate_proximity_alerts(detections)
            alerts.extend(proximity_alerts)
            
            # 3. Generate PPE violation alerts
            ppe_alerts = self._generate_ppe_alerts(detections)
            alerts.extend(ppe_alerts)
            
            # 4. Generate vehicle warning alerts
            vehicle_alerts = self._generate_vehicle_alerts(detections)
            alerts.extend(vehicle_alerts)
            
            # 5. Generate multiple hazards alerts
            multiple_hazard_alerts = self._generate_multiple_hazard_alerts(detections)
            alerts.extend(multiple_hazard_alerts)
            
            # 6. Generate animal detection alerts
            animal_alerts = self._generate_animal_alerts(detections)
            alerts.extend(animal_alerts)
            
            # 7. Generate low confidence alerts
            low_conf_alerts = self._generate_low_confidence_alerts(detections)
            alerts.extend(low_conf_alerts)
            
            # 8. Apply custom alert rules
            rule_alerts = self._apply_alert_rules(detections)
            alerts.extend(rule_alerts)
            
            # Remove duplicate alerts
            alerts = self._deduplicate_alerts(alerts)
            
            # Apply cooldown
            alerts = self._apply_cooldown(alerts)
            
            # Sort by severity (critical first)
            alerts = self._sort_by_severity(alerts)
            
            # Limit number of alerts
            if len(alerts) > self.max_alerts_per_frame:
                alerts = alerts[:self.max_alerts_per_frame]
            
            # Filter by severity if requested
            if severity_filter:
                alerts = self._filter_by_severity(alerts, severity_filter)
            
            # Convert Alert objects to dictionaries and add to history
            alert_dicts = []
            for alert_data in alerts:
                alert = Alert(
                    alert_type=AlertType(alert_data["type"]),
                    severity=AlertSeverity(alert_data["severity"]),
                    message=alert_data["message"],
                    details=alert_data.get("details", {})
                )
                
                if self.enable_history and self.history:
                    self.history.add(alert)
                
                alert_dicts.append(alert.to_dict())
            
            return alert_dicts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}", exc_info=True)
            return []
    
    def _generate_danger_alerts(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts for dangerous objects
        
        Args:
            detections: List of detections
            
        Returns:
            List of danger alerts
        """
        alerts = []
        
        critical_objects = [d for d in detections if d["danger_level"] == "critical"]
        high_danger_objects = [d for d in detections if d["danger_level"] == "high"]
        
        # Critical danger alerts
        for obj in critical_objects:
            alerts.append({
                "type": AlertType.DANGER_DETECTION.value,
                "severity": AlertSeverity.CRITICAL.value,
                "message": f"🚨 CRITICAL DANGER: {obj['class']} detected!",
                "details": {
                    "object_class": obj["class"],
                    "confidence": obj["confidence"],
                    "danger_level": obj["danger_level"],
                    "position": obj["bbox"],
                    "description": obj.get("description", "")
                }
            })
        
        # High danger alerts
        for obj in high_danger_objects:
            alerts.append({
                "type": AlertType.DANGER_DETECTION.value,
                "severity": AlertSeverity.HIGH.value,
                "message": f"⚠️ HIGH DANGER: {obj['class']} detected!",
                "details": {
                    "object_class": obj["class"],
                    "confidence": obj["confidence"],
                    "danger_level": obj["danger_level"],
                    "position": obj["bbox"]
                }
            })
        
        return alerts
    
    def _generate_proximity_alerts(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate proximity warning alerts
        
        Args:
            detections: List of detections
            
        Returns:
            List of proximity alerts
        """
        alerts = []
        
        # Find people and vehicles
        people = [d for d in detections if d["category"] == "people"]
        vehicles = [d for d in detections if d["category"] == "vehicle"]
        
        if people and vehicles:
            # Check proximity between each person and vehicle
            for person in people:
                person_center = (
                    person["bbox"]["center_x"],
                    person["bbox"]["center_y"]
                )
                
                for vehicle in vehicles:
                    vehicle_center = (
                        vehicle["bbox"]["center_x"],
                        vehicle["bbox"]["center_y"]
                    )
                    
                    # Calculate distance
                    distance = (
                        (person_center[0] - vehicle_center[0]) ** 2 +
                        (person_center[1] - vehicle_center[1]) ** 2
                    ) ** 0.5
                    
                    # Alert if within 150 pixels (configurable threshold)
                    if distance < 150:
                        alerts.append({
                            "type": AlertType.PROXIMITY_WARNING.value,
                            "severity": AlertSeverity.CRITICAL.value,
                            "message": f"🚨 DANGER: Worker too close to {vehicle['class']}!",
                            "details": {
                                "person": person["class"],
                                "vehicle": vehicle["class"],
                                "distance_pixels": round(distance, 2),
                                "person_position": person["bbox"],
                                "vehicle_position": vehicle["bbox"]
                            }
                        })
        
        return alerts
    
    def _generate_ppe_alerts(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate PPE (Personal Protective Equipment) violation alerts
        
        Args:
            detections: List of detections
            
        Returns:
            List of PPE alerts
        """
        alerts = []
        
        # Check for objects that require PPE
        ppe_required_objects = [d for d in detections if d.get("requires_ppe", False)]
        
        if ppe_required_objects:
            # Count people detected
            people_count = len([d for d in detections if d["category"] == "people"])
            
            if people_count > 0:
                alerts.append({
                    "type": AlertType.PPE_VIOLATION.value,
                    "severity": AlertSeverity.HIGH.value,
                    "message": f"⚠️ PPE VERIFICATION REQUIRED: {people_count} worker(s) near hazardous area",
                    "details": {
                        "workers_detected": people_count,
                        "hazardous_objects": [obj["class"] for obj in ppe_required_objects],
                        "recommendation": "Verify workers are wearing appropriate PPE (hard hat, safety vest, gloves)"
                    }
                })
        
        return alerts
    
    def _generate_vehicle_alerts(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate vehicle-specific alerts
        
        Args:
            detections: List of detections
            
        Returns:
            List of vehicle alerts
        """
        alerts = []
        
        vehicles = [d for d in detections if d["category"] == "vehicle"]
        
        # Large vehicle warnings
        large_vehicles = [v for v in vehicles if v["class"] in ["truck", "bus"]]
        if large_vehicles:
            alerts.append({
                "type": AlertType.VEHICLE_WARNING.value,
                "severity": AlertSeverity.HIGH.value,
                "message": f"🚛 LARGE VEHICLE ALERT: {len(large_vehicles)} heavy vehicle(s) in area",
                "details": {
                    "vehicle_count": len(large_vehicles),
                    "vehicle_types": [v["class"] for v in large_vehicles],
                    "positions": [v["bbox"] for v in large_vehicles]
                }
            })
        
        # Multiple vehicles warning
        if len(vehicles) >= 3:
            alerts.append({
                "type": AlertType.VEHICLE_WARNING.value,
                "severity": AlertSeverity.MEDIUM.value,
                "message": f"🚗 TRAFFIC ALERT: {len(vehicles)} vehicles detected",
                "details": {
                    "vehicle_count": len(vehicles),
                    "vehicle_types": [v["class"] for v in vehicles]
                }
            })
        
        return alerts
    
    def _generate_multiple_hazard_alerts(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts for multiple hazards
        
        Args:
            detections: List of detections
            
        Returns:
            List of multiple hazard alerts
        """
        alerts = []
        
        # Count hazards
        critical_count = len([d for d in detections if d["danger_level"] == "critical"])
        high_count = len([d for d in detections if d["danger_level"] == "high"])
        
        total_hazards = critical_count + high_count
        
        if total_hazards >= 3:
            alerts.append({
                "type": AlertType.MULTIPLE_HAZARDS.value,
                "severity": AlertSeverity.CRITICAL.value,
                "message": f"🚨 MULTIPLE HAZARDS: {total_hazards} dangerous objects detected!",
                "details": {
                    "critical_hazards": critical_count,
                    "high_hazards": high_count,
                    "total_hazards": total_hazards,
                    "recommendation": "Exercise extreme caution - multiple danger zones present"
                }
            })
        
        return alerts
    
    def _generate_animal_alerts(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts for animal detection
        
        Args:
            detections: List of detections
            
        Returns:
            List of animal alerts
        """
        alerts = []
        
        animals = [d for d in detections if d["category"] == "animal"]
        
        if animals:
            for animal in animals:
                severity = AlertSeverity.MEDIUM if animal["danger_level"] in ["high", "medium"] else AlertSeverity.LOW
                
                alerts.append({
                    "type": AlertType.ANIMAL_DETECTED.value,
                    "severity": severity.value,
                    "message": f"🐾 ANIMAL DETECTED: {animal['class']} on construction site",
                    "details": {
                        "animal_type": animal["class"],
                        "confidence": animal["confidence"],
                        "position": animal["bbox"],
                        "recommendation": "Monitor animal movement and ensure worker safety"
                    }
                })
        
        return alerts
    
    def _generate_low_confidence_alerts(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts for low confidence critical detections
        
        Args:
            detections: List of detections
            
        Returns:
            List of low confidence alerts
        """
        alerts = []
        
        # Low confidence critical objects
        low_conf_critical = [
            d for d in detections
            if d["danger_level"] in ["critical", "high"]
            and d.get("low_confidence_warning", False)
        ]
        
        if low_conf_critical:
            for obj in low_conf_critical:
                alerts.append({
                    "type": AlertType.LOW_CONFIDENCE.value,
                    "severity": AlertSeverity.MEDIUM.value,
                    "message": f"⚠️ LOW CONFIDENCE: Possible {obj['class']} detected",
                    "details": {
                        "object_class": obj["class"],
                        "confidence": obj["confidence"],
                        "min_confidence": obj.get("min_confidence", 0.5),
                        "recommendation": "Visual verification recommended"
                    }
                })
        
        return alerts
    
    def _apply_alert_rules(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply custom alert rules from configuration
        
        Args:
            detections: List of detections
            
        Returns:
            List of rule-based alerts
        """
        alerts = []
        
        enabled_rules = self.config.get_enabled_alert_rules()
        
        for rule in enabled_rules:
            # Check rule conditions
            if self._check_rule_conditions(rule, detections):
                alerts.append({
                    "type": AlertType.PATTERN_ALERT.value,
                    "severity": rule.severity.value,
                    "message": rule.alert_message,
                    "details": {
                        "rule_name": rule.rule_name,
                        "conditions": rule.conditions
                    }
                })
        
        return alerts
    
    def _check_rule_conditions(
        self,
        rule: AlertRule,
        detections: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if alert rule conditions are met
        
        Args:
            rule: AlertRule to check
            detections: List of detections
            
        Returns:
            Boolean indicating if conditions are met
        """
        conditions = rule.conditions
        
        # Check min_persons condition
        if "min_persons" in conditions:
            person_count = len([d for d in detections if d["category"] == "people"])
            if person_count < conditions["min_persons"]:
                return False
        
        # Check vehicle_detected and person_detected
        if "vehicle_detected" in conditions and conditions["vehicle_detected"]:
            if not any(d["category"] == "vehicle" for d in detections):
                return False
        
        if "person_detected" in conditions and conditions["person_detected"]:
            if not any(d["category"] == "people" for d in detections):
                return False
        
        # Check sharp_object_detected
        if "sharp_object_detected" in conditions and conditions["sharp_object_detected"]:
            if not any(d["category"] == "sharp_object" for d in detections):
                return False
        
        # Check animal_detected
        if "animal_detected" in conditions and conditions["animal_detected"]:
            if not any(d["category"] == "animal" for d in detections):
                return False
        
        return True
    
    def _deduplicate_alerts(
        self,
        alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate alerts based on type and message
        
        Args:
            alerts: List of alerts
            
        Returns:
            Deduplicated list of alerts
        """
        seen = set()
        unique_alerts = []
        
        for alert in alerts:
            key = (alert["type"], alert["message"])
            if key not in seen:
                seen.add(key)
                unique_alerts.append(alert)
        
        return unique_alerts
    
    def _apply_cooldown(
        self,
        alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply cooldown period to prevent alert spam
        
        Args:
            alerts: List of alerts
            
        Returns:
            Filtered list of alerts
        """
        current_time = datetime.utcnow()
        filtered_alerts = []
        
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert['message']}"
            
            # Check if alert is in cooldown
            if alert_key in self.alert_cooldowns:
                last_alert_time = self.alert_cooldowns[alert_key]
                time_diff = (current_time - last_alert_time).total_seconds()
                
                if time_diff < self.cooldown_seconds:
                    continue  # Skip this alert
            
            # Add alert and update cooldown
            filtered_alerts.append(alert)
            self.alert_cooldowns[alert_key] = current_time
        
        # Clean up old cooldowns (older than 1 minute)
        expired_keys = [
            key for key, timestamp in self.alert_cooldowns.items()
            if (current_time - timestamp).total_seconds() > 60
        ]
        for key in expired_keys:
            del self.alert_cooldowns[key]
        
        return filtered_alerts
    
    def _sort_by_severity(
        self,
        alerts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort alerts by severity (critical first)
        
        Args:
            alerts: List of alerts
            
        Returns:
            Sorted list of alerts
        """
        severity_order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
            "info": 4
        }
        
        return sorted(alerts, key=lambda x: severity_order.get(x["severity"], 99))
    
    def _filter_by_severity(
        self,
        alerts: List[Dict[str, Any]],
        severity_filter: str
    ) -> List[Dict[str, Any]]:
        """
        Filter alerts by minimum severity
        
        Args:
            alerts: List of alerts
            severity_filter: Minimum severity level
            
        Returns:
            Filtered list of alerts
        """
        severity_order = ["critical", "high", "medium", "low", "info"]
        
        if severity_filter not in severity_order:
            return alerts
        
        min_index = severity_order.index(severity_filter)
        allowed_severities = severity_order[:min_index + 1]
        
        return [a for a in alerts if a["severity"] in allowed_severities]
    
    def generate_critical_alerts(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate only critical alerts (optimized for real-time streaming)
        
        Args:
            detections: List of detections
            
        Returns:
            List of critical alerts only
        """
        all_alerts = self.generate_alerts(detections)
        return [a for a in all_alerts if a["severity"] == "critical"]
    
    def get_total_alerts(self) -> int:
        """
        Get total number of alerts generated
        
        Returns:
            Integer count of total alerts
        """
        if self.enable_history and self.history:
            return self.history.alert_counts.get("total", 0)
        return 0
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics
        
        Returns:
            Dictionary with alert statistics
        """
        if self.enable_history and self.history:
            return self.history.get_statistics()
        return {"error": "Alert history not enabled"}
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            count: Number of recent alerts to return
            
        Returns:
            List of recent alert dictionaries
        """
        if self.enable_history and self.history:
            recent = self.history.get_recent(count)
            return [alert.to_dict() for alert in recent]
        return []
    
    def clear_cache(self):
        """Clear alert history and reset cooldowns"""
        if self.history:
            self.history.clear()
        self.alert_cooldowns.clear()
        logger.info("Alert service cache cleared")
    
    def __repr__(self) -> str:
        total = self.get_total_alerts()
        return f"AlertService(total_alerts={total}, cooldown={self.cooldown_seconds}s)"