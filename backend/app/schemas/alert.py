"""
Alert Schemas Module
Pydantic models for alert-related requests, responses, and data structures
Handles alert management, history, statistics, and notification schemas
"""

from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

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
    ZONE_BREACH = "zone_breach"
    EQUIPMENT_MALFUNCTION = "equipment_malfunction"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


class AlertPriority(str, Enum):
    """Alert priority for handling"""
    IMMEDIATE = "immediate"
    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"


class NotificationChannel(str, Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


# ============================================================================
# Base Alert Schemas
# ============================================================================

class AlertBase(BaseModel):
    """Base alert information"""
    alert_type: AlertType = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    message: str = Field(..., min_length=1, max_length=500, description="Alert message")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional alert details")
    
    class Config:
        use_enum_values = True


class AlertCreate(AlertBase):
    """Schema for creating a new alert"""
    priority: AlertPriority = Field(default=AlertPriority.NORMAL, description="Alert priority")
    tags: Optional[List[str]] = Field(default_factory=list, description="Alert tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags list"""
        if v and len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "alert_type": "proximity_warning",
                "severity": "critical",
                "message": "Worker detected too close to heavy machinery",
                "details": {
                    "worker_id": "W-001",
                    "distance": 45.3,
                    "equipment": "excavator"
                },
                "priority": "immediate",
                "tags": ["safety", "proximity", "zone-a"],
                "metadata": {
                    "camera_id": "cam_03",
                    "location": "Zone A - North"
                }
            }
        }


class Alert(AlertBase):
    """Complete alert with system fields"""
    alert_id: str = Field(..., description="Unique alert identifier")
    status: AlertStatus = Field(default=AlertStatus.ACTIVE, description="Alert status")
    priority: AlertPriority = Field(default=AlertPriority.NORMAL, description="Alert priority")
    timestamp: str = Field(..., description="Alert creation timestamp (ISO format)")
    acknowledged: bool = Field(default=False, description="Whether alert is acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    acknowledged_at: Optional[str] = Field(None, description="Acknowledgement timestamp")
    resolved: bool = Field(default=False, description="Whether alert is resolved")
    resolved_by: Optional[str] = Field(None, description="User who resolved")
    resolved_at: Optional[str] = Field(None, description="Resolution timestamp")
    tags: List[str] = Field(default_factory=list, description="Alert tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Detection related fields
    frame_number: Optional[int] = Field(None, description="Frame number where alert was generated")
    detection_ids: Optional[List[str]] = Field(default_factory=list, description="Related detection IDs")
    
    # Action tracking
    actions_taken: Optional[List[str]] = Field(default_factory=list, description="Actions taken")
    notes: Optional[str] = Field(None, max_length=1000, description="Additional notes")
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4e5f6",
                "alert_type": "proximity_warning",
                "severity": "critical",
                "message": "🚨 DANGER: Worker too close to truck!",
                "details": {
                    "person": "person",
                    "vehicle": "truck",
                    "distance_pixels": 85.3
                },
                "status": "active",
                "priority": "immediate",
                "timestamp": "2025-11-07T12:54:57Z",
                "acknowledged": False,
                "acknowledged_by": None,
                "acknowledged_at": None,
                "resolved": False,
                "resolved_by": None,
                "resolved_at": None,
                "tags": ["proximity", "critical", "zone-a"],
                "metadata": {
                    "camera_id": "cam_01",
                    "location": "Main Construction Area"
                },
                "frame_number": 1523,
                "detection_ids": ["det_001", "det_002"],
                "actions_taken": [],
                "notes": None
            }
        }


# ============================================================================
# Alert Action Schemas
# ============================================================================

class AlertAcknowledge(BaseModel):
    """Schema for acknowledging an alert"""
    alert_id: str = Field(..., description="Alert ID to acknowledge")
    acknowledged_by: str = Field(default="A-P-U-R-B-O", description="User acknowledging the alert")
    notes: Optional[str] = Field(None, max_length=500, description="Acknowledgement notes")
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4e5f6",
                "acknowledged_by": "A-P-U-R-B-O",
                "notes": "Safety supervisor notified, area being cleared"
            }
        }


class AlertResolve(BaseModel):
    """Schema for resolving an alert"""
    alert_id: str = Field(..., description="Alert ID to resolve")
    resolved_by: str = Field(default="A-P-U-R-B-O", description="User resolving the alert")
    resolution_notes: str = Field(..., min_length=1, max_length=1000, description="Resolution details")
    actions_taken: List[str] = Field(..., min_items=1, description="Actions taken to resolve")
    
    @validator('actions_taken')
    def validate_actions(cls, v):
        """Validate actions list"""
        if not v:
            raise ValueError("At least one action must be specified")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4e5f6",
                "resolved_by": "A-P-U-R-B-O",
                "resolution_notes": "Worker moved to safe distance. Spotter assigned to vehicle operations.",
                "actions_taken": [
                    "Relocated worker to safe zone",
                    "Assigned dedicated spotter",
                    "Conducted safety briefing"
                ]
            }
        }


class AlertDismiss(BaseModel):
    """Schema for dismissing a false positive alert"""
    alert_id: str = Field(..., description="Alert ID to dismiss")
    dismissed_by: str = Field(default="A-P-U-R-B-O", description="User dismissing the alert")
    reason: str = Field(..., min_length=1, max_length=500, description="Reason for dismissal")
    false_positive: bool = Field(default=True, description="Whether this is a false positive")
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4e5f6",
                "dismissed_by": "A-P-U-R-B-O",
                "reason": "False positive - detected object was safety mannequin during training",
                "false_positive": True
            }
        }


class AlertEscalate(BaseModel):
    """Schema for escalating an alert"""
    alert_id: str = Field(..., description="Alert ID to escalate")
    escalated_by: str = Field(default="A-P-U-R-B-O", description="User escalating")
    escalate_to: str = Field(..., description="Person/team to escalate to")
    escalation_reason: str = Field(..., min_length=1, max_length=500, description="Escalation reason")
    new_priority: AlertPriority = Field(..., description="New priority level")
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4e5f6",
                "escalated_by": "A-P-U-R-B-O",
                "escalate_to": "site_manager",
                "escalation_reason": "Critical safety violation requiring immediate management intervention",
                "new_priority": "immediate"
            }
        }


class AlertUpdate(BaseModel):
    """Schema for updating alert details"""
    alert_id: str = Field(..., description="Alert ID to update")
    status: Optional[AlertStatus] = Field(None, description="New status")
    priority: Optional[AlertPriority] = Field(None, description="New priority")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    notes: Optional[str] = Field(None, max_length=1000, description="Additional notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4e5f6",
                "status": "acknowledged",
                "priority": "urgent",
                "tags": ["proximity", "critical", "zone-a", "in-progress"],
                "notes": "Safety team en route to location",
                "metadata": {
                    "response_team": "Team Alpha",
                    "eta_minutes": 2
                }
            }
        }


# ============================================================================
# Alert Query Schemas
# ============================================================================

class AlertFilter(BaseModel):
    """Schema for filtering alerts"""
    severity: Optional[List[AlertSeverity]] = Field(None, description="Filter by severity levels")
    alert_type: Optional[List[AlertType]] = Field(None, description="Filter by alert types")
    status: Optional[List[AlertStatus]] = Field(None, description="Filter by status")
    priority: Optional[List[AlertPriority]] = Field(None, description="Filter by priority")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (any match)")
    from_timestamp: Optional[str] = Field(None, description="Start timestamp (ISO format)")
    to_timestamp: Optional[str] = Field(None, description="End timestamp (ISO format)")
    acknowledged: Optional[bool] = Field(None, description="Filter by acknowledgement status")
    resolved: Optional[bool] = Field(None, description="Filter by resolution status")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    sort_by: str = Field(default="timestamp", description="Sort field")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")
    
    class Config:
        schema_extra = {
            "example": {
                "severity": ["critical", "high"],
                "alert_type": ["proximity_warning", "ppe_violation"],
                "status": ["active"],
                "priority": ["immediate", "urgent"],
                "tags": ["zone-a"],
                "from_timestamp": "2025-11-07T00:00:00Z",
                "to_timestamp": "2025-11-07T23:59:59Z",
                "acknowledged": False,
                "resolved": False,
                "limit": 50,
                "offset": 0,
                "sort_by": "timestamp",
                "sort_order": "desc"
            }
        }


class AlertSearchQuery(BaseModel):
    """Schema for searching alerts"""
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    search_fields: List[str] = Field(
        default=["message", "details", "notes"],
        description="Fields to search in"
    )
    filters: Optional[AlertFilter] = Field(None, description="Additional filters")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "truck proximity",
                "search_fields": ["message", "details"],
                "filters": {
                    "severity": ["critical"],
                    "status": ["active"]
                }
            }
        }


# ============================================================================
# Alert Response Schemas
# ============================================================================

class AlertResponse(BaseModel):
    """Single alert response"""
    success: bool = Field(..., description="Operation success")
    alert: Alert = Field(..., description="Alert data")
    message: Optional[str] = Field(None, description="Response message")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "alert": {},
                "message": "Alert acknowledged successfully"
            }
        }


class AlertListResponse(BaseModel):
    """List of alerts response"""
    success: bool = Field(..., description="Operation success")
    total_count: int = Field(..., description="Total alerts matching criteria")
    returned_count: int = Field(..., description="Number of alerts returned")
    alerts: List[Alert] = Field(..., description="List of alerts")
    has_more: bool = Field(..., description="Whether more results available")
    next_offset: Optional[int] = Field(None, description="Next pagination offset")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_count": 145,
                "returned_count": 50,
                "alerts": [],
                "has_more": True,
                "next_offset": 50
            }
        }


class AlertStatistics(BaseModel):
    """Alert statistics"""
    total_alerts: int = Field(..., description="Total alerts generated")
    active_alerts: int = Field(..., description="Currently active alerts")
    acknowledged_alerts: int = Field(..., description="Acknowledged alerts")
    resolved_alerts: int = Field(..., description="Resolved alerts")
    dismissed_alerts: int = Field(..., description="Dismissed alerts")
    
    # Severity breakdown
    severity_distribution: Dict[str, int] = Field(..., description="Alerts by severity")
    
    # Type breakdown
    type_distribution: Dict[str, int] = Field(..., description="Alerts by type")
    
    # Priority breakdown
    priority_distribution: Dict[str, int] = Field(..., description="Alerts by priority")
    
    # Time-based stats
    alerts_last_hour: int = Field(..., description="Alerts in last hour")
    alerts_last_24h: int = Field(..., description="Alerts in last 24 hours")
    alerts_last_7d: int = Field(..., description="Alerts in last 7 days")
    
    # Response metrics
    average_acknowledgement_time_seconds: Optional[float] = Field(
        None, description="Average time to acknowledge"
    )
    average_resolution_time_seconds: Optional[float] = Field(
        None, description="Average time to resolve"
    )
    
    # Top categories
    top_alert_types: List[Dict[str, Any]] = Field(..., description="Most common alert types")
    top_tags: List[Dict[str, Any]] = Field(..., description="Most common tags")
    
    class Config:
        schema_extra = {
            "example": {
                "total_alerts": 1523,
                "active_alerts": 23,
                "acknowledged_alerts": 1245,
                "resolved_alerts": 1200,
                "dismissed_alerts": 55,
                "severity_distribution": {
                    "critical": 145,
                    "high": 423,
                    "medium": 678,
                    "low": 277
                },
                "type_distribution": {
                    "proximity_warning": 456,
                    "ppe_violation": 334,
                    "vehicle_warning": 289
                },
                "priority_distribution": {
                    "immediate": 67,
                    "urgent": 234,
                    "normal": 890,
                    "low": 332
                },
                "alerts_last_hour": 12,
                "alerts_last_24h": 156,
                "alerts_last_7d": 892,
                "average_acknowledgement_time_seconds": 45.3,
                "average_resolution_time_seconds": 320.7,
                "top_alert_types": [
                    {"type": "proximity_warning", "count": 456},
                    {"type": "ppe_violation", "count": 334}
                ],
                "top_tags": [
                    {"tag": "zone-a", "count": 567},
                    {"tag": "critical", "count": 434}
                ]
            }
        }


class AlertStatisticsResponse(BaseModel):
    """Alert statistics response"""
    success: bool = Field(..., description="Operation success")
    statistics: AlertStatistics = Field(..., description="Alert statistics")
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Statistics generation timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "statistics": {},
                "generated_at": "2025-11-07T12:54:57Z"
            }
        }


# ============================================================================
# Alert Notification Schemas
# ============================================================================

class NotificationPreference(BaseModel):
    """User notification preferences"""
    user_id: str = Field(..., description="User identifier")
    channels: List[NotificationChannel] = Field(..., description="Enabled notification channels")
    severity_threshold: AlertSeverity = Field(
        default=AlertSeverity.MEDIUM,
        description="Minimum severity to notify"
    )
    alert_types: Optional[List[AlertType]] = Field(
        None, description="Alert types to notify (None = all)"
    )
    quiet_hours: Optional[Dict[str, str]] = Field(
        None, description="Quiet hours (start/end in HH:MM format)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "A-P-U-R-B-O",
                "channels": ["email", "sms", "push"],
                "severity_threshold": "high",
                "alert_types": ["proximity_warning", "ppe_violation"],
                "quiet_hours": {
                    "start": "22:00",
                    "end": "06:00"
                }
            }
        }


class NotificationPayload(BaseModel):
    """Notification delivery payload"""
    alert_id: str = Field(..., description="Alert identifier")
    recipient: str = Field(..., description="Recipient identifier")
    channel: NotificationChannel = Field(..., description="Delivery channel")
    subject: str = Field(..., description="Notification subject")
    body: str = Field(..., description="Notification body")
    priority: AlertPriority = Field(..., description="Notification priority")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional data")
    
    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4e5f6",
                "recipient": "A-P-U-R-B-O",
                "channel": "email",
                "subject": "🚨 CRITICAL ALERT: Proximity Warning",
                "body": "Worker detected too close to heavy machinery in Zone A",
                "priority": "immediate",
                "metadata": {
                    "location": "Zone A",
                    "camera_id": "cam_03"
                }
            }
        }


class NotificationStatus(BaseModel):
    """Notification delivery status"""
    notification_id: str = Field(..., description="Notification identifier")
    alert_id: str = Field(..., description="Related alert ID")
    recipient: str = Field(..., description="Recipient")
    channel: NotificationChannel = Field(..., description="Delivery channel")
    status: str = Field(..., description="Delivery status")
    sent_at: str = Field(..., description="Send timestamp")
    delivered_at: Optional[str] = Field(None, description="Delivery timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "notification_id": "notif_xyz123",
                "alert_id": "alert_a1b2c3d4e5f6",
                "recipient": "A-P-U-R-B-O",
                "channel": "email",
                "status": "delivered",
                "sent_at": "2025-11-07T12:54:57Z",
                "delivered_at": "2025-11-07T12:55:02Z",
                "error": None
            }
        }


# ============================================================================
# Alert Rule Schemas
# ============================================================================

class AlertRuleCondition(BaseModel):
    """Condition for alert rule"""
    field: str = Field(..., description="Field to check")
    operator: str = Field(..., pattern="^(eq|ne|gt|lt|gte|lte|in|contains)$", description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    
    class Config:
        schema_extra = {
            "example": {
                "field": "danger_level",
                "operator": "in",
                "value": ["critical", "high"]
            }
        }


class AlertRule(BaseModel):
    """Alert generation rule"""
    rule_id: str = Field(..., description="Rule identifier")
    rule_name: str = Field(..., min_length=1, max_length=100, description="Rule name")
    description: Optional[str] = Field(None, max_length=500, description="Rule description")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    conditions: List[AlertRuleCondition] = Field(..., min_items=1, description="Rule conditions")
    alert_template: AlertCreate = Field(..., description="Alert template to generate")
    cooldown_seconds: int = Field(default=60, ge=0, description="Cooldown between alerts")
    
    class Config:
        schema_extra = {
            "example": {
                "rule_id": "rule_001",
                "rule_name": "Critical Object Detection",
                "description": "Generate alert when critical danger objects detected",
                "enabled": True,
                "conditions": [
                    {
                        "field": "danger_level",
                        "operator": "eq",
                        "value": "critical"
                    }
                ],
                "alert_template": {
                    "alert_type": "danger_detection",
                    "severity": "critical",
                    "message": "Critical danger object detected",
                    "priority": "immediate"
                },
                "cooldown_seconds": 30
            }
        }


class AlertRuleResponse(BaseModel):
    """Alert rule response"""
    success: bool = Field(..., description="Operation success")
    rule: AlertRule = Field(..., description="Alert rule")
    message: Optional[str] = Field(None, description="Response message")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "rule": {},
                "message": "Alert rule created successfully"
            }
        }


class AlertRuleListResponse(BaseModel):
    """List of alert rules"""
    success: bool = Field(..., description="Operation success")
    total_rules: int = Field(..., description="Total number of rules")
    enabled_rules: int = Field(..., description="Number of enabled rules")
    rules: List[AlertRule] = Field(..., description="List of rules")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_rules": 15,
                "enabled_rules": 12,
                "rules": []
            }
        }


# ============================================================================
# Bulk Operations Schemas
# ============================================================================

class BulkAlertAction(BaseModel):
    """Bulk action on multiple alerts"""
    alert_ids: List[str] = Field(..., min_items=1, max_items=100, description="Alert IDs")
    action: str = Field(..., pattern="^(acknowledge|resolve|dismiss|delete)$", description="Action to perform")
    performed_by: str = Field(default="A-P-U-R-B-O", description="User performing action")
    notes: Optional[str] = Field(None, max_length=500, description="Action notes")
    
    @validator('alert_ids')
    def validate_alert_ids(cls, v):
        """Validate alert IDs list"""
        if len(v) > 100:
            raise ValueError("Maximum 100 alerts per bulk operation")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "alert_ids": ["alert_001", "alert_002", "alert_003"],
                "action": "acknowledge",
                "performed_by": "A-P-U-R-B-O",
                "notes": "Bulk acknowledged after safety drill completion"
            }
        }


class BulkAlertActionResponse(BaseModel):
    """Bulk action response"""
    success: bool = Field(..., description="Overall operation success")
    total_requested: int = Field(..., description="Total alerts in request")
    successful: int = Field(..., description="Successfully processed")
    failed: int = Field(..., description="Failed to process")
    results: List[Dict[str, Any]] = Field(..., description="Individual results")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_requested": 3,
                "successful": 3,
                "failed": 0,
                "results": [
                    {"alert_id": "alert_001", "success": True},
                    {"alert_id": "alert_002", "success": True},
                    {"alert_id": "alert_003", "success": True}
                ]
            }
        }


# ============================================================================
# Error Schemas
# ============================================================================

class AlertErrorResponse(BaseModel):
    """Alert-specific error response"""
    success: bool = Field(default=False, description="Always False for errors")
    error_code: str = Field(..., description="Error code")
    error: str = Field(..., description="Error message")
    alert_id: Optional[str] = Field(None, description="Related alert ID if applicable")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Error timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error_code": "ALERT_NOT_FOUND",
                "error": "Alert with ID 'alert_xyz' not found",
                "alert_id": "alert_xyz",
                "timestamp": "2025-11-07T12:54:57Z"
            }
        }
