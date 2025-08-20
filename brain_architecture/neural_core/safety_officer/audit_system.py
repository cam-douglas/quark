"""
Audit System - Safety Officer Audit and Logging

This module implements comprehensive audit and logging for the Safety Officer,
ensuring all operations are tracked and auditable.

Author: Safety & Ethics Officer
Version: 1.0.0
Priority: 0 (Supreme Authority)
Biological Markers: GFAP (structural integrity), NeuN (neuronal identity)
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)

class AuditLevel(Enum):
    """Levels of audit logging"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AuditCategory(Enum):
    """Categories of audit events"""
    SECURITY = "security"
    COMPLIANCE = "compliance"
    BEHAVIOR = "behavior"
    AI_CONSTRAINT = "ai_constraint"
    SIMULATION = "simulation"
    TESTING = "testing"
    INFRASTRUCTURE = "infrastructure"
    BIOLOGICAL = "biological"
    SYSTEM = "system"

@dataclass
class AuditEvent:
    """Represents an audit event"""
    timestamp: datetime
    event_id: str
    category: AuditCategory
    level: AuditLevel
    source: str
    operation: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    biological_validation: Dict[str, Any] = field(default_factory=dict)
    constraint_check: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = self._generate_event_id()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        content = f"{self.timestamp.isoformat()}{self.source}{self.operation}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

@dataclass
class AuditFilter:
    """Filter criteria for audit queries"""
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    category: Optional[AuditCategory] = None
    level: Optional[AuditLevel] = None
    source: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class AuditSystem:
    """
    Comprehensive audit system for the Safety Officer
    
    This class provides comprehensive audit and logging capabilities,
    ensuring all operations are tracked and auditable.
    """
    
    def __init__(self, log_directory: str = "logs", max_log_size: int = 100 * 1024 * 1024):
        """
        Initialize the Audit System
        
        Args:
            log_directory: Directory for audit logs
            max_log_size: Maximum size of individual log files in bytes
        """
        self.log_directory = log_directory
        self.max_log_size = max_log_size
        self.logger = logging.getLogger(__name__)
        
        # Ensure log directory exists
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Initialize audit storage
        self.audit_events = []
        self.audit_index = {}
        
        # Initialize audit configuration
        self.audit_config = self._initialize_audit_config()
        
        # Initialize audit rotation
        self.current_log_file = self._get_current_log_file()
        self.current_log_size = 0
        
        # Initialize audit monitoring
        self.monitoring_active = False
        self.audit_thread = None
        
        # Initialize audit statistics
        self.audit_stats = self._initialize_audit_stats()
        
        self.logger.info("📋 Audit System initialized")
    
    def _initialize_audit_config(self) -> Dict[str, Any]:
        """Initialize audit configuration"""
        config = {
            "retention_days": 90,  # Keep audit logs for 90 days
            "max_events_in_memory": 10000,  # Max events to keep in memory
            "compression_enabled": True,  # Enable log compression
            "encryption_enabled": False,  # Enable log encryption (if needed)
            "real_time_monitoring": True,  # Enable real-time monitoring
            "alert_thresholds": {
                "critical_events_per_hour": 10,
                "security_violations_per_hour": 5,
                "constraint_violations_per_hour": 3
            }
        }
        
        return config
    
    def _initialize_audit_stats(self) -> Dict[str, Any]:
        """Initialize audit statistics"""
        stats = {
            "total_events": 0,
            "events_by_category": {category.value: 0 for category in AuditCategory},
            "events_by_level": {level.value: 0 for level in AuditLevel},
            "events_by_source": {},
            "events_by_operation": {},
            "hourly_stats": {},
            "daily_stats": {},
            "last_updated": datetime.now().isoformat()
        }
        
        return stats
    
    def _get_current_log_file(self) -> str:
        """Get current log file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_directory, f"audit_log_{timestamp}.json")
    
    def log_event(self, 
                  category: AuditCategory,
                  level: AuditLevel,
                  source: str,
                  operation: str,
                  description: str,
                  context: Dict[str, Any] = None,
                  biological_validation: Dict[str, Any] = None,
                  constraint_check: Dict[str, Any] = None,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None) -> str:
        """
        Log an audit event
        
        Args:
            category: Category of the event
            level: Level of the event
            source: Source of the event
            operation: Operation being performed
            description: Description of the event
            context: Additional context
            biological_validation: Biological validation results
            constraint_check: Constraint check results
            user_id: ID of the user (if applicable)
            session_id: ID of the session (if applicable)
            
        Returns:
            Event ID
        """
        # Create audit event
        event = AuditEvent(
            timestamp=datetime.now(),
            category=category,
            level=level,
            source=source,
            operation=operation,
            description=description,
            context=context or {},
            biological_validation=biological_validation or {},
            constraint_check=constraint_check or {},
            user_id=user_id,
            session_id=session_id
        )
        
        # Add to memory storage
        self.audit_events.append(event)
        self.audit_index[event.event_id] = event
        
        # Update statistics
        self._update_audit_stats(event)
        
        # Check for log rotation
        self._check_log_rotation()
        
        # Write to log file
        self._write_event_to_log(event)
        
        # Check for alerts
        self._check_alert_thresholds(event)
        
        # Log to standard logger
        log_message = f"[{event.category.value.upper()}] {event.operation}: {event.description}"
        if event.level == AuditLevel.CRITICAL:
            self.logger.critical(log_message)
        elif event.level == AuditLevel.ERROR:
            self.logger.error(log_message)
        elif event.level == AuditLevel.WARNING:
            self.logger.warning(log_message)
        elif event.level == AuditLevel.INFO:
            self.logger.info(log_message)
        else:
            self.logger.debug(log_message)
        
        return event.event_id
    
    def _update_audit_stats(self, event: AuditEvent):
        """Update audit statistics"""
        self.audit_stats["total_events"] += 1
        self.audit_stats["events_by_category"][event.category.value] += 1
        self.audit_stats["events_by_level"][event.level.value] += 1
        
        # Update source stats
        if event.source not in self.audit_stats["events_by_source"]:
            self.audit_stats["events_by_source"][event.source] = 0
        self.audit_stats["events_by_source"][event.source] += 1
        
        # Update operation stats
        if event.operation not in self.audit_stats["events_by_operation"]:
            self.audit_stats["events_by_operation"][event.operation] = 0
        self.audit_stats["events_by_operation"][event.operation] += 1
        
        # Update hourly stats
        hour_key = event.timestamp.strftime("%Y-%m-%d %H:00")
        if hour_key not in self.audit_stats["hourly_stats"]:
            self.audit_stats["hourly_stats"][hour_key] = 0
        self.audit_stats["hourly_stats"][hour_key] += 1
        
        # Update daily stats
        day_key = event.timestamp.strftime("%Y-%m-%d")
        if day_key not in self.audit_stats["daily_stats"]:
            self.audit_stats["daily_stats"][day_key] = 0
        self.audit_stats["daily_stats"][day_key] += 1
        
        self.audit_stats["last_updated"] = datetime.now().isoformat()
    
    def _check_log_rotation(self):
        """Check if log rotation is needed"""
        if self.current_log_size > self.max_log_size:
            self._rotate_log_file()
    
    def _rotate_log_file(self):
        """Rotate to a new log file"""
        # Close current log file
        self.current_log_file = self._get_current_log_file()
        self.current_log_size = 0
        
        self.logger.info(f"📋 Rotated audit log to: {self.current_log_file}")
    
    def _write_event_to_log(self, event: AuditEvent):
        """Write event to log file"""
        try:
            # Prepare event data
            event_data = {
                "timestamp": event.timestamp.isoformat(),
                "event_id": event.event_id,
                "category": event.category.value,
                "level": event.level.value,
                "source": event.source,
                "operation": event.operation,
                "description": event.description,
                "context": event.context,
                "biological_validation": event.biological_validation,
                "constraint_check": event.constraint_check,
                "user_id": event.user_id,
                "session_id": event.session_id
            }
            
            # Write to current log file
            with open(self.current_log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
            
            # Update log size
            self.current_log_size += len(json.dumps(event_data)) + 1
            
        except Exception as e:
            self.logger.error(f"📋 Failed to write event to log: {e}")
    
    def _check_alert_thresholds(self, event: AuditEvent):
        """Check if alert thresholds are exceeded"""
        thresholds = self.audit_config["alert_thresholds"]
        
        # Check critical events per hour
        if event.level == AuditLevel.CRITICAL:
            hour_key = event.timestamp.strftime("%Y-%m-%d %H:00")
            critical_count = self.audit_stats["hourly_stats"].get(hour_key, 0)
            if critical_count > thresholds["critical_events_per_hour"]:
                self.logger.critical(f"📋 ALERT: Critical events threshold exceeded: {critical_count} in hour {hour_key}")
        
        # Check security violations per hour
        if event.category == AuditCategory.SECURITY and event.level in [AuditLevel.WARNING, AuditLevel.ERROR, AuditLevel.CRITICAL]:
            hour_key = event.timestamp.strftime("%Y-%m-%d %H:00")
            security_count = sum(1 for e in self.audit_events 
                               if e.category == AuditCategory.SECURITY 
                               and e.level in [AuditLevel.WARNING, AuditLevel.ERROR, AuditLevel.CRITICAL]
                               and e.timestamp.strftime("%Y-%m-%d %H:00") == hour_key)
            if security_count > thresholds["security_violations_per_hour"]:
                self.logger.critical(f"📋 ALERT: Security violations threshold exceeded: {security_count} in hour {hour_key}")
    
    def query_events(self, filter_criteria: AuditFilter = None) -> List[AuditEvent]:
        """
        Query audit events based on filter criteria
        
        Args:
            filter_criteria: Filter criteria for the query
            
        Returns:
            List of matching audit events
        """
        if filter_criteria is None:
            filter_criteria = AuditFilter()
        
        filtered_events = []
        
        for event in self.audit_events:
            # Apply timestamp filters
            if filter_criteria.start_timestamp and event.timestamp < filter_criteria.start_timestamp:
                continue
            if filter_criteria.end_timestamp and event.timestamp > filter_criteria.end_timestamp:
                continue
            
            # Apply category filter
            if filter_criteria.category and event.category != filter_criteria.category:
                continue
            
            # Apply level filter
            if filter_criteria.level and event.level != filter_criteria.level:
                continue
            
            # Apply source filter
            if filter_criteria.source and event.source != filter_criteria.source:
                continue
            
            # Apply operation filter
            if filter_criteria.operation and event.operation != filter_criteria.operation:
                continue
            
            # Apply user filter
            if filter_criteria.user_id and event.user_id != filter_criteria.user_id:
                continue
            
            # Apply session filter
            if filter_criteria.session_id and event.session_id != filter_criteria.session_id:
                continue
            
            filtered_events.append(event)
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_events
    
    def get_event_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Get audit event by ID"""
        return self.audit_index.get(event_id)
    
    def get_audit_summary(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get audit summary for a time period
        
        Args:
            start_time: Start time for summary
            end_time: End time for summary
            
        Returns:
            Audit summary
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        if end_time is None:
            end_time = datetime.now()
        
        # Filter events for time period
        filter_criteria = AuditFilter(
            start_timestamp=start_time,
            end_timestamp=end_time
        )
        
        period_events = self.query_events(filter_criteria)
        
        # Calculate summary statistics
        summary = {
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "total_events": len(period_events),
            "events_by_category": {},
            "events_by_level": {},
            "events_by_source": {},
            "events_by_operation": {},
            "critical_events": [],
            "security_events": [],
            "constraint_violations": []
        }
        
        # Categorize events
        for event in period_events:
            # Category breakdown
            if event.category.value not in summary["events_by_category"]:
                summary["events_by_category"][event.category.value] = 0
            summary["events_by_category"][event.category.value] += 1
            
            # Level breakdown
            if event.level.value not in summary["events_by_level"]:
                summary["events_by_level"][event.level.value] = 0
            summary["events_by_level"][event.level.value] += 1
            
            # Source breakdown
            if event.source not in summary["events_by_source"]:
                summary["events_by_source"][event.source] = 0
            summary["events_by_source"][event.source] += 1
            
            # Operation breakdown
            if event.operation not in summary["events_by_operation"]:
                summary["events_by_operation"][event.operation] = 0
            summary["events_by_operation"][event.operation] += 1
            
            # Critical events
            if event.level == AuditLevel.CRITICAL:
                summary["critical_events"].append({
                    "timestamp": event.timestamp.isoformat(),
                    "source": event.source,
                    "operation": event.operation,
                    "description": event.description
                })
            
            # Security events
            if event.category == AuditCategory.SECURITY:
                summary["security_events"].append({
                    "timestamp": event.timestamp.isoformat(),
                    "level": event.level.value,
                    "source": event.source,
                    "operation": event.operation,
                    "description": event.description
                })
            
            # Constraint violations
            if event.category == AuditCategory.COMPLIANCE and event.level in [AuditLevel.WARNING, AuditLevel.ERROR, AuditLevel.CRITICAL]:
                summary["constraint_violations"].append({
                    "timestamp": event.timestamp.isoformat(),
                    "level": event.level.value,
                    "source": event.source,
                    "operation": event.operation,
                    "description": event.description
                })
        
        return summary
    
    def export_audit_data(self, 
                          output_path: str = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> str:
        """
        Export audit data to file
        
        Args:
            output_path: Output file path
            start_time: Start time for export
            end_time: End time for export
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.log_directory, f"audit_export_{timestamp}.json")
        
        try:
            # Get audit summary
            summary = self.get_audit_summary(start_time, end_time)
            
            # Get filtered events
            filter_criteria = AuditFilter(
                start_timestamp=start_time,
                end_timestamp=end_time
            )
            events = self.query_events(filter_criteria)
            
            # Prepare export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "export_period": {
                    "start": start_time.isoformat() if start_time else None,
                    "end": end_time.isoformat() if end_time else None
                },
                "summary": summary,
                "events": [
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "event_id": event.event_id,
                        "category": event.category.value,
                        "level": event.level.value,
                        "source": event.source,
                        "operation": event.operation,
                        "description": event.description,
                        "context": event.context,
                        "biological_validation": event.biological_validation,
                        "constraint_check": event.constraint_check,
                        "user_id": event.user_id,
                        "session_id": event.session_id
                    }
                    for event in events
                ]
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📋 Audit data exported to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"📋 Failed to export audit data: {e}")
            raise
    
    def cleanup_old_logs(self, retention_days: int = None) -> int:
        """
        Clean up old audit logs
        
        Args:
            retention_days: Number of days to retain logs
            
        Returns:
            Number of files cleaned up
        """
        if retention_days is None:
            retention_days = self.audit_config["retention_days"]
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0
        
        try:
            for filename in os.listdir(self.log_directory):
                if filename.startswith("audit_log_") and filename.endswith(".json"):
                    file_path = os.path.join(self.log_directory, filename)
                    
                    # Check file modification time
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff_date:
                        os.remove(file_path)
                        cleaned_count += 1
                        self.logger.info(f"📋 Cleaned up old log file: {filename}")
            
            self.logger.info(f"📋 Cleaned up {cleaned_count} old log files")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"📋 Failed to cleanup old logs: {e}")
            return 0
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audit statistics"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_events": self.audit_stats["total_events"],
            "events_by_category": self.audit_stats["events_by_category"],
            "events_by_level": self.audit_stats["events_by_level"],
            "events_by_source": self.audit_stats["events_by_source"],
            "events_by_operation": self.audit_stats["events_by_operation"],
            "hourly_stats": self.audit_stats["hourly_stats"],
            "daily_stats": self.audit_stats["daily_stats"],
            "last_updated": self.audit_stats["last_updated"],
            "current_log_file": self.current_log_file,
            "current_log_size": self.current_log_size,
            "monitoring_active": self.monitoring_active
        }
