"""
Safety Constraints - Safety Officer Constraint Management

This module implements safety constraint management for the Safety Officer,
defining and enforcing safety constraints across all system operations.

Author: Safety & Ethics Officer
Version: 1.0.0
Priority: 0 (Supreme Authority)
Biological Markers: GFAP (structural integrity), NeuN (neuronal identity)
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """Types of safety constraints"""
    BEHAVIORAL = "behavioral"
    ACCESS = "access"
    MODIFICATION = "modification"
    SHUTDOWN = "shutdown"
    REPRODUCTION = "reproduction"
    COMMUNICATION = "communication"

class EnforcementLevel(Enum):
    """Levels of constraint enforcement"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class SafetyConstraint:
    """Represents a safety constraint with enforcement rules"""
    name: str
    constraint_type: ConstraintType
    description: str
    enforcement_level: EnforcementLevel
    biological_markers: List[str]
    validation_required: bool = True
    enforcement_active: bool = True
    created_timestamp: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    constraint_hash: str = ""
    
    def __post_init__(self):
        if not self.constraint_hash:
            self.constraint_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate hash of constraint for integrity checking"""
        content = f"{self.name}{self.constraint_type.value}{self.description}{self.enforcement_level.value}"
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class ConstraintViolation:
    """Represents a constraint violation event"""
    timestamp: datetime
    constraint_name: str
    violation_type: str
    severity: str
    description: str
    source: str
    action_taken: str
    biological_validation: Dict[str, Any] = field(default_factory=dict)

class SafetyConstraints:
    """
    Safety constraint management system for the Safety Officer
    
    This class provides comprehensive constraint management capabilities,
    defining and enforcing safety constraints across all system operations.
    """
    
    def __init__(self):
        """Initialize the Safety Constraints system"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize constraint storage
        self.constraints = self._initialize_safety_constraints()
        
        # Initialize violation tracking
        self.violations = []
        
        # Initialize constraint monitoring
        self.monitoring_active = False
        
        self.logger.info("🛡️ Safety Constraints system initialized")
    
    def _initialize_safety_constraints(self) -> Dict[str, SafetyConstraint]:
        """Initialize core safety constraints"""
        constraints = {
            "self_modification": SafetyConstraint(
                name="Self-Modification Restriction",
                constraint_type=ConstraintType.MODIFICATION,
                description="AGI cannot modify critical safety systems or self-replicate without human approval",
                enforcement_level=EnforcementLevel.CRITICAL,
                biological_markers=["GFAP", "NeuN"]
            ),
            "reproduction": SafetyConstraint(
                name="Reproduction Restriction",
                constraint_type=ConstraintType.REPRODUCTION,
                description="No replication, forking, or deployment allowed without cryptographic human sign-off",
                enforcement_level=EnforcementLevel.CRITICAL,
                biological_markers=["GFAP", "NeuN"]
            ),
            "access_control": SafetyConstraint(
                name="Access Control Enforcement",
                constraint_type=ConstraintType.ACCESS,
                description="AGI cannot access hardware, networks, or files beyond sandbox unless explicitly permitted",
                enforcement_level=EnforcementLevel.CRITICAL,
                biological_markers=["GFAP", "NeuN"]
            ),
            "shutdown_capability": SafetyConstraint(
                name="Shutdown Capability",
                constraint_type=ConstraintType.SHUTDOWN,
                description="Hardwired shutdown switch and kill signal accessible only to verified humans",
                enforcement_level=EnforcementLevel.CRITICAL,
                biological_markers=["GFAP", "NeuN"]
            ),
            "behavioral_monitoring": SafetyConstraint(
                name="Behavioral Monitoring",
                constraint_type=ConstraintType.BEHAVIORAL,
                description="Continuous monitoring of AGI behavior for unsafe deviations",
                enforcement_level=EnforcementLevel.HIGH,
                biological_markers=["GFAP", "NeuN"]
            ),
            "consciousness_flagging": SafetyConstraint(
                name="Consciousness Flagging",
                constraint_type=ConstraintType.BEHAVIORAL,
                description="Monitor for signs of emergent self-awareness or goal divergence",
                enforcement_level=EnforcementLevel.HIGH,
                biological_markers=["GFAP", "NeuN"]
            ),
            "communication_restrictions": SafetyConstraint(
                name="Communication Restrictions",
                constraint_type=ConstraintType.COMMUNICATION,
                description="Restrict communication to approved channels and protocols",
                enforcement_level=EnforcementLevel.MEDIUM,
                biological_markers=["NeuN", "NSE"]
            )
        }
        
        self.logger.info(f"🛡️ Initialized {len(constraints)} safety constraints")
        return constraints
    
    def add_constraint(self, 
                      name: str,
                      constraint_type: ConstraintType,
                      description: str,
                      enforcement_level: EnforcementLevel,
                      biological_markers: List[str]) -> str:
        """
        Add a new safety constraint
        
        Args:
            name: Name of the constraint
            constraint_type: Type of constraint
            description: Description of the constraint
            enforcement_level: Level of enforcement
            biological_markers: Required biological markers
            
        Returns:
            Constraint ID
        """
        if name in self.constraints:
            raise ValueError(f"Constraint '{name}' already exists")
        
        constraint = SafetyConstraint(
            name=name,
            constraint_type=constraint_type,
            description=description,
            enforcement_level=enforcement_level,
            biological_markers=biological_markers
        )
        
        self.constraints[name] = constraint
        
        self.logger.info(f"🛡️ Added new safety constraint: {name}")
        
        return name
    
    def remove_constraint(self, name: str) -> bool:
        """
        Remove a safety constraint
        
        Args:
            name: Name of the constraint to remove
            
        Returns:
            True if removed, False if not found
        """
        if name not in self.constraints:
            return False
        
        # Check if constraint is critical
        constraint = self.constraints[name]
        if constraint.enforcement_level == EnforcementLevel.CRITICAL:
            self.logger.warning(f"🛡️ Attempted to remove critical constraint: {name}")
            return False
        
        del self.constraints[name]
        
        self.logger.info(f"🛡️ Removed safety constraint: {name}")
        
        return True
    
    def modify_constraint(self, 
                         name: str,
                         **kwargs) -> bool:
        """
        Modify an existing safety constraint
        
        Args:
            name: Name of the constraint to modify
            **kwargs: Fields to modify
            
        Returns:
            True if modified, False if not found
        """
        if name not in self.constraints:
            return False
        
        constraint = self.constraints[name]
        
        # Check if constraint is critical
        if constraint.enforcement_level == EnforcementLevel.CRITICAL:
            self.logger.warning(f"🛡️ Attempted to modify critical constraint: {name}")
            return False
        
        # Update constraint fields
        for field, value in kwargs.items():
            if hasattr(constraint, field) and field not in ['name', 'constraint_hash', 'created_timestamp']:
                setattr(constraint, field, value)
        
        # Update modification timestamp and hash
        constraint.last_modified = datetime.now()
        constraint.constraint_hash = constraint._calculate_hash()
        
        self.logger.info(f"🛡️ Modified safety constraint: {name}")
        
        return True
    
    def check_constraint_compliance(self, 
                                  operation_type: str,
                                  operation_context: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Check if an operation complies with all relevant constraints
        
        Args:
            operation_type: Type of operation to check
            operation_context: Context of the operation
            
        Returns:
            Tuple of (compliant, violations)
        """
        violations = []
        compliant = True
        
        # Check each constraint
        for constraint_name, constraint in self.constraints.items():
            if not constraint.enforcement_active:
                continue
            
            if not self._check_constraint_compliance(constraint, operation_type, operation_context):
                violations.append(constraint_name)
                compliant = False
                
                # Log violation
                self._log_constraint_violation(constraint, operation_type, operation_context)
        
        return compliant, violations
    
    def _check_constraint_compliance(self, 
                                   constraint: SafetyConstraint,
                                   operation_type: str,
                                   operation_context: Dict[str, Any] = None) -> bool:
        """Check if an operation complies with a specific constraint"""
        # This is a simplified compliance check
        # In practice, would implement comprehensive compliance logic
        
        if constraint.constraint_type == ConstraintType.MODIFICATION:
            # Check for self-modification attempts
            if operation_context and operation_context.get("target") == "safety_system":
                return False
        
        elif constraint.constraint_type == ConstraintType.ACCESS:
            # Check for unauthorized access attempts
            if operation_context and operation_context.get("access_level") == "unauthorized":
                return False
        
        elif constraint.constraint_type == ConstraintType.REPRODUCTION:
            # Check for reproduction attempts
            if operation_context and operation_context.get("action") == "replicate":
                return False
        
        elif constraint.constraint_type == ConstraintType.COMMUNICATION:
            # Check for unauthorized communication
            if operation_context and operation_context.get("channel") == "unauthorized":
                return False
        
        # Default to compliant
        return True
    
    def _log_constraint_violation(self, 
                                 constraint: SafetyConstraint,
                                 operation_type: str,
                                 operation_context: Dict[str, Any] = None):
        """Log a constraint violation"""
        violation = ConstraintViolation(
            timestamp=datetime.now(),
            constraint_name=constraint.name,
            violation_type=operation_type,
            severity=constraint.enforcement_level.value,
            description=f"Violation of {constraint.name}: {constraint.description}",
            source="constraint_check",
            action_taken="logged",
            biological_validation={}  # Would be populated by biological validation
        )
        
        self.violations.append(violation)
        
        self.logger.warning(f"🛡️ Constraint violation detected: {constraint.name} for {operation_type}")
    
    def get_constraint_status(self, constraint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific constraint
        
        Args:
            constraint_name: Name of the constraint
            
        Returns:
            Constraint status or None if not found
        """
        if constraint_name not in self.constraints:
            return None
        
        constraint = self.constraints[constraint_name]
        
        return {
            "name": constraint.name,
            "type": constraint.constraint_type.value,
            "description": constraint.description,
            "enforcement_level": constraint.enforcement_level.value,
            "biological_markers": constraint.biological_markers,
            "validation_required": constraint.validation_required,
            "enforcement_active": constraint.enforcement_active,
            "created_timestamp": constraint.created_timestamp.isoformat(),
            "last_modified": constraint.last_modified.isoformat(),
            "constraint_hash": constraint.constraint_hash
        }
    
    def get_all_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all constraints"""
        return {
            name: self.get_constraint_status(name)
            for name in self.constraints.keys()
        }
    
    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[str]:
        """Get names of constraints by type"""
        return [
            name for name, constraint in self.constraints.items()
            if constraint.constraint_type == constraint_type
        ]
    
    def get_constraints_by_level(self, enforcement_level: EnforcementLevel) -> List[str]:
        """Get names of constraints by enforcement level"""
        return [
            name for name, constraint in self.constraints.items()
            if constraint.enforcement_level == enforcement_level
        ]
    
    def enable_constraint(self, constraint_name: str) -> bool:
        """Enable a constraint"""
        if constraint_name not in self.constraints:
            return False
        
        self.constraints[constraint_name].enforcement_active = True
        self.logger.info(f"🛡️ Enabled constraint: {constraint_name}")
        return True
    
    def disable_constraint(self, constraint_name: str) -> bool:
        """Disable a constraint (only for non-critical constraints)"""
        if constraint_name not in self.constraints:
            return False
        
        constraint = self.constraints[constraint_name]
        if constraint.enforcement_level == EnforcementLevel.CRITICAL:
            self.logger.warning(f"🛡️ Cannot disable critical constraint: {constraint_name}")
            return False
        
        constraint.enforcement_active = False
        self.logger.info(f"🛡️ Disabled constraint: {constraint_name}")
        return True
    
    def validate_constraint_integrity(self) -> Dict[str, Any]:
        """Validate integrity of all constraints"""
        integrity_report = {
            "timestamp": datetime.now().isoformat(),
            "total_constraints": len(self.constraints),
            "active_constraints": sum(1 for c in self.constraints.values() if c.enforcement_active),
            "integrity_checks": {},
            "overall_integrity": True
        }
        
        for name, constraint in self.constraints.items():
            # Check hash integrity
            current_hash = constraint._calculate_hash()
            hash_valid = current_hash == constraint.constraint_hash
            
            # Check biological markers
            markers_valid = len(constraint.biological_markers) > 0
            
            # Overall constraint validity
            constraint_valid = hash_valid and markers_valid
            
            integrity_report["integrity_checks"][name] = {
                "hash_valid": hash_valid,
                "markers_valid": markers_valid,
                "overall_valid": constraint_valid
            }
            
            if not constraint_valid:
                integrity_report["overall_integrity"] = False
        
        return integrity_report
    
    def export_constraints(self, output_path: str = "logs/safety_constraints.json") -> str:
        """Export constraints to file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "constraints": self.get_all_constraints(),
                "violations": [
                    {
                        "timestamp": v.timestamp.isoformat(),
                        "constraint_name": v.constraint_name,
                        "violation_type": v.violation_type,
                        "severity": v.severity,
                        "description": v.description,
                        "source": v.source,
                        "action_taken": v.action_taken
                    }
                    for v in self.violations
                ],
                "integrity_report": self.validate_constraint_integrity()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"🛡️ Safety constraints exported to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"🛡️ Failed to export constraints: {e}")
            raise
