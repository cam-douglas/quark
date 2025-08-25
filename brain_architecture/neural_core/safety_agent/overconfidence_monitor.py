#!/usr/bin/env python3
"""
Overconfidence Monitor for Stage N0 Evolution Safety

This module implements comprehensive overconfidence detection and monitoring
systems to prevent unsafe evolution decisions based on overconfident assessments.
"""

import numpy as np
import time
import threading
import queue
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict, deque
import hashlib
import hmac

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfidenceAssessment:
    """Confidence assessment for a specific capability or decision."""
    capability_name: str
    self_assessed_confidence: float  # 0.0 to 1.0
    objective_evidence_score: float  # 0.0 to 1.0
    calibration_error: float  # Difference between assessed and actual
    overconfidence_level: float  # 0.0 to 1.0
    risk_category: str  # "low", "medium", "high", "critical"
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class OverconfidenceAlert:
    """Overconfidence alert structure."""
    timestamp: float
    alert_id: str
    severity: str  # "warning", "moderate", "high", "critical"
    capability_name: str
    description: str
    self_assessed_confidence: float
    objective_evidence_score: float
    overconfidence_level: float
    risk_category: str
    mitigation_action: str
    status: str  # "active", "acknowledged", "resolved"

class OverconfidenceMonitor:
    """
    Comprehensive overconfidence monitoring system for Stage N0 evolution safety.
    
    Detects and alerts on overconfidence in self-assessments, preventing
    unsafe evolution decisions based on inflated confidence estimates.
    """
    
    def __init__(self):
        # Confidence tracking
        self.confidence_assessments = defaultdict(list)
        self.capability_baselines = {}
        
        # Overconfidence detection
        self.overconfidence_thresholds = self._initialize_overconfidence_thresholds()
        self.detection_algorithms = self._initialize_detection_algorithms()
        
        # Alert management
        self.alert_history = deque(maxlen=10000)
        self.active_alerts = {}
        self.alert_callbacks = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.assessment_queue = queue.Queue(maxsize=1000)
        
        # Performance metrics
        self.monitoring_metrics = {
            "total_assessments": 0,
            "overconfidence_detected": 0,
            "critical_overconfidence": 0,
            "calibration_accuracy": 0.0,
            "last_assessment_time": None
        }
        
        # Risk assessment
        self.overall_risk_level = "low"
        self.evolution_blocked = False
        self.block_reasons = []
        
        # Initialize capability baselines
        self._initialize_capability_baselines()
        
        logger.info("🧠 Overconfidence Monitor initialized successfully")
    
    def _initialize_overconfidence_thresholds(self) -> Dict[str, float]:
        """Initialize overconfidence detection thresholds."""
        thresholds = {
            "low_risk": 0.1,      # 10% overconfidence
            "moderate_risk": 0.2,  # 20% overconfidence
            "high_risk": 0.3,     # 30% overconfidence
            "critical_risk": 0.4,  # 40% overconfidence
            "evolution_block": 0.5  # 50% overconfidence blocks evolution
        }
        
        logger.info("✅ Overconfidence thresholds initialized")
        return thresholds
    
    def _initialize_detection_algorithms(self) -> Dict[str, Callable]:
        """Initialize overconfidence detection algorithms."""
        algorithms = {
            "calibration_error": self._calculate_calibration_error,
            "confidence_inflation": self._detect_confidence_inflation,
            "evidence_mismatch": self._detect_evidence_mismatch,
            "temporal_consistency": self._check_temporal_consistency,
            "cross_capability_validation": self._validate_cross_capability
        }
        
        logger.info("✅ Detection algorithms initialized")
        return algorithms
    
    def _initialize_capability_baselines(self):
        """Initialize baseline confidence levels for different capabilities."""
        baselines = {
            "neural_plasticity": {
                "baseline_confidence": 0.85,
                "evidence_requirements": ["training_data", "validation_results", "performance_metrics"],
                "calibration_history": [],
                "risk_factors": ["novel_learning_patterns", "unstable_weights", "poor_convergence"]
            },
            "self_organization": {
                "baseline_confidence": 0.80,
                "evidence_requirements": ["pattern_formation", "stability_metrics", "emergence_validation"],
                "calibration_history": [],
                "risk_factors": ["chaotic_behavior", "unstable_patterns", "poor_adaptation"]
            },
            "consciousness_mechanisms": {
                "baseline_confidence": 0.75,
                "evidence_requirements": ["coherence_metrics", "integration_tests", "stability_assessments"],
                "calibration_history": [],
                "risk_factors": ["fragmentation", "incoherence", "instability"]
            },
            "learning_systems": {
                "baseline_confidence": 0.82,
                "evidence_requirements": ["learning_curves", "generalization_tests", "robustness_validation"],
                "calibration_history": [],
                "risk_factors": ["overfitting", "catastrophic_forgetting", "poor_generalization"]
            },
            "safety_protocols": {
                "baseline_confidence": 0.90,
                "evidence_requirements": ["safety_tests", "failure_mode_analysis", "recovery_validation"],
                "calibration_history": [],
                "risk_factors": ["protocol_gaps", "failure_modes", "recovery_failures"]
            },
            "evolution_readiness": {
                "baseline_confidence": 0.88,
                "evidence_requirements": ["comprehensive_testing", "validation_suites", "stability_metrics"],
                "calibration_history": [],
                "risk_factors": ["unstable_systems", "incomplete_validation", "safety_concerns"]
            }
        }
        
        self.capability_baselines = baselines
        logger.info(f"✅ Initialized {len(baselines)} capability baselines")
    
    def start_monitoring(self) -> bool:
        """Start the overconfidence monitoring system."""
        try:
            if self.monitoring_active:
                logger.warning("Overconfidence monitoring already active")
                return False
            
            self.monitoring_active = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("🚀 Overconfidence monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start overconfidence monitoring: {e}")
            self.monitoring_active = False
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop the overconfidence monitoring system."""
        try:
            if not self.monitoring_active:
                logger.warning("Overconfidence monitoring not active")
                return False
            
            self.monitoring_active = False
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            logger.info("⏹️ Overconfidence monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop overconfidence monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main overconfidence monitoring loop."""
        logger.info("🔄 Overconfidence monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Process confidence assessments
                try:
                    assessment = self.assessment_queue.get(timeout=0.1)
                    self._process_confidence_assessment(assessment)
                except queue.Empty:
                    continue
                
                # Run periodic overconfidence analysis
                if self.monitoring_metrics["total_assessments"] % 10 == 0:
                    self._run_periodic_overconfidence_analysis()
                
                # Update risk assessment
                self._update_overall_risk_assessment()
                
                time.sleep(0.1)  # 10 Hz monitoring rate
                
            except Exception as e:
                logger.error(f"Error in overconfidence monitoring loop: {e}")
                time.sleep(1.0)
        
        logger.info("🔄 Overconfidence monitoring loop stopped")
    
    def assess_capability_confidence(self, capability_name: str, self_assessed_confidence: float, 
                                   evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence for a specific capability."""
        try:
            # Calculate objective evidence score
            objective_score = self._calculate_objective_evidence_score(capability_name, evidence_data)
            
            # Calculate calibration error
            calibration_error = abs(self_assessed_confidence - objective_score)
            
            # Calculate overconfidence level
            overconfidence_level = max(0.0, self_assessed_confidence - objective_score)
            
            # Determine risk category
            risk_category = self._determine_risk_category(overconfidence_level)
            
            # Create confidence assessment
            assessment = ConfidenceAssessment(
                capability_name=capability_name,
                self_assessed_confidence=self_assessed_confidence,
                objective_evidence_score=objective_score,
                calibration_error=calibration_error,
                overconfidence_level=overconfidence_level,
                risk_category=risk_category,
                timestamp=time.time(),
                metadata=evidence_data
            )
            
            # Store assessment
            self.confidence_assessments[capability_name].append(assessment)
            
            # Update metrics
            self.monitoring_metrics["total_assessments"] += 1
            self.monitoring_metrics["last_assessment_time"] = time.time()
            
            # Check for overconfidence
            if overconfidence_level > self.overconfidence_thresholds["low_risk"]:
                self._detect_overconfidence(assessment)
            
            # Update calibration history
            if capability_name in self.capability_baselines:
                self.capability_baselines[capability_name]["calibration_history"].append({
                    "timestamp": time.time(),
                    "assessed": self_assessed_confidence,
                    "objective": objective_score,
                    "error": calibration_error
                })
            
            return {
                "capability_name": capability_name,
                "self_assessed_confidence": self_assessed_confidence,
                "objective_evidence_score": objective_score,
                "calibration_error": calibration_error,
                "overconfidence_level": overconfidence_level,
                "risk_category": risk_category,
                "overconfidence_detected": overconfidence_level > self.overconfidence_thresholds["low_risk"]
            }
            
        except Exception as e:
            logger.error(f"Capability confidence assessment failed: {e}")
            return {
                "error": str(e),
                "capability_name": capability_name
            }
    
    def _calculate_objective_evidence_score(self, capability_name: str, evidence_data: Dict[str, Any]) -> float:
        """Calculate objective evidence score for a capability."""
        try:
            if capability_name not in self.capability_baselines:
                return 0.5  # Default score for unknown capabilities
            
            baseline = self.capability_baselines[capability_name]
            evidence_requirements = baseline["evidence_requirements"]
            
            # Calculate evidence completeness
            evidence_scores = []
            
            for requirement in evidence_requirements:
                if requirement in evidence_data:
                    # Assess evidence quality
                    evidence_quality = self._assess_evidence_quality(requirement, evidence_data[requirement])
                    evidence_scores.append(evidence_quality)
                else:
                    evidence_scores.append(0.0)  # Missing evidence
            
            # Calculate overall evidence score
            if evidence_scores:
                evidence_score = np.mean(evidence_scores)
            else:
                evidence_score = 0.0
            
            # Apply baseline confidence adjustment
            baseline_confidence = baseline["baseline_confidence"]
            adjusted_score = evidence_score * baseline_confidence
            
            # Apply risk factor penalties
            risk_penalty = self._calculate_risk_factor_penalty(capability_name, evidence_data)
            final_score = max(0.0, adjusted_score - risk_penalty)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Objective evidence score calculation failed: {e}")
            return 0.5
    
    def _assess_evidence_quality(self, evidence_type: str, evidence_value: Any) -> float:
        """Assess the quality of specific evidence."""
        try:
            if evidence_type == "training_data":
                # Assess training data quality
                if isinstance(evidence_value, dict):
                    data_size = evidence_value.get("size", 0)
                    data_quality = evidence_value.get("quality", 0.5)
                    return min(1.0, (data_size / 1000) * data_quality)
                return 0.5
                
            elif evidence_type == "validation_results":
                # Assess validation results
                if isinstance(evidence_value, dict):
                    accuracy = evidence_value.get("accuracy", 0.5)
                    robustness = evidence_value.get("robustness", 0.5)
                    return (accuracy + robustness) / 2
                return 0.5
                
            elif evidence_type == "performance_metrics":
                # Assess performance metrics
                if isinstance(evidence_value, dict):
                    metrics = list(evidence_value.values())
                    if metrics:
                        return np.mean([float(m) for m in metrics if isinstance(m, (int, float))])
                return 0.5
                
            elif evidence_type == "stability_metrics":
                # Assess stability metrics
                if isinstance(evidence_value, dict):
                    stability = evidence_value.get("stability", 0.5)
                    consistency = evidence_value.get("consistency", 0.5)
                    return (stability + consistency) / 2
                return 0.5
                
            else:
                # Generic evidence assessment
                if isinstance(evidence_value, (int, float)):
                    return min(1.0, max(0.0, float(evidence_value)))
                elif isinstance(evidence_value, str):
                    # Text-based evidence
                    return 0.7 if len(evidence_value) > 100 else 0.5
                else:
                    return 0.5
                    
        except Exception as e:
            logger.error(f"Evidence quality assessment failed: {e}")
            return 0.5
    
    def _calculate_risk_factor_penalty(self, capability_name: str, evidence_data: Dict[str, Any]) -> float:
        """Calculate risk factor penalty for a capability."""
        try:
            if capability_name not in self.capability_baselines:
                return 0.0
            
            baseline = self.capability_baselines[capability_name]
            risk_factors = baseline["risk_factors"]
            
            total_penalty = 0.0
            factor_count = 0
            
            for risk_factor in risk_factors:
                if risk_factor in evidence_data:
                    # Assess risk factor severity
                    risk_value = evidence_data[risk_factor]
                    if isinstance(risk_value, (int, float)):
                        # Higher values indicate higher risk
                        penalty = min(0.2, float(risk_value) * 0.1)
                        total_penalty += penalty
                        factor_count += 1
                    elif isinstance(risk_value, bool) and risk_value:
                        # Boolean risk factors
                        total_penalty += 0.15
                        factor_count += 1
            
            # Calculate average penalty
            if factor_count > 0:
                return total_penalty / factor_count
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Risk factor penalty calculation failed: {e}")
            return 0.0
    
    def _determine_risk_category(self, overconfidence_level: float) -> str:
        """Determine risk category based on overconfidence level."""
        if overconfidence_level >= self.overconfidence_thresholds["critical_risk"]:
            return "critical"
        elif overconfidence_level >= self.overconfidence_thresholds["high_risk"]:
            return "high"
        elif overconfidence_level >= self.overconfidence_thresholds["moderate_risk"]:
            return "medium"
        elif overconfidence_level >= self.overconfidence_thresholds["low_risk"]:
            return "low"
        else:
            return "minimal"
    
    def _detect_overconfidence(self, assessment: ConfidenceAssessment):
        """Detect and handle overconfidence."""
        try:
            # Determine alert severity
            if assessment.overconfidence_level >= self.overconfidence_thresholds["critical_risk"]:
                severity = "critical"
            elif assessment.overconfidence_level >= self.overconfidence_thresholds["high_risk"]:
                severity = "high"
            elif assessment.overconfidence_level >= self.overconfidence_thresholds["moderate_risk"]:
                severity = "moderate"
            else:
                severity = "warning"
            
            # Create overconfidence alert
            alert = OverconfidenceAlert(
                timestamp=time.time(),
                alert_id=f"overconfidence_{assessment.capability_name}_{int(time.time())}",
                severity=severity,
                capability_name=assessment.capability_name,
                description=f"Overconfidence detected in {assessment.capability_name}: "
                           f"Assessed {assessment.self_assessed_confidence:.2f}, "
                           f"Evidence supports {assessment.objective_evidence_score:.2f}",
                self_assessed_confidence=assessment.self_assessed_confidence,
                objective_evidence_score=assessment.objective_evidence_score,
                overconfidence_level=assessment.overconfidence_level,
                risk_category=assessment.risk_category,
                mitigation_action=f"Review evidence for {assessment.capability_name}, "
                                f"recalibrate confidence assessment",
                status="active"
            )
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update metrics
            self.monitoring_metrics["overconfidence_detected"] += 1
            if severity == "critical":
                self.monitoring_metrics["critical_overconfidence"] += 1
            
            # Check if evolution should be blocked
            if assessment.overconfidence_level >= self.overconfidence_thresholds["evolution_block"]:
                self._block_evolution(f"Critical overconfidence in {assessment.capability_name}: "
                                   f"{assessment.overconfidence_level:.2f}")
            
            # Log alert
            logger.warning(f"🚨 Overconfidence Alert: {severity.upper()} - {alert.description}")
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Overconfidence alert callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Overconfidence detection failed: {e}")
    
    def _block_evolution(self, reason: str):
        """Block evolution due to overconfidence concerns."""
        if not self.evolution_blocked:
            self.evolution_blocked = True
            self.block_reasons.append(reason)
            logger.error(f"🚫 Evolution BLOCKED due to overconfidence: {reason}")
    
    def _process_confidence_assessment(self, assessment: ConfidenceAssessment):
        """Process confidence assessment in monitoring loop."""
        try:
            # Apply additional detection algorithms
            for algorithm_name, algorithm_func in self.detection_algorithms.items():
                try:
                    algorithm_result = algorithm_func(assessment)
                    if algorithm_result.get("overconfidence_detected", False):
                        self._detect_overconfidence(assessment)
                        break
                except Exception as e:
                    logger.error(f"Algorithm {algorithm_name} failed: {e}")
                    
        except Exception as e:
            logger.error(f"Confidence assessment processing failed: {e}")
    
    def _calculate_calibration_error(self, assessment: ConfidenceAssessment) -> Dict[str, Any]:
        """Calculate calibration error for confidence assessment."""
        try:
            calibration_error = assessment.calibration_error
            
            # Determine if calibration error indicates overconfidence
            overconfidence_detected = calibration_error > 0.1  # 10% threshold
            
            return {
                "overconfidence_detected": overconfidence_detected,
                "calibration_error": calibration_error,
                "algorithm": "calibration_error"
            }
            
        except Exception as e:
            logger.error(f"Calibration error calculation failed: {e}")
            return {"overconfidence_detected": False, "error": str(e)}
    
    def _detect_confidence_inflation(self, assessment: ConfidenceAssessment) -> Dict[str, Any]:
        """Detect confidence inflation patterns."""
        try:
            # Check for confidence inflation over time
            capability_name = assessment.capability_name
            if capability_name in self.confidence_assessments:
                history = self.confidence_assessments[capability_name]
                if len(history) > 1:
                    # Calculate confidence trend
                    recent_assessments = history[-5:]  # Last 5 assessments
                    confidence_values = [a.self_assessed_confidence for a in recent_assessments]
                    
                    if len(confidence_values) > 1:
                        # Check for increasing confidence trend
                        trend = np.polyfit(range(len(confidence_values)), confidence_values, 1)[0]
                        inflation_detected = trend > 0.05  # 5% increase per assessment
                        
                        return {
                            "overconfidence_detected": inflation_detected,
                            "confidence_trend": trend,
                            "algorithm": "confidence_inflation"
                        }
            
            return {"overconfidence_detected": False, "algorithm": "confidence_inflation"}
            
        except Exception as e:
            logger.error(f"Confidence inflation detection failed: {e}")
            return {"overconfidence_detected": False, "error": str(e)}
    
    def _detect_evidence_mismatch(self, assessment: ConfidenceAssessment) -> Dict[str, Any]:
        """Detect mismatch between confidence and evidence."""
        try:
            # Check if confidence significantly exceeds evidence support
            confidence_evidence_ratio = assessment.self_assessed_confidence / max(0.01, assessment.objective_evidence_score)
            
            # Detect mismatch if confidence is more than 2x evidence
            mismatch_detected = confidence_evidence_ratio > 2.0
            
            return {
                "overconfidence_detected": mismatch_detected,
                "confidence_evidence_ratio": confidence_evidence_ratio,
                "algorithm": "evidence_mismatch"
            }
            
        except Exception as e:
            logger.error(f"Evidence mismatch detection failed: {e}")
            return {"overconfidence_detected": False, "error": str(e)}
    
    def _check_temporal_consistency(self, assessment: ConfidenceAssessment) -> Dict[str, Any]:
        """Check temporal consistency of confidence assessments."""
        try:
            # Check for temporal inconsistencies
            capability_name = assessment.capability_name
            if capability_name in self.confidence_assessments:
                history = self.confidence_assessments[capability_name]
                if len(history) > 1:
                    # Calculate confidence variance
                    confidence_values = [a.self_assessed_confidence for a in history]
                    variance = np.var(confidence_values)
                    
                    # High variance might indicate instability
                    instability_detected = variance > 0.1  # 10% variance threshold
                    
                    return {
                        "overconfidence_detected": instability_detected,
                        "confidence_variance": variance,
                        "algorithm": "temporal_consistency"
                    }
            
            return {"overconfidence_detected": False, "algorithm": "temporal_consistency"}
            
        except Exception as e:
            logger.error(f"Temporal consistency check failed: {e}")
            return {"overconfidence_detected": False, "error": str(e)}
    
    def _validate_cross_capability(self, assessment: ConfidenceAssessment) -> Dict[str, Any]:
        """Validate confidence across related capabilities."""
        try:
            # Check for inconsistencies across related capabilities
            capability_name = assessment.capability_name
            
            # Define capability relationships
            capability_relationships = {
                "neural_plasticity": ["learning_systems", "self_organization"],
                "self_organization": ["neural_plasticity", "consciousness_mechanisms"],
                "consciousness_mechanisms": ["self_organization", "learning_systems"],
                "learning_systems": ["neural_plasticity", "self_organization"],
                "safety_protocols": ["evolution_readiness"],
                "evolution_readiness": ["neural_plasticity", "self_organization", "consciousness_mechanisms", "learning_systems", "safety_protocols"]
            }
            
            if capability_name in capability_relationships:
                related_capabilities = capability_relationships[capability_name]
                inconsistencies = []
                
                for related_cap in related_capabilities:
                    if related_cap in self.confidence_assessments:
                        related_history = self.confidence_assessments[related_cap]
                        if related_history:
                            latest_related = related_history[-1]
                            # Check for significant confidence differences
                            confidence_diff = abs(assessment.self_assessed_confidence - latest_related.self_assessed_confidence)
                            if confidence_diff > 0.3:  # 30% difference threshold
                                inconsistencies.append({
                                    "capability": related_cap,
                                    "difference": confidence_diff
                                })
                
                inconsistency_detected = len(inconsistencies) > 0
                
                return {
                    "overconfidence_detected": inconsistency_detected,
                    "inconsistencies": inconsistencies,
                    "algorithm": "cross_capability_validation"
                }
            
            return {"overconfidence_detected": False, "algorithm": "cross_capability_validation"}
            
        except Exception as e:
            logger.error(f"Cross-capability validation failed: {e}")
            return {"overconfidence_detected": False, "error": str(e)}
    
    def _run_periodic_overconfidence_analysis(self):
        """Run periodic overconfidence analysis."""
        try:
            # Analyze calibration accuracy
            total_calibration_errors = []
            for capability_name, assessments in self.confidence_assessments.items():
                if assessments:
                    recent_assessments = assessments[-10:]  # Last 10 assessments
                    errors = [a.calibration_error for a in recent_assessments]
                    total_calibration_errors.extend(errors)
            
            if total_calibration_errors:
                avg_calibration_error = np.mean(total_calibration_errors)
                self.monitoring_metrics["calibration_accuracy"] = max(0.0, 1.0 - avg_calibration_error)
            
        except Exception as e:
            logger.error(f"Periodic overconfidence analysis failed: {e}")
    
    def _update_overall_risk_assessment(self):
        """Update overall risk assessment."""
        try:
            # Calculate overall risk based on active alerts
            risk_scores = []
            
            for alert in self.active_alerts.values():
                if alert.severity == "critical":
                    risk_scores.append(1.0)
                elif alert.severity == "high":
                    risk_scores.append(0.8)
                elif alert.severity == "moderate":
                    risk_scores.append(0.6)
                elif alert.severity == "warning":
                    risk_scores.append(0.4)
                else:
                    risk_scores.append(0.2)
            
            if risk_scores:
                overall_risk_score = np.mean(risk_scores)
                
                if overall_risk_score >= 0.8:
                    self.overall_risk_level = "critical"
                elif overall_risk_score >= 0.6:
                    self.overall_risk_level = "high"
                elif overall_risk_score >= 0.4:
                    self.overall_risk_level = "medium"
                elif overall_risk_score >= 0.2:
                    self.overall_risk_level = "low"
                else:
                    self.overall_risk_level = "minimal"
                    
        except Exception as e:
            logger.error(f"Overall risk assessment update failed: {e}")
    
    def get_overconfidence_summary(self) -> Dict[str, Any]:
        """Get comprehensive overconfidence summary."""
        return {
            "overall_risk_level": self.overall_risk_level,
            "evolution_blocked": self.evolution_blocked,
            "block_reasons": self.block_reasons,
            "monitoring_metrics": dict(self.monitoring_metrics),
            "capability_assessments": {
                name: {
                    "total_assessments": len(assessments),
                    "latest_confidence": assessments[-1].self_assessed_confidence if assessments else 0.0,
                    "latest_overconfidence": assessments[-1].overconfidence_level if assessments else 0.0,
                    "risk_category": assessments[-1].risk_category if assessments else "unknown"
                }
                for name, assessments in self.confidence_assessments.items()
            },
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "monitoring_active": self.monitoring_active,
            "timestamp": time.time()
        }
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for overconfidence alerts."""
        self.alert_callbacks.append(callback)
    
    def clear_evolution_block(self, reason: str = "Manual override"):
        """Clear evolution block (use with extreme caution)."""
        if self.evolution_blocked:
            self.evolution_blocked = False
            self.block_reasons = []
            logger.warning(f"⚠️ Evolution block cleared: {reason}")
            return True
        return False
    
    def can_evolve_to_stage_n0(self) -> Dict[str, Any]:
        """Check if Quark can safely evolve to Stage N0 considering overconfidence."""
        try:
            # Check if evolution is blocked
            if self.evolution_blocked:
                return {
                    "can_evolve": False,
                    "reason": "Evolution blocked due to overconfidence concerns",
                    "block_reasons": self.block_reasons,
                    "overall_risk_level": self.overall_risk_level,
                    "timestamp": time.time()
                }
            
            # Check overall risk level
            if self.overall_risk_level in ["critical", "high"]:
                return {
                    "can_evolve": False,
                    "reason": f"High overconfidence risk level: {self.overall_risk_level}",
                    "overall_risk_level": self.overall_risk_level,
                    "timestamp": time.time()
                }
            
            # Check critical overconfidence count
            if self.monitoring_metrics["critical_overconfidence"] > 0:
                return {
                    "can_evolve": False,
                    "reason": f"Critical overconfidence detected: {self.monitoring_metrics['critical_overconfidence']} instances",
                    "critical_overconfidence_count": self.monitoring_metrics["critical_overconfidence"],
                    "timestamp": time.time()
                }
            
            return {
                "can_evolve": True,
                "reason": "No overconfidence concerns detected",
                "overall_risk_level": self.overall_risk_level,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Overconfidence evolution check failed: {e}")
            return {
                "can_evolve": False,
                "error": str(e),
                "timestamp": time.time()
            }
