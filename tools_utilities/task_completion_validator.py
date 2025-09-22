#!/usr/bin/env python3
"""
Task Completion Confidence Validator

Applies anti-overconfidence principles to task completion claims.
Treats "task is done" with the same skepticism as any technical claim.

CORE PRINCIPLE: Never claim a task is complete without high confidence and clear evidence.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

# Import test execution validator
try:
    from .test_execution_validator import execute_test_with_confidence_validation, TestConfidenceLevel
    TEST_VALIDATION_AVAILABLE = True
except ImportError:
    try:
        from test_execution_validator import execute_test_with_confidence_validation, TestConfidenceLevel
        TEST_VALIDATION_AVAILABLE = True
    except ImportError:
        logger.warning("⚠️ Test execution validator not available")
        TEST_VALIDATION_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCompletionStatus(Enum):
    """Task completion status with confidence requirements"""
    INCOMPLETE = "incomplete"           # <40% confidence - clearly not done
    PARTIALLY_COMPLETE = "partial"      # 40-60% confidence - some progress
    LIKELY_COMPLETE = "likely"          # 60-75% confidence - probably done
    CONFIDENTLY_COMPLETE = "complete"   # >75% confidence - high evidence of completion
    VERIFIED_COMPLETE = "verified"      # >85% confidence - external verification


class EvidenceType(Enum):
    """Types of evidence for task completion"""
    TEST_RESULTS = "test_results"
    VALIDATED_TEST_RESULTS = "validated_test_results"  # Tests that have been confidence-validated
    FILE_OUTPUTS = "file_outputs"
    FUNCTIONAL_VERIFICATION = "functional_verification"
    USER_ACCEPTANCE = "user_acceptance"
    METRICS_ACHIEVED = "metrics_achieved"
    DOCUMENTATION_COMPLETE = "documentation_complete"
    ERROR_FREE_EXECUTION = "error_free_execution"
    INTEGRATION_VERIFIED = "integration_verified"


@dataclass
class TaskEvidence:
    """Evidence supporting task completion"""
    evidence_type: EvidenceType
    description: str
    confidence_weight: float  # 0.0 to 1.0
    verification_method: str
    timestamp: datetime
    details: Dict[str, Any]
    
    @property
    def strength_score(self) -> float:
        """Calculate evidence strength based on type and verification"""
        base_strength = {
            EvidenceType.VALIDATED_TEST_RESULTS: 0.95,  # Highest - tests that passed confidence validation
            EvidenceType.USER_ACCEPTANCE: 0.90,
            EvidenceType.METRICS_ACHIEVED: 0.85,
            EvidenceType.FUNCTIONAL_VERIFICATION: 0.80,
            EvidenceType.INTEGRATION_VERIFIED: 0.80,
            EvidenceType.TEST_RESULTS: 0.60,  # Lower - just "tests passed" without validation
            EvidenceType.ERROR_FREE_EXECUTION: 0.70,
            EvidenceType.FILE_OUTPUTS: 0.60,
            EvidenceType.DOCUMENTATION_COMPLETE: 0.50
        }
        
        return base_strength.get(self.evidence_type, 0.5) * self.confidence_weight


class TaskCompletionValidator:
    """
    Validates task completion claims with anti-overconfidence principles
    
    NEVER claims a task is complete without substantial evidence.
    Applies the same skeptical approach used for technical validation.
    """
    
    def __init__(self):
        self.completion_history: Dict[str, List[Dict]] = {}
        self.evidence_requirements = self._initialize_evidence_requirements()
        
    def _initialize_evidence_requirements(self) -> Dict[str, Dict]:
        """Define evidence requirements for different confidence levels"""
        return {
            'high_confidence': {
                'min_evidence_count': 3,
                'required_types': [EvidenceType.VALIDATED_TEST_RESULTS, EvidenceType.FUNCTIONAL_VERIFICATION],
                'preferred_types': [EvidenceType.VALIDATED_TEST_RESULTS],  # Prefer confidence-validated tests
                'min_total_strength': 2.2,
                'confidence_threshold': 0.75
            },
            'moderate_confidence': {
                'min_evidence_count': 2,
                'required_types': [EvidenceType.FUNCTIONAL_VERIFICATION],
                'preferred_types': [EvidenceType.TEST_RESULTS, EvidenceType.VALIDATED_TEST_RESULTS],
                'min_total_strength': 1.4,
                'confidence_threshold': 0.60
            },
            'low_confidence': {
                'min_evidence_count': 1,
                'required_types': [],
                'preferred_types': [],
                'min_total_strength': 0.5,
                'confidence_threshold': 0.40
            }
        }
    
    def validate_task_completion(self, 
                               task_id: str,
                               task_description: str,
                               claimed_completion: bool,
                               evidence_list: List[TaskEvidence],
                               additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate a task completion claim with anti-overconfidence principles
        
        Returns detailed confidence assessment and evidence breakdown
        """
        logger.info(f"🔍 Validating completion claim for task: {task_id}")
        
        # Start with deep skepticism about completion
        base_completion_confidence = 0.10  # Very skeptical about "done"
        
        # Analyze evidence
        evidence_analysis = self._analyze_evidence(evidence_list)
        total_evidence_strength = evidence_analysis['total_strength']
        evidence_gaps = evidence_analysis['gaps']
        
        # Calculate completion confidence with heavy skepticism
        confidence_from_evidence = min(total_evidence_strength * 0.25, 0.6)  # Cap evidence contribution
        
        # Apply skepticism penalties
        skepticism_penalties = self._calculate_skepticism_penalties(
            task_description, evidence_list, additional_context or {}
        )
        
        # Calculate raw completion confidence
        raw_confidence = base_completion_confidence + confidence_from_evidence - skepticism_penalties
        
        # Apply natural ceiling - completion claims need exceptional evidence
        if raw_confidence > 0.70:
            exceptional_criteria = self._check_exceptional_completion_criteria(evidence_list)
            if not exceptional_criteria['meets_requirements']:
                # Compress high confidence naturally
                excess = raw_confidence - 0.70
                raw_confidence = 0.70 + (excess * 0.2)
                logger.info(f"🤔 Reduced completion confidence from {raw_confidence + excess * 0.8:.1%} to {raw_confidence:.1%}")
                logger.info(f"   Exceptional criteria not met: {exceptional_criteria['missing']}")
        
        # Final confidence with hard cap
        final_confidence = min(raw_confidence, 0.90)
        
        # Determine completion status
        completion_status = self._determine_completion_status(final_confidence, evidence_analysis)
        
        # Generate detailed breakdown
        result = {
            'task_id': task_id,
            'claimed_complete': claimed_completion,
            'actual_status': completion_status.value,
            'completion_confidence': final_confidence,
            'confidence_level': self._get_confidence_description(final_confidence),
            'evidence_analysis': evidence_analysis,
            'skepticism_applied': skepticism_penalties,
            'completion_breakdown': {
                'base_skepticism': base_completion_confidence,
                'evidence_contribution': confidence_from_evidence,
                'skepticism_penalties': skepticism_penalties,
                'natural_ceiling_applied': raw_confidence != final_confidence,
                'reasoning': f"Started skeptical ({base_completion_confidence:.1%}), added evidence ({confidence_from_evidence:.1%}), applied penalties (-{skepticism_penalties:.1%})"
            },
            'evidence_gaps': evidence_gaps,
            'recommended_actions': self._generate_completion_recommendations(completion_status, evidence_gaps),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self._record_completion_validation(task_id, result)
        
        return result
    
    def validate_test_evidence(self,
                             test_command: str,
                             task_requirements: List[str],
                             user_intentions: List[str],
                             working_directory: Optional[Path] = None) -> TaskEvidence:
        """
        Execute and validate a test with anti-overconfidence principles
        
        Returns TaskEvidence with confidence-validated test results
        """
        if not TEST_VALIDATION_AVAILABLE:
            logger.warning("⚠️ Test validation not available - using basic test evidence")
            return TaskEvidence(
                evidence_type=EvidenceType.TEST_RESULTS,
                description=f"Test executed: {test_command}",
                confidence_weight=0.5,  # Lower confidence without validation
                verification_method="basic_execution",
                timestamp=datetime.now(),
                details={'test_command': test_command, 'validation_available': False}
            )
        
        logger.info(f"🧪 Validating test evidence with anti-overconfidence: {test_command}")
        
        # Execute test with confidence validation
        test_result = execute_test_with_confidence_validation(
            test_command=test_command,
            requirements=task_requirements,
            user_intentions=user_intentions,
            working_directory=working_directory
        )
        
        # Determine evidence type based on test confidence
        test_confidence = test_result['confidence']
        should_trust, trust_reason = test_result['should_trust']
        
        if test_confidence >= 0.75 and should_trust:
            evidence_type = EvidenceType.VALIDATED_TEST_RESULTS
            description = f"Test passed with high confidence ({test_confidence:.1%}): {test_command}"
            confidence_weight = min(test_confidence, 0.95)  # Cap at 95%
        elif test_confidence >= 0.60:
            evidence_type = EvidenceType.TEST_RESULTS
            description = f"Test likely passed ({test_confidence:.1%}): {test_command}"
            confidence_weight = test_confidence * 0.8  # Reduce for uncertainty
        else:
            evidence_type = EvidenceType.ERROR_FREE_EXECUTION
            description = f"Test executed but validation unclear ({test_confidence:.1%}): {test_command}"
            confidence_weight = max(test_confidence * 0.5, 0.1)  # Very low confidence
        
        # Create evidence with detailed test validation results
        evidence = TaskEvidence(
            evidence_type=evidence_type,
            description=description,
            confidence_weight=confidence_weight,
            verification_method="confidence_validated_testing",
            timestamp=datetime.now(),
            details={
                'test_command': test_command,
                'test_confidence': test_confidence,
                'confidence_level': test_result['confidence_level'],
                'should_trust': should_trust,
                'trust_reason': trust_reason,
                'requirement_coverage': test_result['requirement_coverage'],
                'intention_alignment': test_result['intention_alignment'],
                'evidence_count': test_result['evidence_count'],
                'execution_time': test_result['execution_time'],
                'validation_report': test_result['validation_report']
            }
        )
        
        logger.info(f"🔍 Test evidence confidence: {confidence_weight:.1%}")
        logger.info(f"📋 Evidence type: {evidence_type.value}")
        
        return evidence
    
    def execute_and_validate_tests(self,
                                 test_commands: List[str],
                                 task_requirements: List[str],
                                 user_intentions: List[str],
                                 working_directory: Optional[Path] = None) -> List[TaskEvidence]:
        """
        Execute multiple tests and return validated evidence for each
        
        Applies anti-overconfidence to each test individually
        """
        evidence_list = []
        
        for test_command in test_commands:
            try:
                evidence = self.validate_test_evidence(
                    test_command=test_command,
                    task_requirements=task_requirements,
                    user_intentions=user_intentions,
                    working_directory=working_directory
                )
                evidence_list.append(evidence)
                
            except Exception as e:
                logger.error(f"❌ Test validation failed for '{test_command}': {e}")
                # Add failed test as weak evidence
                evidence_list.append(TaskEvidence(
                    evidence_type=EvidenceType.ERROR_FREE_EXECUTION,
                    description=f"Test failed to execute: {test_command}",
                    confidence_weight=0.0,
                    verification_method="failed_execution",
                    timestamp=datetime.now(),
                    details={'error': str(e), 'test_command': test_command}
                ))
        
        return evidence_list
    
    def _analyze_evidence(self, evidence_list: List[TaskEvidence]) -> Dict[str, Any]:
        """Analyze provided evidence for task completion"""
        if not evidence_list:
            return {
                'total_strength': 0.0,
                'evidence_count': 0,
                'types_covered': [],
                'gaps': ['No evidence provided for task completion'],
                'strongest_evidence': None,
                'weakest_evidence': None
            }
        
        total_strength = sum(evidence.strength_score for evidence in evidence_list)
        types_covered = list(set(evidence.evidence_type for evidence in evidence_list))
        
        # Identify gaps in evidence
        gaps = []
        critical_types = [EvidenceType.TEST_RESULTS, EvidenceType.FUNCTIONAL_VERIFICATION]
        for critical_type in critical_types:
            if critical_type not in types_covered:
                gaps.append(f"Missing {critical_type.value} evidence")
        
        if len(evidence_list) < 2:
            gaps.append("Insufficient evidence count (need at least 2 pieces)")
        
        if total_strength < 1.0:
            gaps.append("Evidence strength below minimum threshold")
        
        # Find strongest and weakest evidence
        strongest = max(evidence_list, key=lambda e: e.strength_score)
        weakest = min(evidence_list, key=lambda e: e.strength_score)
        
        return {
            'total_strength': total_strength,
            'evidence_count': len(evidence_list),
            'types_covered': [t.value for t in types_covered],
            'gaps': gaps,
            'strongest_evidence': {
                'type': strongest.evidence_type.value,
                'strength': strongest.strength_score,
                'description': strongest.description
            },
            'weakest_evidence': {
                'type': weakest.evidence_type.value,
                'strength': weakest.strength_score,
                'description': weakest.description
            }
        }
    
    def _calculate_skepticism_penalties(self, 
                                      task_description: str, 
                                      evidence_list: List[TaskEvidence],
                                      context: Dict[str, Any]) -> float:
        """Calculate skepticism penalties for completion claims"""
        penalties = 0.0
        
        # Complexity penalty - complex tasks are harder to complete
        if len(task_description.split()) > 20:
            penalties += 0.08  # Long descriptions = complex tasks
        
        if any(word in task_description.lower() for word in ['integrate', 'system', 'complex', 'multiple']):
            penalties += 0.06  # Complex integration tasks
        
        # Recency penalty - recent claims are more suspect
        recent_evidence = [e for e in evidence_list if (datetime.now() - e.timestamp).days < 1]
        if len(recent_evidence) == len(evidence_list):
            penalties += 0.05  # All evidence is very recent
        
        # Self-reported penalty - self-validation is less reliable
        self_reported = sum(1 for e in evidence_list if 'self' in e.verification_method.lower())
        if self_reported > 0:
            penalties += self_reported * 0.04
        
        # Missing critical evidence penalty
        has_validated_tests = any(e.evidence_type == EvidenceType.VALIDATED_TEST_RESULTS for e in evidence_list)
        has_basic_tests = any(e.evidence_type == EvidenceType.TEST_RESULTS for e in evidence_list)
        has_functional = any(e.evidence_type == EvidenceType.FUNCTIONAL_VERIFICATION for e in evidence_list)
        
        if not has_validated_tests and not has_basic_tests:
            penalties += 0.15  # No test results at all
        elif not has_validated_tests and has_basic_tests:
            penalties += 0.08  # Only basic tests, no confidence validation
        
        if not has_functional:
            penalties += 0.08  # No functional verification
        
        # Additional penalty for relying only on unvalidated test results
        unvalidated_test_count = sum(1 for e in evidence_list if e.evidence_type == EvidenceType.TEST_RESULTS)
        if unvalidated_test_count > 0 and not has_validated_tests:
            penalties += unvalidated_test_count * 0.05  # Penalty per unvalidated test
        
        logger.info(f"📉 Applied skepticism penalties: {penalties:.1%}")
        return penalties
    
    def _check_exceptional_completion_criteria(self, evidence_list: List[TaskEvidence]) -> Dict[str, Any]:
        """Check if evidence meets exceptional completion criteria"""
        criteria = {
            'multiple_evidence_types': len(set(e.evidence_type for e in evidence_list)) >= 3,
            'high_strength_evidence': any(e.strength_score >= 0.8 for e in evidence_list),
            'external_verification': any('external' in e.verification_method.lower() for e in evidence_list),
            'test_coverage': any(e.evidence_type == EvidenceType.TEST_RESULTS for e in evidence_list),
            'functional_verification': any(e.evidence_type == EvidenceType.FUNCTIONAL_VERIFICATION for e in evidence_list)
        }
        
        met_criteria = sum(criteria.values())
        missing = [k for k, v in criteria.items() if not v]
        
        return {
            'meets_requirements': met_criteria >= 4,  # Need 4/5 criteria
            'criteria_met': met_criteria,
            'total_criteria': len(criteria),
            'missing': missing
        }
    
    def _determine_completion_status(self, confidence: float, evidence_analysis: Dict) -> TaskCompletionStatus:
        """Determine task completion status based on confidence and evidence"""
        if confidence >= 0.85 and evidence_analysis['evidence_count'] >= 3:
            return TaskCompletionStatus.VERIFIED_COMPLETE
        elif confidence >= 0.75:
            return TaskCompletionStatus.CONFIDENTLY_COMPLETE
        elif confidence >= 0.60:
            return TaskCompletionStatus.LIKELY_COMPLETE
        elif confidence >= 0.40:
            return TaskCompletionStatus.PARTIALLY_COMPLETE
        else:
            return TaskCompletionStatus.INCOMPLETE
    
    def _get_confidence_description(self, confidence: float) -> str:
        """Get human-readable confidence description"""
        if confidence >= 0.85:
            return "⭐ VERIFIED COMPLETE - Exceptional evidence of completion"
        elif confidence >= 0.75:
            return "✅ HIGH CONFIDENCE - Strong evidence of completion"
        elif confidence >= 0.60:
            return "🟡 MODERATE CONFIDENCE - Some evidence of completion"
        elif confidence >= 0.40:
            return "⚠️ LOW CONFIDENCE - Limited evidence of completion"
        else:
            return "❌ VERY LOW CONFIDENCE - Insufficient evidence of completion"
    
    def _generate_completion_recommendations(self, 
                                          status: TaskCompletionStatus, 
                                          gaps: List[str]) -> List[str]:
        """Generate recommendations for improving completion confidence"""
        recommendations = []
        
        if status in [TaskCompletionStatus.INCOMPLETE, TaskCompletionStatus.PARTIALLY_COMPLETE]:
            recommendations.append("❌ DO NOT mark task as complete - insufficient evidence")
            recommendations.append("🔍 Gather more evidence before claiming completion")
        
        for gap in gaps:
            if "test" in gap.lower():
                recommendations.append("🧪 Run comprehensive tests to verify functionality")
            elif "functional" in gap.lower():
                recommendations.append("⚙️ Perform functional verification of all requirements")
            elif "evidence count" in gap.lower():
                recommendations.append("📊 Collect additional evidence from multiple sources")
        
        if status == TaskCompletionStatus.LIKELY_COMPLETE:
            recommendations.append("🤔 Consider additional verification before marking complete")
            recommendations.append("📋 Document all completion evidence clearly")
        
        return recommendations
    
    def _record_completion_validation(self, task_id: str, result: Dict[str, Any]):
        """Record completion validation in history"""
        if task_id not in self.completion_history:
            self.completion_history[task_id] = []
        
        self.completion_history[task_id].append({
            'timestamp': result['validation_timestamp'],
            'confidence': result['completion_confidence'],
            'status': result['actual_status'],
            'evidence_count': result['evidence_analysis']['evidence_count']
        })
    
    def generate_completion_report(self, task_id: str, result: Dict[str, Any]) -> str:
        """Generate detailed completion report with anti-overconfidence language"""
        confidence = result['completion_confidence']
        status = result['actual_status']
        
        report = f"""
🔍 TASK COMPLETION VALIDATION REPORT

Task ID: {task_id}
Claimed Status: {"Complete" if result['claimed_complete'] else "Incomplete"}
Actual Assessment: {status.upper()}

{result['confidence_level']}

WHAT I'M REASONABLY CONFIDENT ABOUT:
"""
        
        if confidence > 0.60:
            evidence = result['evidence_analysis']
            report += f"- Task has {evidence['evidence_count']} pieces of supporting evidence\n"
            report += f"- Evidence types covered: {', '.join(evidence['types_covered'])}\n"
            if evidence['strongest_evidence']:
                report += f"- Strongest evidence: {evidence['strongest_evidence']['description']}\n"
        else:
            report += "- Very limited confidence in task completion\n"
        
        # Add confidence calculation breakdown
        breakdown = result['completion_breakdown']
        report += f"""
📊 COMPLETION CONFIDENCE CALCULATION:
- Started with deep skepticism: {breakdown['base_skepticism']:.1%}
- Evidence contribution: +{breakdown['evidence_contribution']:.1%}
- Skepticism penalties: -{breakdown['skepticism_penalties']:.1%}
- Final confidence: {confidence:.1%}
- Reasoning: {breakdown['reasoning']}
"""
        
        report += """
WHAT I'M UNCERTAIN ABOUT:
"""
        for gap in result['evidence_gaps']:
            report += f"- {gap}\n"
        
        report += """
RECOMMENDED ACTIONS:
"""
        for recommendation in result['recommended_actions']:
            report += f"{recommendation}\n"
        
        report += f"""
🤔 MANDATORY COMPLETION UNCERTAINTY REMINDERS:
- No task can ever be 100% complete without exceptional evidence
- Always consider what might still be missing or broken
- Seek additional verification before claiming completion
- Remember: Claiming "done" prematurely causes more problems than thorough validation

⚠️ COMPLETION CONFIDENCE HARD CAP: Maximum 90% - Absolute completion certainty is forbidden
"""
        
        return report


# Integration functions for existing systems
def validate_task_completion_claim(task_id: str, 
                                 task_description: str, 
                                 evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to validate task completion with evidence
    
    Usage:
    evidence = [
        {
            'type': 'test_results',
            'description': 'All unit tests pass',
            'verification_method': 'automated_testing',
            'confidence_weight': 0.9,
            'details': {'test_count': 15, 'pass_rate': 1.0}
        }
    ]
    result = validate_task_completion_claim('task_123', 'Implement feature X', evidence)
    """
    validator = TaskCompletionValidator()
    
    # Convert evidence dictionaries to TaskEvidence objects
    evidence_objects = []
    for ev in evidence:
        evidence_objects.append(TaskEvidence(
            evidence_type=EvidenceType(ev.get('type', 'functional_verification')),
            description=ev['description'],
            confidence_weight=ev.get('confidence_weight', 0.5),
            verification_method=ev.get('verification_method', 'self_reported'),
            timestamp=datetime.now(),
            details=ev.get('details', {})
        ))
    
    return validator.validate_task_completion(
        task_id=task_id,
        task_description=task_description,
        claimed_completion=True,
        evidence_list=evidence_objects
    )


def should_mark_task_complete(completion_result: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Determine if a task should be marked as complete based on validation result
    
    Returns (should_complete, reason)
    """
    confidence = completion_result['completion_confidence']
    status = completion_result['actual_status']
    
    # Only mark complete with high confidence
    if confidence >= 0.75 and status in ['complete', 'verified']:
        return True, f"High confidence ({confidence:.1%}) with strong evidence"
    elif confidence >= 0.60:
        return False, f"Moderate confidence ({confidence:.1%}) - need more evidence"
    else:
        return False, f"Low confidence ({confidence:.1%}) - significant work remaining"


if __name__ == "__main__":
    # Example usage
    validator = TaskCompletionValidator()
    
    # Test with minimal evidence (should be low confidence)
    minimal_evidence = [
        TaskEvidence(
            evidence_type=EvidenceType.FILE_OUTPUTS,
            description="Created output file",
            confidence_weight=0.6,
            verification_method="self_reported",
            timestamp=datetime.now(),
            details={"file_count": 1}
        )
    ]
    
    result = validator.validate_task_completion(
        task_id="test_task_1",
        task_description="Create a simple output file",
        claimed_completion=True,
        evidence_list=minimal_evidence
    )
    
    print("🔍 TESTING TASK COMPLETION VALIDATION")
    print("="*50)
    print(f"Completion confidence: {result['completion_confidence']:.1%}")
    print(f"Status: {result['actual_status']}")
    print(f"Should mark complete: {should_mark_task_complete(result)}")
    
    # Generate report
    report = validator.generate_completion_report("test_task_1", result)
    print("\n" + report)
