#!/usr/bin/env python3
"""
Test Execution Confidence Validator

Applies anti-overconfidence principles to test execution and validation.
NEVER assumes a test "passed" without substantial evidence that it actually verified the intended behavior.

CORE PRINCIPLES:
1. Test execution ≠ Test validation
2. No errors ≠ Test passed
3. Test passed ≠ Requirements met
4. Requirements met ≠ User intentions fulfilled

Treats "test passed" claims with the same skepticism as any other technical assertion.
"""

import logging
import subprocess
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConfidenceLevel(Enum):
    """Test confidence levels with strict requirements"""
    EXECUTION_FAILED = "execution_failed"           # Test couldn't run
    EXECUTION_UNCLEAR = "execution_unclear"         # Test ran but results unclear
    LIKELY_PASSED = "likely_passed"                 # 40-60% confidence test worked
    PROBABLY_PASSED = "probably_passed"             # 60-75% confidence with good evidence
    CONFIDENTLY_PASSED = "confidently_passed"      # >75% confidence with strong validation
    VERIFIED_PASSED = "verified_passed"             # >85% confidence with external verification


class TestValidationType(Enum):
    """Types of test validation evidence"""
    OUTPUT_ANALYSIS = "output_analysis"
    ASSERTION_VERIFICATION = "assertion_verification"
    BEHAVIOR_CONFIRMATION = "behavior_confirmation"
    REQUIREMENT_MAPPING = "requirement_mapping"
    USER_INTENTION_ALIGNMENT = "user_intention_alignment"
    INTEGRATION_VERIFICATION = "integration_verification"
    ERROR_ABSENCE_CONFIRMATION = "error_absence_confirmation"
    PERFORMANCE_VALIDATION = "performance_validation"


@dataclass
class TestValidationEvidence:
    """Evidence supporting test validation confidence"""
    validation_type: TestValidationType
    description: str
    confidence_weight: float  # 0.0 to 1.0
    verification_method: str
    evidence_details: Dict[str, Any]
    timestamp: datetime
    
    @property
    def strength_score(self) -> float:
        """Calculate evidence strength"""
        base_strength = {
            TestValidationType.USER_INTENTION_ALIGNMENT: 0.95,
            TestValidationType.REQUIREMENT_MAPPING: 0.90,
            TestValidationType.BEHAVIOR_CONFIRMATION: 0.85,
            TestValidationType.ASSERTION_VERIFICATION: 0.80,
            TestValidationType.INTEGRATION_VERIFICATION: 0.75,
            TestValidationType.PERFORMANCE_VALIDATION: 0.70,
            TestValidationType.OUTPUT_ANALYSIS: 0.60,
            TestValidationType.ERROR_ABSENCE_CONFIRMATION: 0.50
        }
        
        return base_strength.get(self.validation_type, 0.5) * self.confidence_weight


@dataclass
class TestExecutionResult:
    """Comprehensive test execution analysis"""
    test_command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timestamp: datetime
    
    # Analysis results
    raw_success: bool  # Did command exit successfully?
    parsed_results: Dict[str, Any]
    validation_evidence: List[TestValidationEvidence]
    confidence_assessment: Dict[str, Any]
    
    # Intention validation
    original_requirements: List[str]
    user_intentions: List[str]
    requirement_coverage: Dict[str, float]
    intention_alignment: float


class TestExecutionValidator:
    """
    Validates test execution with extreme skepticism about "passing" tests
    
    CORE PRINCIPLE: A test that runs without errors is NOT the same as a test that validates requirements.
    """
    
    def __init__(self):
        self.execution_history: List[TestExecutionResult] = []
        self.validation_patterns = self._initialize_validation_patterns()
        
    def _initialize_validation_patterns(self) -> Dict[str, Dict]:
        """Initialize patterns for detecting test validation quality"""
        return {
            'pytest': {
                'success_patterns': [
                    r'(\d+) passed',
                    r'(\d+) passed.*in ([\d.]+)s',
                    r'=+ (\d+) passed'
                ],
                'failure_patterns': [
                    r'(\d+) failed',
                    r'(\d+) error',
                    r'FAILED.*::'
                ],
                'assertion_patterns': [
                    r'assert\s+\w+',
                    r'AssertionError',
                    r'assert.*==.*'
                ],
                'coverage_patterns': [
                    r'coverage.*(\d+)%',
                    r'(\d+)%.*coverage'
                ]
            },
            'unittest': {
                'success_patterns': [
                    r'Ran (\d+) tests.*OK',
                    r'(\d+) tests.*OK'
                ],
                'failure_patterns': [
                    r'FAILED.*failures=(\d+)',
                    r'(\d+) failures',
                    r'ERROR.*errors=(\d+)'
                ]
            },
            'jest': {
                'success_patterns': [
                    r'(\d+) passed',
                    r'Tests:.*(\d+) passed'
                ],
                'failure_patterns': [
                    r'(\d+) failed',
                    r'Tests:.*(\d+) failed'
                ]
            },
            'generic': {
                'success_indicators': [
                    'all tests passed',
                    'success',
                    'ok',
                    '✓'
                ],
                'failure_indicators': [
                    'failed',
                    'error',
                    'exception',
                    '✗',
                    'assertion'
                ]
            }
        }
    
    def execute_and_validate_test(self,
                                test_command: str,
                                original_requirements: List[str],
                                user_intentions: List[str],
                                working_directory: Optional[Path] = None,
                                timeout: int = 300) -> TestExecutionResult:
        """
        Execute test with comprehensive validation and confidence assessment
        
        NEVER assumes success just because exit_code == 0
        """
        logger.info(f"🧪 Executing test with anti-overconfidence validation: {test_command}")
        
        start_time = datetime.now()
        
        # Execute the test command
        try:
            result = subprocess.run(
                test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_directory
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create initial result
            test_result = TestExecutionResult(
                test_command=test_command,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                timestamp=start_time,
                raw_success=result.returncode == 0,
                parsed_results={},
                validation_evidence=[],
                confidence_assessment={},
                original_requirements=original_requirements,
                user_intentions=user_intentions,
                requirement_coverage={},
                intention_alignment=0.0
            )
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Test execution timed out after {timeout}s")
            return self._create_failed_result(test_command, "timeout", original_requirements, user_intentions)
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            return self._create_failed_result(test_command, str(e), original_requirements, user_intentions)
        
        # Parse test output with skepticism
        test_result.parsed_results = self._parse_test_output(test_result)
        
        # Gather validation evidence (this is where the real work happens)
        test_result.validation_evidence = self._gather_validation_evidence(test_result)
        
        # Assess confidence with anti-overconfidence principles
        test_result.confidence_assessment = self._assess_test_confidence(test_result)
        
        # Validate requirement coverage
        test_result.requirement_coverage = self._validate_requirement_coverage(test_result)
        
        # Assess user intention alignment
        test_result.intention_alignment = self._assess_intention_alignment(test_result)
        
        # Store in history
        self.execution_history.append(test_result)
        
        # Log skeptical assessment
        confidence = test_result.confidence_assessment.get('final_confidence', 0.0)
        logger.info(f"🤔 Test confidence assessment: {confidence:.1%}")
        
        if confidence < 0.60:
            logger.warning(f"⚠️ LOW CONFIDENCE: Test may not have validated intended behavior")
        
        return test_result
    
    def _create_failed_result(self, command: str, error: str, requirements: List[str], intentions: List[str]) -> TestExecutionResult:
        """Create result for failed test execution"""
        return TestExecutionResult(
            test_command=command,
            exit_code=-1,
            stdout="",
            stderr=error,
            execution_time=0.0,
            timestamp=datetime.now(),
            raw_success=False,
            parsed_results={'error': error},
            validation_evidence=[],
            confidence_assessment={'final_confidence': 0.0, 'level': TestConfidenceLevel.EXECUTION_FAILED},
            original_requirements=requirements,
            user_intentions=intentions,
            requirement_coverage={},
            intention_alignment=0.0
        )
    
    def _parse_test_output(self, test_result: TestExecutionResult) -> Dict[str, Any]:
        """Parse test output with skeptical analysis"""
        output = test_result.stdout + test_result.stderr
        parsed = {
            'raw_exit_success': test_result.exit_code == 0,
            'output_length': len(output),
            'has_output': len(output.strip()) > 0,
            'detected_framework': None,
            'test_counts': {},
            'assertion_indicators': [],
            'error_indicators': [],
            'success_indicators': []
        }
        
        # Detect test framework
        if 'pytest' in test_result.test_command.lower() or '::' in output:
            parsed['detected_framework'] = 'pytest'
        elif 'unittest' in output or 'python -m unittest' in test_result.test_command:
            parsed['detected_framework'] = 'unittest'
        elif 'jest' in test_result.test_command.lower() or 'Jest' in output:
            parsed['detected_framework'] = 'jest'
        else:
            parsed['detected_framework'] = 'generic'
        
        # Parse test counts and indicators
        framework = parsed['detected_framework']
        if framework in self.validation_patterns:
            patterns = self.validation_patterns[framework]
            
            # Look for success patterns
            for pattern in patterns.get('success_patterns', []):
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    parsed['success_indicators'].extend(matches)
            
            # Look for failure patterns
            for pattern in patterns.get('failure_patterns', []):
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    parsed['error_indicators'].extend(matches)
            
            # Look for assertion patterns
            for pattern in patterns.get('assertion_patterns', []):
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    parsed['assertion_indicators'].extend(matches)
        
        # Generic indicators
        generic_patterns = self.validation_patterns.get('generic', {})
        for indicator in generic_patterns.get('success_indicators', []):
            if indicator.lower() in output.lower():
                parsed['success_indicators'].append(indicator)
        
        for indicator in generic_patterns.get('failure_indicators', []):
            if indicator.lower() in output.lower():
                parsed['error_indicators'].append(indicator)
        
        return parsed
    
    def _gather_validation_evidence(self, test_result: TestExecutionResult) -> List[TestValidationEvidence]:
        """Gather evidence for test validation confidence (the critical part)"""
        evidence = []
        
        # Evidence 1: Output Analysis
        if test_result.parsed_results.get('has_output'):
            confidence = 0.3 if test_result.parsed_results.get('raw_exit_success') else 0.1
            evidence.append(TestValidationEvidence(
                validation_type=TestValidationType.OUTPUT_ANALYSIS,
                description=f"Test produced output ({test_result.parsed_results['output_length']} chars)",
                confidence_weight=confidence,
                verification_method="output_parsing",
                evidence_details=test_result.parsed_results,
                timestamp=datetime.now()
            ))
        
        # Evidence 2: Error Absence (weak evidence)
        if test_result.exit_code == 0:
            evidence.append(TestValidationEvidence(
                validation_type=TestValidationType.ERROR_ABSENCE_CONFIRMATION,
                description="Command exited with code 0",
                confidence_weight=0.2,  # Very weak evidence
                verification_method="exit_code_check",
                evidence_details={'exit_code': test_result.exit_code},
                timestamp=datetime.now()
            ))
        
        # Evidence 3: Assertion Detection (moderate evidence)
        assertions = test_result.parsed_results.get('assertion_indicators', [])
        if assertions:
            evidence.append(TestValidationEvidence(
                validation_type=TestValidationType.ASSERTION_VERIFICATION,
                description=f"Detected {len(assertions)} assertion indicators",
                confidence_weight=0.6,
                verification_method="pattern_matching",
                evidence_details={'assertions_found': assertions},
                timestamp=datetime.now()
            ))
        
        # Evidence 4: Success Pattern Detection (moderate evidence)
        success_indicators = test_result.parsed_results.get('success_indicators', [])
        if success_indicators:
            evidence.append(TestValidationEvidence(
                validation_type=TestValidationType.BEHAVIOR_CONFIRMATION,
                description=f"Found {len(success_indicators)} success indicators",
                confidence_weight=0.5,
                verification_method="pattern_matching",
                evidence_details={'success_patterns': success_indicators},
                timestamp=datetime.now()
            ))
        
        # Evidence 5: Framework-Specific Validation
        framework = test_result.parsed_results.get('detected_framework')
        if framework and framework != 'generic':
            evidence.append(TestValidationEvidence(
                validation_type=TestValidationType.OUTPUT_ANALYSIS,
                description=f"Recognized {framework} test framework output",
                confidence_weight=0.4,
                verification_method="framework_detection",
                evidence_details={'framework': framework},
                timestamp=datetime.now()
            ))
        
        return evidence
    
    def _assess_test_confidence(self, test_result: TestExecutionResult) -> Dict[str, Any]:
        """Assess test confidence with anti-overconfidence principles"""
        
        # Start with deep skepticism about test success
        base_confidence = 0.05  # Very skeptical about "passed" tests
        
        # Analyze evidence with diminishing returns
        evidence_contribution = 0.0
        if test_result.validation_evidence:
            # Sort evidence by strength
            sorted_evidence = sorted(test_result.validation_evidence, 
                                   key=lambda e: e.strength_score, reverse=True)
            
            # Apply diminishing returns
            confidence_increments = [0.20, 0.15, 0.10, 0.08, 0.05]
            for i, evidence in enumerate(sorted_evidence[:5]):
                if i < len(confidence_increments):
                    increment = confidence_increments[i] * evidence.strength_score
                    evidence_contribution += increment
                    logger.info(f"📊 Evidence {i+1}: +{increment:.1%} from {evidence.validation_type.value}")
        
        # Apply skepticism penalties
        skepticism_penalties = 0.0
        
        # Penalty for no clear assertions
        if not any(e.validation_type == TestValidationType.ASSERTION_VERIFICATION 
                  for e in test_result.validation_evidence):
            skepticism_penalties += 0.15
            logger.info("📉 Penalty: No clear assertions detected")
        
        # Penalty for generic/unclear output
        if test_result.parsed_results.get('detected_framework') == 'generic':
            skepticism_penalties += 0.10
            logger.info("📉 Penalty: Generic/unclear test framework")
        
        # Penalty for short execution time (might not have done much)
        if test_result.execution_time < 0.1:
            skepticism_penalties += 0.08
            logger.info("📉 Penalty: Very short execution time")
        
        # Penalty for no specific success patterns
        if not test_result.parsed_results.get('success_indicators'):
            skepticism_penalties += 0.12
            logger.info("📉 Penalty: No specific success patterns detected")
        
        # Calculate raw confidence
        raw_confidence = base_confidence + evidence_contribution - skepticism_penalties
        
        # Apply natural ceiling for test confidence
        if raw_confidence > 0.70:
            # Tests need exceptional evidence to be highly confident
            exceptional_criteria = [
                len(test_result.validation_evidence) >= 4,
                any(e.validation_type == TestValidationType.ASSERTION_VERIFICATION for e in test_result.validation_evidence),
                any(e.validation_type == TestValidationType.BEHAVIOR_CONFIRMATION for e in test_result.validation_evidence),
                test_result.execution_time > 0.5,  # Took some time to run
                len(test_result.parsed_results.get('success_indicators', [])) > 0
            ]
            
            exceptional_count = sum(exceptional_criteria)
            if exceptional_count < 4:
                excess = raw_confidence - 0.70
                raw_confidence = 0.70 + (excess * 0.2)
                logger.info(f"🤔 Reduced test confidence to {raw_confidence:.1%} - insufficient evidence for high certainty")
        
        # Final confidence with hard cap
        final_confidence = min(raw_confidence, 0.90)
        
        # Determine confidence level
        if final_confidence >= 0.85:
            level = TestConfidenceLevel.VERIFIED_PASSED
        elif final_confidence >= 0.75:
            level = TestConfidenceLevel.CONFIDENTLY_PASSED
        elif final_confidence >= 0.60:
            level = TestConfidenceLevel.PROBABLY_PASSED
        elif final_confidence >= 0.40:
            level = TestConfidenceLevel.LIKELY_PASSED
        elif test_result.raw_success:
            level = TestConfidenceLevel.EXECUTION_UNCLEAR
        else:
            level = TestConfidenceLevel.EXECUTION_FAILED
        
        return {
            'final_confidence': final_confidence,
            'level': level,
            'base_skepticism': base_confidence,
            'evidence_contribution': evidence_contribution,
            'skepticism_penalties': skepticism_penalties,
            'exceptional_criteria_met': exceptional_count if raw_confidence > 0.70 else 0,
            'reasoning': f"Started skeptical ({base_confidence:.1%}), added evidence ({evidence_contribution:.1%}), applied penalties (-{skepticism_penalties:.1%})"
        }
    
    def _validate_requirement_coverage(self, test_result: TestExecutionResult) -> Dict[str, float]:
        """Validate how well the test covers original requirements"""
        coverage = {}
        
        # This is a simplified implementation - in practice, this would need
        # sophisticated analysis of test code vs requirements
        for requirement in test_result.original_requirements:
            # Check if requirement keywords appear in test command or output
            requirement_words = requirement.lower().split()
            test_content = (test_result.test_command + " " + test_result.stdout).lower()
            
            matches = sum(1 for word in requirement_words if word in test_content)
            coverage_score = min(matches / len(requirement_words), 1.0) if requirement_words else 0.0
            
            coverage[requirement] = coverage_score
        
        return coverage
    
    def _assess_intention_alignment(self, test_result: TestExecutionResult) -> float:
        """Assess how well the test aligns with user intentions"""
        if not test_result.user_intentions:
            return 0.5  # Neutral if no intentions specified
        
        # Simplified implementation - check for intention keywords in test
        total_alignment = 0.0
        for intention in test_result.user_intentions:
            intention_words = intention.lower().split()
            test_content = (test_result.test_command + " " + test_result.stdout).lower()
            
            matches = sum(1 for word in intention_words if word in test_content)
            alignment = min(matches / len(intention_words), 1.0) if intention_words else 0.0
            total_alignment += alignment
        
        return total_alignment / len(test_result.user_intentions)
    
    def generate_test_validation_report(self, test_result: TestExecutionResult) -> str:
        """Generate comprehensive test validation report"""
        confidence = test_result.confidence_assessment.get('final_confidence', 0.0)
        level = test_result.confidence_assessment.get('level', TestConfidenceLevel.EXECUTION_UNCLEAR)
        
        report = f"""
🧪 TEST EXECUTION VALIDATION REPORT

Command: {test_result.test_command}
Exit Code: {test_result.exit_code}
Execution Time: {test_result.execution_time:.2f}s
Raw Success: {'✅' if test_result.raw_success else '❌'}

🤔 CONFIDENCE ASSESSMENT: {confidence:.1%}
Level: {level.value.replace('_', ' ').title()}

WHAT I'M REASONABLY CONFIDENT ABOUT:
"""
        
        if confidence > 0.60:
            report += f"- Test command executed without fatal errors\n"
            report += f"- Found {len(test_result.validation_evidence)} pieces of validation evidence\n"
            if test_result.parsed_results.get('success_indicators'):
                report += f"- Detected success indicators: {test_result.parsed_results['success_indicators']}\n"
        else:
            report += "- Very limited confidence in test validation\n"
        
        # Add confidence calculation breakdown
        assessment = test_result.confidence_assessment
        report += f"""
📊 TEST CONFIDENCE CALCULATION:
- Started with deep skepticism: {assessment.get('base_skepticism', 0):.1%}
- Evidence contribution: +{assessment.get('evidence_contribution', 0):.1%}
- Skepticism penalties: -{assessment.get('skepticism_penalties', 0):.1%}
- Final confidence: {confidence:.1%}
- Reasoning: {assessment.get('reasoning', 'N/A')}
"""
        
        report += """
WHAT I'M UNCERTAIN ABOUT:
"""
        
        uncertainties = []
        if confidence < 0.75:
            uncertainties.append("Whether test actually validated intended behavior")
        if not test_result.validation_evidence:
            uncertainties.append("No clear validation evidence found")
        if test_result.intention_alignment < 0.7:
            uncertainties.append("Test may not align with user intentions")
        if not test_result.parsed_results.get('assertion_indicators'):
            uncertainties.append("No clear assertions detected in test")
        
        for uncertainty in uncertainties:
            report += f"- {uncertainty}\n"
        
        # Requirement coverage
        if test_result.requirement_coverage:
            report += f"\nREQUIREMENT COVERAGE:\n"
            for req, coverage in test_result.requirement_coverage.items():
                status = "✅" if coverage > 0.7 else "⚠️" if coverage > 0.3 else "❌"
                report += f"{status} {req}: {coverage:.1%} coverage\n"
        
        # User intention alignment
        report += f"\nUSER INTENTION ALIGNMENT: {test_result.intention_alignment:.1%}\n"
        
        # Recommendations
        report += f"\nRECOMMENDATIONS:\n"
        if confidence < 0.60:
            report += "❌ DO NOT assume test passed - insufficient validation evidence\n"
            report += "🔍 Investigate test output more thoroughly\n"
            report += "🧪 Consider adding more specific assertions\n"
        elif confidence < 0.75:
            report += "⚠️ Test likely passed but needs more validation\n"
            report += "📋 Verify test actually checks intended behavior\n"
        else:
            report += "✅ Test appears to have passed with good evidence\n"
        
        report += f"""
🤔 MANDATORY TEST UNCERTAINTY REMINDERS:
- Test execution ≠ Test validation
- No errors ≠ Requirements verified  
- Exit code 0 ≠ User intentions met
- Always verify tests actually check what they claim to check
- Consider what the test might have missed or bypassed

⚠️ TEST CONFIDENCE HARD CAP: Maximum 90% - Perfect test validation is extremely rare
"""
        
        return report
    
    def should_trust_test_result(self, test_result: TestExecutionResult) -> Tuple[bool, str]:
        """Determine if test result should be trusted"""
        confidence = test_result.confidence_assessment.get('final_confidence', 0.0)
        
        if confidence >= 0.75:
            return True, f"High confidence ({confidence:.1%}) with strong validation evidence"
        elif confidence >= 0.60:
            return False, f"Moderate confidence ({confidence:.1%}) - need more validation"
        else:
            return False, f"Low confidence ({confidence:.1%}) - test results unclear"


# Integration functions
def execute_test_with_confidence_validation(test_command: str,
                                          requirements: List[str],
                                          user_intentions: List[str],
                                          working_directory: Optional[Path] = None) -> Dict[str, Any]:
    """
    Execute test with comprehensive confidence validation
    
    Args:
        test_command: The test command to execute
        requirements: List of original requirements the test should verify
        user_intentions: List of user intentions behind the test
        working_directory: Directory to run test in
        
    Returns:
        Comprehensive test validation result
    """
    validator = TestExecutionValidator()
    result = validator.execute_and_validate_test(
        test_command=test_command,
        original_requirements=requirements,
        user_intentions=user_intentions,
        working_directory=working_directory
    )
    
    # Convert to dictionary for easier integration
    return {
        'test_command': result.test_command,
        'raw_success': result.raw_success,
        'confidence': result.confidence_assessment.get('final_confidence', 0.0),
        'confidence_level': result.confidence_assessment.get('level', TestConfidenceLevel.EXECUTION_UNCLEAR).value,
        'should_trust': validator.should_trust_test_result(result),
        'requirement_coverage': result.requirement_coverage,
        'intention_alignment': result.intention_alignment,
        'validation_report': validator.generate_test_validation_report(result),
        'evidence_count': len(result.validation_evidence),
        'execution_time': result.execution_time,
        'parsed_results': result.parsed_results
    }


if __name__ == "__main__":
    # Example usage
    validator = TestExecutionValidator()
    
    print("🧪 TESTING TEST EXECUTION VALIDATOR")
    print("="*60)
    
    # Test with a simple command that will "pass" but may not validate much
    requirements = [
        "Function should return correct calculation",
        "Function should handle edge cases",
        "Function should validate input parameters"
    ]
    
    user_intentions = [
        "Ensure the calculator works correctly",
        "Prevent bugs in production",
        "Validate all mathematical operations"
    ]
    
    # Simulate a test that exits successfully but may not validate much
    print("\n1. Testing simple command (may have low confidence):")
    result = execute_test_with_confidence_validation(
        test_command="echo 'All tests passed'",
        requirements=requirements,
        user_intentions=user_intentions
    )
    
    print(f"Raw Success: {result['raw_success']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Should Trust: {result['should_trust']}")
    print(f"Evidence Count: {result['evidence_count']}")
    
    # Show the full report
    print("\n2. Full Validation Report:")
    print(result['validation_report'])
