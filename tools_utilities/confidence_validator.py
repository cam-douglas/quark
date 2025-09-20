#!/usr/bin/env python3
"""
Confidence Validation System for Cursor AI
Enforces anti-overconfidence rules and mandatory validation checkpoints
"""

import json
import logging
import re
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories with strict thresholds"""
    LOW = (0, 40, "⚠️ LOW CONFIDENCE")
    MEDIUM = (40, 70, "🟡 MEDIUM CONFIDENCE")
    HIGH = (70, 90, "✅ HIGH CONFIDENCE")
    FORBIDDEN = (90, 100, "🚫 OVERCONFIDENT (FORBIDDEN)")
    

class ValidationSource(Enum):
    """Authoritative source hierarchy"""
    PRIMARY = 1.0  # Official docs, peer-reviewed papers
    SECONDARY = 0.7  # Community best practices, high-vote answers
    EXPERIMENTAL = 0.3  # Untested code, personal interpretations
    

class ConfidenceValidator:
    """
    Enforces anti-overconfidence rules and validation requirements
    """
    
    def __init__(self, max_confidence: float = 0.90):
        """
        Initialize validator with maximum allowed confidence
        
        Args:
            max_confidence: Hard cap on confidence (default 90%)
        """
        self.max_confidence = max_confidence
        self.validation_history: List[Dict[str, Any]] = []
        self.current_confidence: float = 0.0
        self.validation_sources: List[Dict[str, Any]] = []
        self.uncertainty_areas: List[str] = []
        
    def calculate_confidence(
        self,
        source_authority: float,
        cross_validation: float,
        test_coverage: float,
        peer_review: float
    ) -> float:
        """
        Calculate confidence score with hard cap at 90%
        
        Args:
            source_authority: Authority level of primary source (0-1)
            cross_validation: Cross-validation score (0-1)
            test_coverage: Test coverage percentage (0-1)
            peer_review: Peer review validation (0-1)
            
        Returns:
            Confidence score capped at max_confidence
        """
        raw_confidence = (
            source_authority * 0.3 +
            cross_validation * 0.3 +
            test_coverage * 0.2 +
            peer_review * 0.2
        )
        
        # Apply hard cap
        confidence = min(raw_confidence, self.max_confidence)
        
        # Store for tracking
        self.current_confidence = confidence
        
        # Log calculation
        logger.info(f"Confidence calculated: {confidence:.1%} (raw: {raw_confidence:.1%})")
        
        return confidence
        
    def get_confidence_level(self, score: float) -> Tuple[ConfidenceLevel, str]:
        """
        Get confidence level category and prefix
        
        Args:
            score: Confidence score (0-1)
            
        Returns:
            Tuple of (ConfidenceLevel, prefix string)
        """
        score_percent = score * 100
        
        for level in ConfidenceLevel:
            min_val, max_val, prefix = level.value
            if min_val <= score_percent < max_val:
                return level, f"{prefix} ({score_percent:.0f}%)"
                
        # Should never reach here, but handle edge case
        return ConfidenceLevel.LOW, f"⚠️ LOW CONFIDENCE ({score_percent:.0f}%)"
        
    def validate_sources(self, sources: List[Dict[str, Any]]) -> float:
        """
        Validate and score information sources
        
        Args:
            sources: List of source dictionaries with 'type' and 'authority' keys
            
        Returns:
            Average authority score
        """
        if not sources:
            logger.warning("No validation sources provided")
            return 0.0
            
        total_authority = 0.0
        valid_sources = 0
        
        for source in sources:
            source_type = source.get('type', 'experimental')
            
            # Map source type to validation level
            if source_type in ['official_docs', 'peer_reviewed', 'api_spec']:
                authority = ValidationSource.PRIMARY.value
            elif source_type in ['community', 'stackoverflow', 'blog']:
                authority = ValidationSource.SECONDARY.value
            else:
                authority = ValidationSource.EXPERIMENTAL.value
                
            # Adjust for recency if provided
            if 'date' in source:
                try:
                    source_date = datetime.fromisoformat(source['date'])
                    age_days = (datetime.now() - source_date).days
                    if age_days > 730:  # Older than 2 years
                        authority *= 0.7
                except:
                    pass
                    
            total_authority += authority
            valid_sources += 1
            
            # Store for reporting
            self.validation_sources.append({
                'source': source.get('name', 'Unknown'),
                'type': source_type,
                'authority': authority,
                'url': source.get('url', '')
            })
            
        return total_authority / valid_sources if valid_sources > 0 else 0.0
        
    def check_uncertainty_triggers(self, context: Dict[str, Any]) -> List[str]:
        """
        Check for conditions that should trigger uncertainty expression
        
        Args:
            context: Context dictionary with various flags
            
        Returns:
            List of triggered uncertainty conditions
        """
        triggers = []
        
        if not context.get('documentation_found', True):
            triggers.append("No direct documentation found")
            
        if context.get('conflicting_sources', False):
            triggers.append("Conflicting information between sources")
            
        if context.get('complex_biology', False):
            triggers.append("Complex biological/scientific concepts")
            
        if context.get('performance_unclear', False):
            triggers.append("Performance implications unclear")
            
        if context.get('security_implications', False):
            triggers.append("Security implications present")
            
        if context.get('multiple_approaches', False):
            triggers.append("Multiple valid approaches exist")
            
        if context.get('ambiguous_requirements', False):
            triggers.append("Requirements are ambiguous")
            
        self.uncertainty_areas = triggers
        return triggers
        
    def generate_validation_report(self) -> str:
        """
        Generate formatted validation report with confidence assessment
        
        Returns:
            Formatted validation report string
        """
        level, prefix = self.get_confidence_level(self.current_confidence)
        
        report = f"""
{prefix}

📊 VALIDATION METRICS:
- Confidence Score: {self.current_confidence:.1%}
- Sources Consulted: {len(self.validation_sources)}
- Uncertainty Areas: {len(self.uncertainty_areas)}

🔍 VALIDATION SOURCES:
"""
        
        for i, source in enumerate(self.validation_sources, 1):
            report += f"{i}. [{source['source']}] "
            report += f"(Authority: {source['authority']:.1f}) "
            if source['url']:
                report += f"- {source['url']}\n"
            else:
                report += "\n"
                
        if self.uncertainty_areas:
            report += f"""
⚠️ UNCERTAINTY TRIGGERS:
"""
            for area in self.uncertainty_areas:
                report += f"- {area}\n"
                
        # Add recommendations based on confidence level
        if level == ConfidenceLevel.LOW:
            report += """
🔔 RECOMMENDED ACTIONS:
- Seek additional authoritative sources
- Request user verification
- Consider alternative approaches
- Run comprehensive tests before proceeding
"""
        elif level == ConfidenceLevel.MEDIUM:
            report += """
🔔 RECOMMENDED ACTIONS:
- Cross-validate with additional sources
- Run edge case tests
- Document assumptions clearly
"""
        elif level == ConfidenceLevel.FORBIDDEN:
            report += """
🚫 WARNING: OVERCONFIDENCE DETECTED
- Reduce certainty in claims
- Add explicit uncertainty markers
- Seek contradictory evidence
- Never claim 100% certainty
"""
            
        return report
        
    def validate_response(self, response: str) -> Dict[str, Any]:
        """
        Validate that response includes proper confidence markers
        
        Args:
            response: Response text to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'has_confidence_statement': False,
            'has_uncertainty_markers': False,
            'has_source_citations': False,
            'confidence_level_stated': False,
            'validation_checklist': []
        }
        
        # Check for confidence statements
        confidence_patterns = [
            r'confidence.*\d+%',
            r'(?:low|medium|high)\s+confidence',
            r'(?:uncertain|moderately confident|reasonably confident)'
        ]
        
        for pattern in confidence_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                results['has_confidence_statement'] = True
                break
                
        # Check for uncertainty markers
        uncertainty_patterns = [
            r'(?:might|may|could|possibly|potentially)',
            r'(?:uncertain|unsure|unclear)',
            r'I (?:don\'t|do not) know',
            r'(?:assumption|guess|estimate)'
        ]
        
        for pattern in uncertainty_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                results['has_uncertainty_markers'] = True
                break
                
        # Check for source citations
        citation_patterns = [
            r'(?:source|reference|according to|based on)',
            r'(?:documentation|paper|study|article)',
            r'\[.*\]',  # Bracketed citations
            r'https?://'  # URLs
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                results['has_source_citations'] = True
                break
                
        # Check if confidence level is explicitly stated
        if re.search(r'\d+%|confidence level', response, re.IGNORECASE):
            results['confidence_level_stated'] = True
            
        # Validation checklist
        checklist = [
            ('Confidence explicitly stated', results['has_confidence_statement']),
            ('Uncertainty acknowledged', results['has_uncertainty_markers']),
            ('Sources cited', results['has_source_citations']),
            ('Confidence level quantified', results['confidence_level_stated'])
        ]
        
        results['validation_checklist'] = checklist
        results['passes_validation'] = all(v for _, v in checklist)
        
        return results
        
    def enforce_validation(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Enforce validation rules on content
        
        Args:
            content: Content to validate
            context: Optional context dictionary
            
        Returns:
            Tuple of (passes_validation, feedback_message)
        """
        # Validate response format
        validation = self.validate_response(content)
        
        if not validation['passes_validation']:
            missing = [item for item, passed in validation['validation_checklist'] if not passed]
            
            feedback = "⚠️ VALIDATION FAILED - Missing required elements:\n"
            for item in missing:
                feedback += f"  - {item}\n"
                
            feedback += "\n📋 Required elements:\n"
            feedback += "  1. Explicit confidence statement (X%)\n"
            feedback += "  2. Uncertainty acknowledgment where appropriate\n"
            feedback += "  3. Source citations for claims\n"
            feedback += "  4. Quantified confidence level\n"
            
            return False, feedback
            
        # Check for overconfidence
        if "100%" in content or re.search(r'(absolutely|definitely|certainly) (certain|sure|correct)', content, re.IGNORECASE):
            feedback = "🚫 OVERCONFIDENCE DETECTED\n"
            feedback += "Never claim 100% certainty or use absolute language.\n"
            feedback += "Maximum allowed confidence is 90%.\n"
            return False, feedback
            
        return True, "✅ Validation passed"


def main():
    """
    CLI interface for confidence validation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Confidence Validation System')
    parser.add_argument('--check', type=str, help='Check confidence in a response file')
    parser.add_argument('--validate', type=str, help='Validate response format')
    parser.add_argument('--report', action='store_true', help='Generate validation report')
    parser.add_argument('--sources', type=str, help='JSON file with validation sources')
    
    args = parser.parse_args()
    
    validator = ConfidenceValidator()
    
    if args.check:
        # Check confidence in a file
        try:
            with open(args.check, 'r') as f:
                content = f.read()
                
            passed, feedback = validator.enforce_validation(content)
            print(feedback)
            
            if not passed:
                sys.exit(1)
                
        except FileNotFoundError:
            print(f"Error: File '{args.check}' not found")
            sys.exit(1)
            
    elif args.validate:
        # Validate specific content
        passed, feedback = validator.enforce_validation(args.validate)
        print(feedback)
        
        if not passed:
            sys.exit(1)
            
    elif args.report:
        # Generate validation report
        if args.sources:
            try:
                with open(args.sources, 'r') as f:
                    sources = json.load(f)
                    
                authority = validator.validate_sources(sources.get('sources', []))
                
                # Calculate sample confidence
                validator.calculate_confidence(
                    source_authority=authority,
                    cross_validation=sources.get('cross_validation', 0.5),
                    test_coverage=sources.get('test_coverage', 0.3),
                    peer_review=sources.get('peer_review', 0.2)
                )
                
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading sources: {e}")
                sys.exit(1)
                
        print(validator.generate_validation_report())
        
    else:
        # Interactive mode - validate stdin
        print("Confidence Validator - Interactive Mode")
        print("Enter content to validate (Ctrl+D to finish):")
        
        try:
            content = sys.stdin.read()
            passed, feedback = validator.enforce_validation(content)
            print("\n" + feedback)
            
            if not passed:
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\nValidation cancelled")
            sys.exit(0)


if __name__ == "__main__":
    main()
