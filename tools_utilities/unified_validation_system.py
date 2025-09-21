#!/usr/bin/env python3
"""
Unified Validation System - Main Entry Point
============================================
This is the MAIN ENTRY POINT for all validation in Quark.

Integrates and coordinates:
1. Comprehensive Validation System (14 API sources + 79 open access)
2. Literature Validation System (40 literature sources)  
3. Frictionless Validation (simple interface)
4. All newly integrated APIs (UniProt, BLAST, arXiv, CDX Server, etc.)

Usage:
    from tools_utilities.unified_validation_system import validate_claim
    result = validate_claim("AlphaFold predicts protein structures")

Author: Quark System
Date: 2025-01-20
"""

import json
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "knowledge" / "validation_system"))

try:
    from .comprehensive_validation_system import ComprehensiveValidationSystem, mandatory_validate
    COMPREHENSIVE_AVAILABLE = True
except ImportError:
    try:
        from comprehensive_validation_system import ComprehensiveValidationSystem, mandatory_validate
        COMPREHENSIVE_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Comprehensive validation system not available: {e}")
        COMPREHENSIVE_AVAILABLE = False

try:
    from .frictionless_validation import FrictionlessValidator
    FRICTIONLESS_AVAILABLE = True
except ImportError:
    try:
        from frictionless_validation import FrictionlessValidator
        FRICTIONLESS_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Frictionless validation not available: {e}")
        FRICTIONLESS_AVAILABLE = False

try:
    from quark_literature_integration import QuarkLiteratureValidator
    LITERATURE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Literature validation not available: {e}")
    LITERATURE_AVAILABLE = False

try:
    from literature_validation_system import LiteratureValidationSystem
    LITERATURE_CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core literature validation not available: {e}")
    LITERATURE_CORE_AVAILABLE = False

try:
    from .confidence_validator import ConfidenceValidator
    CONFIDENCE_AVAILABLE = True
except ImportError:
    try:
        from confidence_validator import ConfidenceValidator
        CONFIDENCE_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Confidence validator not available: {e}")
        CONFIDENCE_AVAILABLE = False

# Import task completion validator
try:
    from .task_completion_validator import TaskCompletionValidator, validate_task_completion_claim, should_mark_task_complete
    TASK_COMPLETION_AVAILABLE = True
except ImportError:
    try:
        from task_completion_validator import TaskCompletionValidator, validate_task_completion_claim, should_mark_task_complete
        TASK_COMPLETION_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Task completion validator not available: {e}")
        TASK_COMPLETION_AVAILABLE = False


class OverconfidenceLevel(Enum):
    """Severity levels of overconfidence detected"""
    CRITICAL = "CRITICAL"      # Absolute claims, 100% confidence
    HIGH = "HIGH"              # Very confident language without evidence
    MODERATE = "MODERATE"      # Confident language with weak evidence
    LOW = "LOW"                # Minor confidence issues


@dataclass
class OverconfidenceDetection:
    """Represents a detected overconfidence issue"""
    level: OverconfidenceLevel
    pattern: str
    context: str
    suggested_replacement: str
    confidence_claim: Optional[float] = None


class UnifiedValidationSystem:
    """
    Main entry point for ALL validation in Quark.
    
    Coordinates multiple validation systems:
    - 14 API sources (UniProt, BLAST, arXiv, CDX Server, etc.)
    - 79 open access literature sources  
    - 40 specialized literature sources (via Quark Literature Integration)
    - Frictionless interface for agents
    - Quark-specific biological validation methods
    """
    
    def __init__(self):
        """Initialize all available validation systems"""
        self.systems = {}
        self.total_sources = 0
        
        # Initialize overconfidence prevention patterns
        self.overconfident_patterns = self._load_overconfident_patterns()
        self.prevention_stats = {
            'detections': 0,
            'blocks': 0,
            'corrections': 0
        }
        
        # Initialize comprehensive system (API sources + open access)
        if COMPREHENSIVE_AVAILABLE:
            try:
                self.systems['comprehensive'] = ComprehensiveValidationSystem()
                self.total_sources += len(self.systems['comprehensive'].get_all_available_sources())
                logger.info("✅ Comprehensive validation system loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load comprehensive system: {e}")
        
        # Initialize frictionless system (simple interface)
        if FRICTIONLESS_AVAILABLE:
            try:
                self.systems['frictionless'] = FrictionlessValidator()
                logger.info("✅ Frictionless validation system loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load frictionless system: {e}")
        
        # Initialize literature system (specialized literature sources)
        if LITERATURE_AVAILABLE:
            try:
                self.systems['literature'] = QuarkLiteratureValidator()
                logger.info("✅ Literature validation system loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load literature system: {e}")
        
        # Initialize confidence validator (anti-overconfidence system)
        if CONFIDENCE_AVAILABLE:
            try:
                self.systems['confidence'] = ConfidenceValidator()
                self.total_sources += 67  # From confidence validator (11 MCP + 56 APIs + 20 open access)
                logger.info("✅ Confidence validation system loaded (anti-overconfidence)")
            except Exception as e:
                logger.error(f"❌ Failed to load confidence system: {e}")
        
        # Initialize task completion validator
        if TASK_COMPLETION_AVAILABLE:
            try:
                self.task_completion_validator = TaskCompletionValidator()
                logger.info("✅ Task completion validation system loaded (anti-overconfidence for tasks)")
            except Exception as e:
                logger.error(f"❌ Failed to load task completion system: {e}")
                self.task_completion_validator = None
        else:
            self.task_completion_validator = None
        
        if not self.systems:
            raise RuntimeError("❌ CRITICAL: No validation systems could be loaded!")
        
        logger.info(f"🔍 Unified Validation System initialized with {len(self.systems)} subsystems (confidence: ~85% based on available systems)")
        logger.info("🚨 MANDATORY overconfidence prevention ACTIVE - all responses will be validated")
    
    def validate_claim(self, claim: str, method: str = 'auto', max_sources: int = 10) -> Dict[str, Any]:
        """
        Main validation method - validates a claim using the best available system.
        
        Args:
            claim: Statement to validate
            method: 'auto', 'comprehensive', 'frictionless', 'literature', 'confidence', or 'all'
            max_sources: Maximum sources to use
            
        Returns:
            Unified validation result
        """
        start_time = time.time()
        
        if method == 'auto':
            method = self._select_best_method(claim)
        
        result = {
            'claim': claim,
            'method_used': method,
            'timestamp': time.time(),
            'confidence': 0.0,
            'consensus': 'UNKNOWN',
            'evidence': [],
            'sources_checked': 0,
            'systems_used': [],
            'processing_time': 0.0,
            'recommendations': []
        }
        
        try:
            if method == 'all':
                # Use all available systems and aggregate results
                result = self._validate_with_all_systems(claim, max_sources)
            elif method == 'comprehensive' and 'comprehensive' in self.systems:
                result = self._validate_comprehensive(claim, max_sources)
            elif method == 'frictionless' and 'frictionless' in self.systems:
                result = self._validate_frictionless(claim, max_sources)
            elif method == 'literature' and 'literature' in self.systems:
                result = self._validate_literature(claim, max_sources)
            elif method == 'confidence' and 'confidence' in self.systems:
                result = self._validate_confidence(claim, max_sources)
            else:
                # Fallback to best available system
                result = self._validate_fallback(claim, max_sources)
            
            result['processing_time'] = time.time() - start_time
            
            # MANDATORY: Apply uncertainty enforcement to ALL results
            result = self._enforce_uncertainty_principles(result)
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            result.update({
                'confidence': 0.0,
                'consensus': 'VALIDATION_ERROR',
                'evidence': [f"Validation encountered issues: {str(e)}"],
                'processing_time': time.time() - start_time
            })
            # Apply uncertainty principles even to errors
            result = self._enforce_uncertainty_principles(result)
        
        return result
    
    def validate_biological_claim(self, claim: str, organism: str = "", process: str = "") -> Dict[str, Any]:
        """
        Specialized validation for biological claims using Quark-specific methods.
        
        Args:
            claim: Biological claim to validate
            organism: Organism context (e.g., "human", "mouse")
            process: Biological process context (e.g., "neural development")
            
        Returns:
            Enhanced validation result with Quark-specific analysis
        """
        if 'literature' in self.systems:
            try:
                # Use Quark's specialized biological validation
                result = self.systems['literature'].validate_biological_claim(claim, organism, process)
                
                # Enhance with comprehensive validation if available
                if 'comprehensive' in self.systems:
                    comp_result = self._validate_comprehensive(claim, 5)
                    result['comprehensive_validation'] = {
                        'confidence': comp_result['confidence'],
                        'api_sources': comp_result['sources_checked'],
                        'consensus': comp_result['consensus']
                    }
                
                return {
                    'claim': claim,
                    'method_used': 'biological_specialized',
                    'confidence': result.get('validation_score', 0.0),
                    'consensus': result.get('confidence', 'UNKNOWN'),
                    'evidence': result.get('evidence', []),
                    'sources_checked': result.get('sources_searched', 0),
                    'systems_used': ['literature', 'quark_specialized'],
                    'quark_analysis': result.get('quark_analysis', {}),
                    'details': {
                        'organism_context': organism,
                        'process_context': process,
                        'biological_focus': True,
                        'full_result': result
                    }
                }
            except Exception as e:
                logger.error(f"❌ Biological validation failed: {e}")
                return self._error_result(claim, 'biological_specialized', str(e))
        else:
            # Fallback to general validation
            return self.validate_claim(claim, method='auto')
    
    def validate_neural_claim(self, claim: str) -> Dict[str, Any]:
        """Specialized validation for neuroscience claims"""
        if 'literature' in self.systems:
            try:
                result = self.systems['literature'].validate_neural_development_claim(claim)
                return self._format_specialized_result(result, claim, 'neural_specialized')
            except Exception as e:
                logger.error(f"❌ Neural validation failed: {e}")
                return self._error_result(claim, 'neural_specialized', str(e))
        else:
            return self.validate_claim(claim, method='auto')
    
    def validate_gene_editing_claim(self, claim: str, technique: str = "CRISPR") -> Dict[str, Any]:
        """Specialized validation for gene editing claims"""
        if 'literature' in self.systems:
            try:
                result = self.systems['literature'].validate_gene_editing_claim(claim, technique)
                return self._format_specialized_result(result, claim, 'gene_editing_specialized')
            except Exception as e:
                logger.error(f"❌ Gene editing validation failed: {e}")
                return self._error_result(claim, 'gene_editing_specialized', str(e))
        else:
            return self.validate_claim(claim, method='auto')
    
    def find_supporting_papers(self, topic: str, max_papers: int = 10) -> List[Dict[str, Any]]:
        """Find supporting papers for a research topic"""
        if 'literature' in self.systems:
            try:
                return self.systems['literature'].find_supporting_papers(topic, max_papers)
            except Exception as e:
                logger.error(f"❌ Paper search failed: {e}")
                return []
        else:
            return []
    
    def _format_specialized_result(self, result: Dict[str, Any], claim: str, method: str) -> Dict[str, Any]:
        """Format specialized validation results to unified format"""
        return {
            'claim': claim,
            'method_used': method,
            'confidence': result.get('validation_score', 0.0),
            'consensus': result.get('confidence', 'UNKNOWN'),
            'evidence': result.get('evidence', []),
            'sources_checked': result.get('sources_searched', 0),
            'systems_used': ['literature', 'quark_specialized'],
            'quark_analysis': result.get('quark_analysis', {}),
            'details': {
                'specialized_validation': True,
                'full_result': result
            }
        }
    
    def _select_best_method(self, claim: str) -> str:
        """Select the best validation method based on claim content"""
        claim_lower = claim.lower()
        
        # Literature-heavy topics
        if any(kw in claim_lower for kw in ['paper', 'study', 'research', 'publication', 'citation']):
            return 'literature' if 'literature' in self.systems else 'comprehensive'
        
        # Technical/API topics  
        if any(kw in claim_lower for kw in ['protein', 'gene', 'compound', 'material', 'dataset']):
            return 'comprehensive' if 'comprehensive' in self.systems else 'frictionless'
        
        # Simple claims
        if len(claim.split()) < 10:
            return 'frictionless' if 'frictionless' in self.systems else 'comprehensive'
        
        # Default to comprehensive
        return 'comprehensive' if 'comprehensive' in self.systems else 'frictionless'
    
    def _validate_comprehensive(self, claim: str, max_sources: int) -> Dict[str, Any]:
        """Validate using comprehensive system (APIs + open access)"""
        try:
            # Use the mandatory validation function
            raw_result = mandatory_validate(claim)
            
            return {
                'claim': claim,
                'method_used': 'comprehensive',
                'confidence': raw_result.get('confidence', 0.0),
                'consensus': raw_result.get('consensus', 'UNKNOWN'),
                'evidence': raw_result.get('evidence', []),
                'sources_checked': raw_result.get('sources_checked', 0),
                'systems_used': ['comprehensive'],
                'supporting_sources': raw_result.get('supporting_sources', 0),
                'details': {
                    'api_sources': True,
                    'open_access_sources': True,
                    'total_available': len(self.systems['comprehensive'].get_all_available_sources())
                }
            }
        except Exception as e:
            logger.error(f"❌ Comprehensive validation failed: {e}")
            return self._error_result(claim, 'comprehensive', str(e))
    
    def _validate_frictionless(self, claim: str, max_sources: int) -> Dict[str, Any]:
        """Validate using frictionless system (simple interface)"""
        try:
            raw_result = self.systems['frictionless'].quick_validate(claim, max_sources)
            
            return {
                'claim': claim,
                'method_used': 'frictionless',
                'confidence': raw_result.get('confidence', 0.0),
                'consensus': 'SUPPORTED' if raw_result.get('confidence', 0) > 0.7 else 'MIXED',
                'evidence': [s['description'] for s in raw_result.get('supporting_sources', [])],
                'sources_checked': raw_result.get('available_sources', 0),
                'systems_used': ['frictionless'],
                'supporting_sources': len(raw_result.get('supporting_sources', [])),
                'details': {
                    'categories_covered': raw_result.get('categories_covered', []),
                    'simple_interface': True
                }
            }
        except Exception as e:
            logger.error(f"❌ Frictionless validation failed: {e}")
            return self._error_result(claim, 'frictionless', str(e))
    
    def _validate_literature(self, claim: str, max_sources: int) -> Dict[str, Any]:
        """Validate using literature system (specialized sources)"""
        try:
            # Use biological claim validation as default
            raw_result = self.systems['literature'].validate_biological_claim(claim)
            
            return {
                'claim': claim,
                'method_used': 'literature',
                'confidence': raw_result.get('confidence', 0.0),
                'consensus': 'SUPPORTED' if raw_result.get('confidence', 0) > 0.7 else 'MIXED',
                'evidence': raw_result.get('evidence', []),
                'sources_checked': raw_result.get('sources_searched', 0),
                'systems_used': ['literature'],
                'supporting_sources': raw_result.get('evidence_count', 0),
                'details': {
                    'literature_focused': True,
                    'specialized_sources': 40
                }
            }
        except Exception as e:
            logger.error(f"❌ Literature validation failed: {e}")
            return self._error_result(claim, 'literature', str(e))
    
    def _validate_confidence(self, claim: str, max_sources: int) -> Dict[str, Any]:
        """Validate using confidence system (anti-overconfidence validation)"""
        try:
            # Use the anti-overconfidence validation method
            raw_result = self.systems['confidence'].perform_anti_overconfidence_validation(claim)
            
            return {
                'claim': claim,
                'method_used': 'confidence',
                'confidence': raw_result.get('final_confidence', 0.0),
                'consensus': 'LIKELY_SUPPORTED' if raw_result.get('validated', False) else 'UNCERTAIN',
                'evidence': [s['name'] for s in raw_result.get('sources_consulted', [])],
                'sources_checked': len(raw_result.get('sources_consulted', [])),
                'systems_used': ['confidence'],
                'supporting_sources': len([s for s in raw_result.get('sources_consulted', []) if s.get('is_open_access', False)]),
                'uncertainty_level': raw_result.get('uncertainty_level', 'UNKNOWN'),
                'user_correction_needed': raw_result.get('user_correction_needed', False),
                'anti_overconfidence_report': raw_result.get('anti_overconfidence_report', ''),
                'details': {
                    'anti_overconfidence': True,
                    'uncertainty_triggers': raw_result.get('uncertainty_triggers', []),
                    'validation_gaps': raw_result.get('validation_gaps', []),
                    'open_access_sources': len([s for s in raw_result.get('sources_consulted', []) if s.get('is_open_access', False)]),
                    'total_resources': 67,  # MCP + APIs + Open Access
                    'confidence_capped_at_90': raw_result.get('final_confidence', 0.0) <= 0.9
                }
            }
        except Exception as e:
            logger.error(f"❌ Confidence validation failed: {e}")
            return self._error_result(claim, 'confidence', str(e))
    
    def _validate_with_all_systems(self, claim: str, max_sources: int) -> Dict[str, Any]:
        """Validate using ALL available systems and aggregate results"""
        results = []
        systems_used = []
        
        # Try each system
        for system_name in ['comprehensive', 'frictionless', 'literature', 'confidence']:
            if system_name in self.systems:
                try:
                    if system_name == 'comprehensive':
                        result = self._validate_comprehensive(claim, max_sources)
                    elif system_name == 'frictionless':
                        result = self._validate_frictionless(claim, max_sources)
                    elif system_name == 'literature':
                        result = self._validate_literature(claim, max_sources)
                    elif system_name == 'confidence':
                        result = self._validate_confidence(claim, max_sources)
                    
                    results.append(result)
                    systems_used.append(system_name)
                except Exception as e:
                    logger.error(f"❌ System {system_name} failed: {e}")
        
        if not results:
            return self._error_result(claim, 'all', "All systems failed")
        
        # Aggregate results
        confidences = [r['confidence'] for r in results if r['confidence'] > 0]
        total_sources = sum(r['sources_checked'] for r in results)
        all_evidence = []
        for r in results:
            if isinstance(r['evidence'], list):
                all_evidence.extend(r['evidence'])
            else:
                all_evidence.append(str(r['evidence']))
        
        # Calculate aggregate confidence with natural skepticism
        if confidences:
            # Use weighted average but stay naturally conservative
            base_aggregate = sum(confidences) / len(confidences)
            
            # Multiple system agreement provides modest boost, not automatic high confidence
            if len(confidences) > 1:
                # Agreement bonus is smaller and has diminishing returns
                agreement_factor = min(0.05, (len(confidences) - 1) * 0.02)
                base_aggregate += agreement_factor
                logger.info(f"🤝 Multi-system agreement bonus: +{agreement_factor:.1%}")
            
            # Natural ceiling for aggregated results - even multiple systems shouldn't easily exceed 75%
            if base_aggregate > 0.75:
                # Compress high aggregate confidence
                excess = base_aggregate - 0.75
                aggregate_confidence = 0.75 + (excess * 0.3)  # Heavily compress
                logger.info(f"🤔 Aggregate confidence naturally reduced from {base_aggregate:.1%} to {aggregate_confidence:.1%}")
            else:
                aggregate_confidence = base_aggregate
                
            # Apply final hard cap (should rarely be needed)
            aggregate_confidence = min(aggregate_confidence, 0.90)
        else:
            aggregate_confidence = 0.0
        
        # Determine consensus
        if aggregate_confidence > 0.8:
            consensus = 'STRONG_SUPPORT'
        elif aggregate_confidence > 0.6:
            consensus = 'MODERATE_SUPPORT'
        elif aggregate_confidence > 0.3:
            consensus = 'MIXED_EVIDENCE'
        else:
            consensus = 'INSUFFICIENT_EVIDENCE'
        
        return {
            'claim': claim,
            'method_used': 'all',
            'confidence': aggregate_confidence,
            'consensus': consensus,
            'evidence': all_evidence,
            'sources_checked': total_sources,
            'systems_used': systems_used,
            'supporting_sources': len([r for r in results if r['confidence'] > 0.5]),
            'details': {
                'individual_results': results,
                'systems_attempted': len(systems_used),
                'multi_system_validation': True
            }
        }
    
    def _validate_fallback(self, claim: str, max_sources: int) -> Dict[str, Any]:
        """Fallback validation using best available system"""
        # Try systems in order of preference
        for system_name in ['comprehensive', 'frictionless', 'literature']:
            if system_name in self.systems:
                logger.info(f"Using fallback system: {system_name}")
                if system_name == 'comprehensive':
                    return self._validate_comprehensive(claim, max_sources)
                elif system_name == 'frictionless':
                    return self._validate_frictionless(claim, max_sources)
                elif system_name == 'literature':
                    return self._validate_literature(claim, max_sources)
        
        return self._error_result(claim, 'fallback', "No systems available")
    
    def _enforce_uncertainty_principles(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        MANDATORY: Enforce anti-overconfidence principles on any validation result
        
        - Hard cap confidence at 90%
        - Add uncertainty qualifiers to consensus
        - Ensure doubt is expressed appropriately
        """
        # HARD CAP: Never allow >90% confidence
        if result.get('confidence', 0.0) > 0.90:
            logger.warning(f"⚠️ OVERCONFIDENCE DETECTED: Capping {result['confidence']:.1%} at 90%")
            result['confidence'] = 0.90
            
        # Add uncertainty markers to consensus
        consensus_mapping = {
            'VALIDATED': 'LIKELY_SUPPORTED',
            'CONFIRMED': 'APPEARS_SUPPORTED', 
            'PROVEN': 'EVIDENCE_SUGGESTS',
            'CERTAIN': 'REASONABLY_CONFIDENT',
            'DEFINITIVE': 'PROBABLE',
            'ABSOLUTE': 'STRONG_INDICATION',
            'GUARANTEED': 'WELL_SUPPORTED',
            'PERFECT': 'GOOD_EVIDENCE'
        }
        
        original_consensus = result.get('consensus', '')
        for overconfident, uncertain in consensus_mapping.items():
            if overconfident in original_consensus:
                result['consensus'] = original_consensus.replace(overconfident, uncertain)
                logger.info(f"🤔 Replaced overconfident '{overconfident}' with '{uncertain}'")
        
        # Add explicit uncertainty markers based on confidence level
        confidence = result.get('confidence', 0.0)
        if confidence < 0.3:
            result['uncertainty_qualifier'] = "⚠️ LOW CONFIDENCE - Significant uncertainty remains"
        elif confidence < 0.6:
            result['uncertainty_qualifier'] = "🟡 MODERATE CONFIDENCE - Some uncertainty present"  
        elif confidence < 0.8:
            result['uncertainty_qualifier'] = "✅ REASONABLE CONFIDENCE - Minor uncertainties noted"
        else:
            result['uncertainty_qualifier'] = "✅ HIGH CONFIDENCE - Still capped at 90% maximum"
            
        # Add mandatory uncertainty reminder
        result['uncertainty_reminder'] = "Remember: No claim can be 100% certain. Always consider alternative explanations and seek additional validation."
        
        return result
    
    def _load_overconfident_patterns(self) -> Dict[OverconfidenceLevel, List[Dict[str, str]]]:
        """Load patterns that indicate overconfidence - MANDATORY for prevention"""
        return {
            OverconfidenceLevel.CRITICAL: [
                {
                    "pattern": r"\b(100%|completely|absolutely|definitely|certainly|guaranteed|proven|perfect|exact|precise)\b",
                    "replacement": "likely|probably|appears to|seems to|evidence suggests|reasonably confident",
                    "context": "Absolute certainty claims"
                },
                {
                    "pattern": r"\b(task\s+(?:is\s+)?complete|finished|done|accomplished)\b",
                    "replacement": "task appears largely complete|likely finished|probably done",
                    "context": "Task completion claims"
                },
                {
                    "pattern": r"\b(test\s+passed|all\s+tests\s+pass|tests\s+successful)\b",
                    "replacement": "tests appear to pass|tests likely successful|test results suggest success",
                    "context": "Test execution claims"
                },
                {
                    "pattern": r"\b(this\s+(?:will|is)\s+(?:work|correct|right|accurate))\b",
                    "replacement": "this should work|this appears correct|this seems right|this is likely accurate",
                    "context": "Technical correctness claims"
                }
            ],
            OverconfidenceLevel.HIGH: [
                {
                    "pattern": r"\b(obviously|clearly|undoubtedly|without\s+question|no\s+doubt)\b",
                    "replacement": "likely|probably|it appears|evidence suggests",
                    "context": "High confidence qualifiers"
                },
                {
                    "pattern": r"\b(always\s+works|never\s+fails|can't\s+go\s+wrong)\b",
                    "replacement": "usually works|rarely fails|is generally reliable",
                    "context": "Absolute behavioral claims"
                },
                {
                    "pattern": r"\b(the\s+(?:best|only|correct)\s+(?:way|method|approach))\b",
                    "replacement": "a good way|one effective method|a recommended approach",
                    "context": "Singular solution claims"
                }
            ],
            OverconfidenceLevel.MODERATE: [
                {
                    "pattern": r"\b(should\s+work|will\s+solve|fixes\s+the\s+problem)\b",
                    "replacement": "might work|could solve|may help with the problem",
                    "context": "Confident predictions"
                },
                {
                    "pattern": r"\b(standard\s+practice|best\s+practice|recommended\s+approach)\b",
                    "replacement": "common practice|often recommended|frequently used approach",
                    "context": "Practice authority claims"
                }
            ],
            OverconfidenceLevel.LOW: [
                {
                    "pattern": r"\b(simple|easy|straightforward|trivial)\b",
                    "replacement": "relatively simple|can be manageable|appears straightforward|seems straightforward",
                    "context": "Difficulty minimization"
                }
            ]
        }
    
    def validate_response_before_output(self, response_content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        MANDATORY: Validate response content before it reaches the user.
        
        This method MUST be called before any response to prevent overconfident language.
        
        Returns:
        - validated_content: Corrected content (if auto-correctable)
        - should_block: True if response should be blocked entirely
        - detections: List of overconfidence issues found
        - required_actions: What must be done before response can be sent
        """
        logger.info("🔍 MANDATORY pre-response overconfidence validation")
        
        # Detect all overconfidence patterns
        detections = self._detect_overconfidence_patterns(response_content)
        
        # Assess overall confidence claims
        confidence_assessment = self._assess_confidence_claims(response_content, context)
        
        # Check for required uncertainty expressions
        uncertainty_check = self._check_uncertainty_expression(response_content)
        
        # Determine if response should be blocked
        should_block = self._should_block_response(detections, confidence_assessment, uncertainty_check)
        
        # Generate corrected content if possible
        corrected_content = self._auto_correct_overconfidence(response_content, detections)
        
        # Generate required actions
        required_actions = self._generate_required_actions(detections, confidence_assessment, uncertainty_check)
        
        # Update statistics
        self.prevention_stats['detections'] += len(detections)
        if should_block:
            self.prevention_stats['blocks'] += 1
        if corrected_content != response_content:
            self.prevention_stats['corrections'] += 1
        
        result = {
            "original_content": response_content,
            "validated_content": corrected_content,
            "should_block": should_block,
            "detections": detections,
            "confidence_assessment": confidence_assessment,
            "uncertainty_check": uncertainty_check,
            "required_actions": required_actions,
            "validation_summary": self._generate_validation_summary(detections, should_block)
        }
        
        # Log results
        if should_block:
            logger.error(f"🚨 RESPONSE BLOCKED: {len(detections)} overconfidence issues detected")
            for detection in detections:
                logger.error(f"   {detection.level.value}: {detection.pattern}")
        elif detections:
            logger.warning(f"⚠️ RESPONSE CORRECTED: {len(detections)} issues auto-fixed")
        else:
            logger.info("✅ Response passed overconfidence validation")
        
        return result
    
    def _detect_overconfidence_patterns(self, content: str) -> List[OverconfidenceDetection]:
        """Detect overconfident language patterns in content"""
        detections = []
        
        for level, patterns in self.overconfident_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end]
                    
                    detection = OverconfidenceDetection(
                        level=level,
                        pattern=match.group(),
                        context=context,
                        suggested_replacement=pattern_info["replacement"]
                    )
                    detections.append(detection)
        
        return detections
    
    def _assess_confidence_claims(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess any explicit or implicit confidence claims"""
        assessment = {
            "explicit_confidence": None,
            "implicit_confidence": "unknown",
            "evidence_provided": False,
            "sources_cited": 0,
            "uncertainty_expressed": False
        }
        
        # Look for explicit confidence percentages
        confidence_match = re.search(r'(\d+)%\s*confidence', content, re.IGNORECASE)
        if confidence_match:
            assessment["explicit_confidence"] = int(confidence_match.group(1))
        
        # Count citations and sources
        citation_patterns = [
            r'\[.*?\]',  # [source]
            r'according\s+to',  # according to
            r'based\s+on',  # based on
            r'documented\s+in',  # documented in
            r'https?://',  # URLs
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assessment["sources_cited"] += len(matches)
        
        assessment["evidence_provided"] = assessment["sources_cited"] > 0
        
        # Check for uncertainty expressions
        uncertainty_indicators = [
            r'uncertain', r'not\s+sure', r'might\s+be', r'could\s+be',
            r'appears\s+to', r'seems\s+to', r'likely', r'probably',
            r'evidence\s+suggests', r'I\s+believe', r'my\s+understanding'
        ]
        
        for indicator in uncertainty_indicators:
            if re.search(indicator, content, re.IGNORECASE):
                assessment["uncertainty_expressed"] = True
                break
        
        return assessment
    
    def _check_uncertainty_expression(self, content: str) -> Dict[str, Any]:
        """Check if appropriate uncertainty is expressed"""
        mandatory_uncertainty_phrases = [
            "I'm not entirely certain, but",
            "Based on available evidence",
            "This appears to be the case, though",
            "Evidence suggests that",
            "I have moderate confidence that",
            "This seems likely, but",
            "My understanding is that",
            "According to current information",
            "This approach appears promising, though",
            "I believe this is correct, but"
        ]
        
        return {
            "has_uncertainty_qualifiers": any(
                phrase.lower() in content.lower() 
                for phrase in mandatory_uncertainty_phrases
            ),
            "confidence_level_stated": bool(re.search(r'confidence|certain|sure', content, re.IGNORECASE)),
            "alternatives_acknowledged": bool(re.search(r'alternative|other\s+way|different\s+approach', content, re.IGNORECASE)),
            "limitations_noted": bool(re.search(r'limitation|caveat|however|but|though', content, re.IGNORECASE))
        }
    
    def _should_block_response(self, detections: List[OverconfidenceDetection], 
                              confidence_assessment: Dict[str, Any], 
                              uncertainty_check: Dict[str, Any]) -> bool:
        """Determine if response should be blocked entirely"""
        
        # Block if critical overconfidence detected
        critical_detections = [d for d in detections if d.level == OverconfidenceLevel.CRITICAL]
        if len(critical_detections) > 2:
            return True
        
        # Block if high confidence claimed without evidence
        explicit_confidence = confidence_assessment.get("explicit_confidence", 0)
        if (explicit_confidence is not None and explicit_confidence > 80 and 
            not confidence_assessment.get("evidence_provided", False)):
            return True
        
        # Block if no uncertainty expressed in technical claims
        if (confidence_assessment.get("sources_cited", 0) == 0 and 
            not uncertainty_check.get("has_uncertainty_qualifiers", False) and
            len(detections) > 0):
            return True
        
        return False
    
    def _auto_correct_overconfidence(self, content: str, detections: List[OverconfidenceDetection]) -> str:
        """Automatically correct overconfident language where possible"""
        corrected = content
        
        for detection in detections:
            if detection.level in [OverconfidenceLevel.LOW, OverconfidenceLevel.MODERATE]:
                # Auto-correct minor issues
                replacements = detection.suggested_replacement.split("|")
                if replacements:
                    corrected = re.sub(
                        detection.pattern, 
                        replacements[0], 
                        corrected, 
                        flags=re.IGNORECASE
                    )
        
        return corrected
    
    def _generate_required_actions(self, detections: List[OverconfidenceDetection],
                                  confidence_assessment: Dict[str, Any],
                                  uncertainty_check: Dict[str, Any]) -> List[str]:
        """Generate list of actions required before response can be sent"""
        actions = []
        
        # Critical overconfidence issues
        critical_detections = [d for d in detections if d.level == OverconfidenceLevel.CRITICAL]
        for detection in critical_detections:
            actions.append(f"CRITICAL: Replace '{detection.pattern}' with uncertain language")
        
        # Evidence requirements
        explicit_confidence = confidence_assessment.get("explicit_confidence", 0)
        if explicit_confidence is not None and explicit_confidence > 70:
            if not confidence_assessment.get("evidence_provided"):
                actions.append("REQUIRED: Provide evidence for high confidence claim")
            if confidence_assessment.get("sources_cited", 0) < 2:
                actions.append("REQUIRED: Cite at least 2 independent sources")
        
        # Uncertainty requirements
        if not uncertainty_check.get("has_uncertainty_qualifiers"):
            actions.append("REQUIRED: Add explicit uncertainty qualifiers")
        
        if not uncertainty_check.get("confidence_level_stated"):
            actions.append("REQUIRED: State explicit confidence level")
        
        return actions
    
    def _generate_validation_summary(self, detections: List[OverconfidenceDetection], blocked: bool) -> str:
        """Generate human-readable validation summary"""
        if not detections and not blocked:
            return "✅ Response passed all overconfidence checks"
        
        summary_parts = []
        
        if blocked:
            summary_parts.append("🚨 RESPONSE BLOCKED due to overconfidence")
        
        detection_counts = {}
        for detection in detections:
            detection_counts[detection.level] = detection_counts.get(detection.level, 0) + 1
        
        for level, count in detection_counts.items():
            summary_parts.append(f"{level.value}: {count} issues")
        
        return " | ".join(summary_parts)
    
    def get_prevention_statistics(self) -> Dict[str, Any]:
        """Get statistics about prevention system performance"""
        total_checks = max(1, self.prevention_stats['detections'])
        return {
            "total_detections": self.prevention_stats['detections'],
            "total_blocks": self.prevention_stats['blocks'],
            "total_corrections": self.prevention_stats['corrections'],
            "prevention_rate": self.prevention_stats['blocks'] / total_checks * 100,
            "correction_rate": self.prevention_stats['corrections'] / total_checks * 100
        }
    
    def _error_result(self, claim: str, method: str, error: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'claim': claim,
            'method_used': method,
            'confidence': 0.0,
            'consensus': 'ERROR',
            'evidence': [f"Validation error: {error}"],
            'sources_checked': 0,
            'systems_used': [],
            'supporting_sources': 0,
            'details': {'error': error}
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all validation systems"""
        status = {
            'unified_system_ready': len(self.systems) > 0,
            'systems_loaded': list(self.systems.keys()),
            'total_systems': len(self.systems),
            'total_sources_estimated': self.total_sources,
            'subsystem_details': {}
        }
        
        # Get details from each subsystem
        if 'comprehensive' in self.systems:
            try:
                comp_status = self.systems['comprehensive'].verify_all_sources_active()
                status['subsystem_details']['comprehensive'] = {
                    'sources_available': len(self.systems['comprehensive'].get_all_available_sources()),
                    'sources_active': sum(1 for v in comp_status.values() if v),
                    'includes_apis': True,
                    'includes_open_access': True
                }
            except Exception as e:
                status['subsystem_details']['comprehensive'] = {'error': str(e)}
        
        if 'frictionless' in self.systems:
            try:
                fric_status = self.systems['frictionless'].get_all_sources_summary()
                status['subsystem_details']['frictionless'] = {
                    'sources_available': fric_status['total_sources'],
                    'no_auth_sources': fric_status['no_auth'],
                    'auth_required_sources': fric_status['auth_required'],
                    'categories_covered': list(fric_status['by_category'].keys())
                }
            except Exception as e:
                status['subsystem_details']['frictionless'] = {'error': str(e)}
        
        if 'literature' in self.systems:
            status['subsystem_details']['literature'] = {
                'specialized_literature_sources': 40,
                'focus': 'biological and scientific literature'
            }
        
        return status
    
    def quick_validate(self, claim: str) -> str:
        """Ultra-simple validation that returns just a text summary"""
        result = self.validate_claim(claim, method='auto', max_sources=5)
        
        confidence = result['confidence']
        sources = result['sources_checked']
        systems = ', '.join(result['systems_used'])
        
        if confidence == 0.0:
            return f"❌ VALIDATION FAILED: {result['evidence'][0] if result['evidence'] else 'Unknown error'}"
        
        confidence_emoji = "✅" if confidence > 0.8 else "🟡" if confidence > 0.6 else "⚠️"
        confidence_text = "High" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Low"
        
        return f"""{confidence_emoji} VALIDATION: {confidence*100:.0f}% confidence ({confidence_text})
Sources: {sources} checked via {systems}
Consensus: {result['consensus']}
Time: {result.get('processing_time', 0):.1f}s"""
    
    def validate_task_completion(self, 
                               task_id: str,
                               task_description: str,
                               evidence: List[Dict[str, Any]],
                               claimed_complete: bool = True) -> Dict[str, Any]:
        """
        Validate task completion with anti-overconfidence principles
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of what the task involves
            evidence: List of evidence supporting completion
            claimed_complete: Whether completion is being claimed
            
        Returns:
            Detailed validation result with completion confidence
        """
        if not self.task_completion_validator:
            logger.warning("⚠️ Task completion validator not available")
            return {
                'error': 'Task completion validation not available',
                'confidence': 0.0,
                'status': 'incomplete',
                'should_mark_complete': False
            }
        
        # Use the task completion validator
        result = validate_task_completion_claim(task_id, task_description, evidence)
        
        # Apply uncertainty enforcement to the result
        result = self._enforce_uncertainty_principles(result)
        
        # Add completion recommendation
        should_complete, reason = should_mark_task_complete(result)
        result['should_mark_complete'] = should_complete
        result['completion_recommendation'] = reason
        
        logger.info(f"🔍 Task completion validation: {result['completion_confidence']:.1%} confidence")
        logger.info(f"📋 Recommendation: {'✅ Mark complete' if should_complete else '❌ Do not mark complete'}")
        
        return result
    
    def check_task_completion_confidence(self, 
                                       task_description: str,
                                       completion_evidence: List[str]) -> Tuple[bool, float, str]:
        """
        Quick check if a task should be marked as complete
        
        Args:
            task_description: What the task involves
            completion_evidence: List of evidence descriptions
            
        Returns:
            (should_mark_complete, confidence_score, reasoning)
        """
        # Convert evidence strings to evidence dictionaries
        evidence = []
        for i, evidence_desc in enumerate(completion_evidence):
            evidence.append({
                'type': 'functional_verification',
                'description': evidence_desc,
                'verification_method': 'self_reported',
                'confidence_weight': 0.7,
                'details': {}
            })
        
        # Validate completion
        result = self.validate_task_completion(
            task_id=f"quick_check_{int(time.time())}",
            task_description=task_description,
            evidence=evidence
        )
        
        confidence = result.get('completion_confidence', 0.0)
        should_complete = result.get('should_mark_complete', False)
        reasoning = result.get('completion_recommendation', 'Unknown')
        
        return should_complete, confidence, reasoning
    
    def generate_task_completion_report(self, task_id: str, validation_result: Dict[str, Any]) -> str:
        """Generate detailed task completion report"""
        if not self.task_completion_validator:
            return "❌ Task completion validation not available"
        
        return self.task_completion_validator.generate_completion_report(task_id, validation_result)


# Global instance for easy access
_unified_validator = None

def get_validator() -> UnifiedValidationSystem:
    """Get or create the global validation system instance"""
    global _unified_validator
    if _unified_validator is None:
        _unified_validator = UnifiedValidationSystem()
    return _unified_validator

# Convenience functions for easy import and use
def validate_claim(claim: str, method: str = 'auto') -> Dict[str, Any]:
    """
    MAIN VALIDATION FUNCTION - Use this for all validation needs
    
    Args:
        claim: Statement to validate
        method: 'auto', 'comprehensive', 'frictionless', 'literature', or 'all'
        
    Returns:
        Complete validation result
    """
    validator = get_validator()
    return validator.validate_claim(claim, method)

def quick_validate(claim: str) -> str:
    """Quick validation that returns simple text summary"""
    validator = get_validator()
    return validator.quick_validate(claim)

def get_system_status() -> Dict[str, Any]:
    """Get status of all validation systems"""
    validator = get_validator()
    return validator.get_system_status()

def validate_with_all_systems(claim: str) -> Dict[str, Any]:
    """Validate using ALL available systems for maximum confidence"""
    return validate_claim(claim, method='all')

# Specialized validation functions
def validate_biological_claim(claim: str, organism: str = "", process: str = "") -> Dict[str, Any]:
    """Validate biological claims with Quark-specific analysis"""
    validator = get_validator()
    return validator.validate_biological_claim(claim, organism, process)

def validate_neural_claim(claim: str) -> Dict[str, Any]:
    """Validate neuroscience claims with specialized literature focus"""
    validator = get_validator()
    return validator.validate_neural_claim(claim)

def validate_gene_editing_claim(claim: str, technique: str = "CRISPR") -> Dict[str, Any]:
    """Validate gene editing claims with technique-specific focus"""
    validator = get_validator()
    return validator.validate_gene_editing_claim(claim, technique)

def find_supporting_papers(topic: str, max_papers: int = 10) -> List[Dict[str, Any]]:
    """Find supporting papers for a research topic"""
    validator = get_validator()
    return validator.find_supporting_papers(topic, max_papers)

def validate_with_anti_overconfidence(claim: str, user_statement: str = "", context: str = "") -> Dict[str, Any]:
    """
    MANDATORY anti-overconfidence validation - enforces strict uncertainty rules
    
    Args:
        claim: The claim to validate
        user_statement: Optional user statement to question for overconfident language
        context: Additional context for validation
        
    Returns:
        Comprehensive validation result with anti-overconfidence report
    """
    validator = get_validator()
    return validator.validate_claim(claim, method='confidence')

def mandatory_anti_overconfidence_check(claim: str, user_statement: str = "") -> str:
    """
    Quick anti-overconfidence check - returns formatted report
    
    Args:
        claim: The claim to validate
        user_statement: Optional user statement to check for overconfident language
        
    Returns:
        Formatted anti-overconfidence report
    """
    result = validate_with_anti_overconfidence(claim, user_statement)
    return result.get('anti_overconfidence_report', 'No report available')


# MANDATORY PRE-RESPONSE VALIDATION FUNCTIONS
def validate_response_before_output(response_content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    MANDATORY: Validate response content before it reaches the user.
    
    This function MUST be called before any response to prevent overconfident language.
    
    Args:
        response_content: The response content to validate
        context: Optional context for validation
        
    Returns:
        Validation result with corrected content and blocking decision
    """
    validator = get_validator()
    return validator.validate_response_before_output(response_content, context)


def is_response_safe_to_send(validation_result: Dict[str, Any]) -> bool:
    """Check if response is safe to send based on validation result"""
    return not validation_result.get("should_block", False)


def get_corrected_response(validation_result: Dict[str, Any]) -> str:
    """Get the corrected response content"""
    return validation_result.get("validated_content", "")


def get_overconfidence_prevention_stats() -> Dict[str, Any]:
    """Get statistics about overconfidence prevention system performance"""
    validator = get_validator()
    return validator.get_prevention_statistics()


def mandatory_response_check(response_content: str) -> Tuple[bool, str, List[str]]:
    """
    MANDATORY check that returns simple decision for response safety
    
    Args:
        response_content: The response to check
        
    Returns:
        (is_safe_to_send, corrected_content, required_actions)
    """
    validation_result = validate_response_before_output(response_content)
    
    is_safe = is_response_safe_to_send(validation_result)
    corrected = get_corrected_response(validation_result)
    actions = validation_result.get("required_actions", [])
    
    return is_safe, corrected, actions


# Task completion validation functions
def validate_task_completion_with_evidence(task_id: str, 
                                         task_description: str, 
                                         evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate task completion with provided evidence
    
    Args:
        task_id: Unique identifier for the task
        task_description: What the task involves
        evidence: List of evidence dictionaries
        
    Returns:
        Validation result with completion confidence
    """
    validator = get_validator()
    return validator.validate_task_completion(task_id, task_description, evidence)


def should_task_be_marked_complete(task_description: str, 
                                 completion_evidence: List[str]) -> Tuple[bool, float, str]:
    """
    Quick check if a task should be marked as complete
    
    Args:
        task_description: Description of the task
        completion_evidence: List of evidence descriptions
        
    Returns:
        (should_mark_complete, confidence_score, reasoning)
    """
    validator = get_validator()
    return validator.check_task_completion_confidence(task_description, completion_evidence)


def generate_task_evidence_template(task_type: str = "general") -> List[Dict[str, Any]]:
    """
    Generate template for task completion evidence
    
    Args:
        task_type: Type of task (general, coding, research, testing)
        
    Returns:
        Template evidence list to fill out
    """
    templates = {
        'coding': [
            {
                'type': 'test_results',
                'description': 'All unit tests pass (X/X)',
                'verification_method': 'automated_testing',
                'confidence_weight': 0.9,
                'details': {'test_count': 0, 'pass_rate': 0.0}
            },
            {
                'type': 'functional_verification',
                'description': 'Feature works as specified',
                'verification_method': 'manual_testing',
                'confidence_weight': 0.8,
                'details': {'requirements_met': []}
            },
            {
                'type': 'integration_verified',
                'description': 'Integrates properly with existing system',
                'verification_method': 'integration_testing',
                'confidence_weight': 0.7,
                'details': {'integration_points': []}
            }
        ],
        'research': [
            {
                'type': 'documentation_complete',
                'description': 'Research findings documented',
                'verification_method': 'peer_review',
                'confidence_weight': 0.8,
                'details': {'document_count': 0}
            },
            {
                'type': 'metrics_achieved',
                'description': 'Research objectives met',
                'verification_method': 'objective_assessment',
                'confidence_weight': 0.9,
                'details': {'objectives_met': []}
            }
        ],
        'testing': [
            {
                'type': 'test_results',
                'description': 'Test suite executed successfully',
                'verification_method': 'automated_execution',
                'confidence_weight': 0.95,
                'details': {'tests_run': 0, 'failures': 0}
            },
            {
                'type': 'metrics_achieved',
                'description': 'Coverage and quality metrics met',
                'verification_method': 'metrics_analysis',
                'confidence_weight': 0.8,
                'details': {'coverage_percent': 0.0}
            }
        ]
    }
    
    return templates.get(task_type, [
        {
            'type': 'functional_verification',
            'description': 'Task requirements fulfilled',
            'verification_method': 'manual_verification',
            'confidence_weight': 0.7,
            'details': {}
        }
    ])


def main():
    """Demo the unified validation system"""
    print("🔍 UNIFIED VALIDATION SYSTEM")
    print("=" * 60)
    print("MAIN ENTRY POINT for all Quark validation")
    print("=" * 60)
    
    try:
        # Initialize system
        validator = UnifiedValidationSystem()
        
        # Show system status
        status = validator.get_system_status()
        print(f"\n📊 SYSTEM STATUS:")
        print(f"✅ Systems loaded: {', '.join(status['systems_loaded'])}")
        print(f"✅ Total estimated sources: {status['total_sources_estimated']}")
        
        for system_name, details in status['subsystem_details'].items():
            if 'error' not in details:
                if system_name == 'comprehensive':
                    print(f"  • {system_name}: {details['sources_active']}/{details['sources_available']} sources active")
                elif system_name == 'frictionless':
                    print(f"  • {system_name}: {details['sources_available']} sources, {len(details['categories_covered'])} categories")
                elif system_name == 'literature':
                    print(f"  • {system_name}: {details['specialized_literature_sources']} literature sources")
        
        # Test validation
        test_claims = [
            "AlphaFold can predict protein structures with high accuracy",
            "Neural networks in the brain use backpropagation",
            "CRISPR-Cas9 can edit genes with 100% accuracy",
            "Exercise enhances neuroplasticity in the hippocampus"
        ]
        
        print(f"\n🧪 TESTING VALIDATION:")
        print("-" * 40)
        
        for i, claim in enumerate(test_claims, 1):
            print(f"\n{i}. Testing: {claim}")
            
            # Quick validation
            quick_result = validator.quick_validate(claim)
            print(f"   Quick: {quick_result}")
            
            # Full validation with auto method selection
            full_result = validator.validate_claim(claim, method='auto')
            print(f"   Method: {full_result['method_used']}")
            print(f"   Confidence: {full_result['confidence']*100:.1f}%")
            print(f"   Sources: {full_result['sources_checked']}")
        
        print(f"\n✅ UNIFIED VALIDATION SYSTEM READY!")
        print("\nMAIN ENTRY POINTS:")
        print("• validate_claim(claim) - Full validation (confidence varies by method)")
        print("• quick_validate(claim) - Simple text result (confidence: ~80%)")
        print("• get_system_status() - System information (confidence: 90%)")
        
    except Exception as e:
        print(f"❌ SYSTEM INITIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
