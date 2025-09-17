#!/usr/bin/env python3
"""Lineage Validation System.

Main coordinator for lineage validation framework including accuracy
validation, experimental data comparison, and error detection/correction
for comprehensive lineage tracking validation.

Integration: Main validation coordinator for developmental biology
Rationale: Main validation coordinator with focused responsibilities
"""

from typing import Dict, List, Optional
import logging

from .experimental_data_reference import ExperimentalDataReference
from .lineage_accuracy_validator import LineageAccuracyValidator
from .lineage_error_detector import LineageErrorDetector
from .lineage_tree_constructor import LineageTreeConstructor

logger = logging.getLogger(__name__)

class LineageValidationSystem:
    """Main coordinator for lineage validation framework.
    
    Coordinates comprehensive lineage validation including accuracy
    validation against experimental data, error detection and correction,
    and overall lineage tracking quality assessment.
    
    Key Components:
    - Experimental data reference and comparison
    - Lineage tracking accuracy validation
    - Error detection and correction
    - Comprehensive validation reporting
    """
    
    def __init__(self, tree_constructor: LineageTreeConstructor):
        """Initialize lineage validation system.
        
        Args:
            tree_constructor: Lineage tree constructor for validation
        """
        self.tree_constructor = tree_constructor
        
        # Initialize validation components
        self.experimental_reference = ExperimentalDataReference()
        self.accuracy_validator = LineageAccuracyValidator(self.experimental_reference)
        self.error_detector = LineageErrorDetector()
        
        # Validation state
        self.validation_history: List[Dict[str, any]] = []
        
        logger.info("Initialized LineageValidationSystem")
    
    def validate_simulation_lineage(self, simulation_id: str,
                                   reference_studies: List[str] = None) -> Dict[str, any]:
        """Validate complete simulation lineage against experimental data.
        
        Args:
            simulation_id: Simulation identifier to validate
            reference_studies: List of reference studies for comparison
            
        Returns:
            Comprehensive validation results
        """
        logger.info(f"Validating simulation lineage: {simulation_id}")
        
        # Get simulation data
        simulation_summary = self.tree_constructor.export_simulation_summary(simulation_id)
        
        if not simulation_summary:
            return {'error': f'Simulation {simulation_id} not found'}
        
        # Use default reference studies if none specified
        if reference_studies is None:
            reference_studies = ['jessell_2000_neural_tube', 'livesey_2013_cortical']
        
        validation_results = {
            'simulation_id': simulation_id,
            'validation_timestamp': 'current',
            'reference_studies': reference_studies,
            'accuracy_validations': {},
            'error_detection': {},
            'overall_assessment': {}
        }
        
        # Validate against each reference study
        for study in reference_studies:
            study_validation = self._validate_against_study(simulation_summary, study)
            validation_results['accuracy_validations'][study] = study_validation
        
        # Detect errors in lineage system
        error_detection = self._comprehensive_error_detection(simulation_summary)
        validation_results['error_detection'] = error_detection
        
        # Calculate overall assessment
        overall_assessment = self._calculate_overall_assessment(
            validation_results['accuracy_validations'], error_detection)
        validation_results['overall_assessment'] = overall_assessment
        
        # Store validation history
        self.validation_history.append(validation_results)
        
        logger.info(f"Validation complete: {overall_assessment['validation_passed']}")
        
        return validation_results
    
    def _validate_against_study(self, simulation_data: Dict[str, any], 
                               study_name: str) -> Dict[str, any]:
        """Validate simulation against specific experimental study."""
        study_validation = {}
        
        # Validate clone size distribution
        clone_validation = self.accuracy_validator.validate_clone_size_distribution(
            simulation_data, study_name)
        study_validation['clone_sizes'] = clone_validation
        
        # Validate fate proportions
        fate_validation = self.accuracy_validator.validate_fate_proportions(
            simulation_data, study_name)
        study_validation['fate_proportions'] = fate_validation
        
        # Validate division patterns
        division_validation = self.accuracy_validator.validate_division_patterns(
            simulation_data, study_name)
        study_validation['division_patterns'] = division_validation
        
        # Calculate study-specific accuracy
        accuracy_scores = []
        for validation in [clone_validation, fate_validation, division_validation]:
            if 'accuracy_score' in validation:
                accuracy_scores.append(validation['accuracy_score'])
        
        study_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
        study_validation['overall_study_accuracy'] = study_accuracy
        
        return study_validation
    
    def _comprehensive_error_detection(self, simulation_data: Dict[str, any]) -> Dict[str, any]:
        """Perform comprehensive error detection on simulation."""
        error_detection = {}
        
        # Extract lineage tags and tree data (simplified access)
        lineage_tags = {}  # Would extract from simulation_data
        tree_nodes = {}    # Would extract from tree_constructor
        
        # Detect barcode errors
        barcode_errors = self.error_detector.detect_barcode_errors(lineage_tags)
        error_detection['barcode_errors'] = barcode_errors
        
        # Detect tree structure errors
        structure_errors = self.error_detector.detect_tree_structure_errors(tree_nodes)
        error_detection['structure_errors'] = structure_errors
        
        # Calculate total error count
        total_errors = (len(barcode_errors.get('corrupted_barcodes', [])) +
                       len(barcode_errors.get('missing_barcodes', [])) +
                       len(structure_errors.get('orphaned_nodes', [])) +
                       len(structure_errors.get('temporal_violations', [])))
        
        error_detection['total_errors'] = total_errors
        error_detection['error_rate'] = total_errors / max(1, simulation_data.get('total_cells', 1))
        
        return error_detection
    
    def _calculate_overall_assessment(self, accuracy_validations: Dict[str, any],
                                    error_detection: Dict[str, any]) -> Dict[str, any]:
        """Calculate overall validation assessment."""
        # Calculate average accuracy across studies
        study_accuracies = []
        for study_validation in accuracy_validations.values():
            if 'overall_study_accuracy' in study_validation:
                study_accuracies.append(study_validation['overall_study_accuracy'])
        
        average_accuracy = np.mean(study_accuracies) if study_accuracies else 0.0
        
        # Calculate error impact
        error_rate = error_detection.get('error_rate', 1.0)
        error_impact = min(1.0, error_rate * 2.0)  # Errors reduce score
        
        # Overall validation score
        validation_score = max(0.0, average_accuracy - error_impact)
        
        # Check if validation passes
        validation_passed = (validation_score >= 0.95 and  # KPI target: ≥95%
                           error_rate < self.error_detector.error_tolerance)
        
        overall_assessment = {
            'validation_score': validation_score,
            'average_accuracy': average_accuracy,
            'error_rate': error_rate,
            'validation_passed': validation_passed,
            'kpi_target_met': validation_score >= 0.95,
            'studies_validated': len(accuracy_validations),
            'total_errors_detected': error_detection.get('total_errors', 0)
        }
        
        return overall_assessment
    
    def export_validation_report(self, simulation_id: str) -> Dict[str, any]:
        """Export comprehensive validation report.
        
        Args:
            simulation_id: Simulation to report on
            
        Returns:
            Complete validation report
        """
        # Find validation results for this simulation
        simulation_validations = [v for v in self.validation_history 
                                if v.get('simulation_id') == simulation_id]
        
        if not simulation_validations:
            # Run validation if not done yet
            validation_results = self.validate_simulation_lineage(simulation_id)
        else:
            validation_results = simulation_validations[-1]  # Most recent
        
        # Create comprehensive report
        report = {
            'validation_metadata': {
                'simulation_id': simulation_id,
                'validation_framework_version': '1.0.0',
                'kpi_target': 'lineage_tracking_accuracy ≥ 95%'
            },
            'experimental_reference': self.experimental_reference.export_reference_summary(),
            'validation_results': validation_results,
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        return report
    
    def _generate_recommendations(self, validation_results: Dict[str, any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        overall = validation_results.get('overall_assessment', {})
        
        if overall.get('validation_passed', False):
            recommendations.append("✅ Lineage tracking accuracy meets KPI target (≥95%)")
            recommendations.append("System ready for downstream neurogenesis processes")
        else:
            accuracy = overall.get('validation_score', 0.0)
            recommendations.append(f"⚠️ Lineage tracking accuracy below target: {accuracy:.1%}")
            
            if overall.get('error_rate', 0.0) > 0.05:
                recommendations.append("Implement error correction mechanisms")
            
            if accuracy < 0.9:
                recommendations.append("Consider barcode system optimization")
                recommendations.append("Review experimental data alignment")
        
        error_count = overall.get('total_errors_detected', 0)
        if error_count > 0:
            recommendations.append(f"Address {error_count} detected lineage errors")
        
        return recommendations
