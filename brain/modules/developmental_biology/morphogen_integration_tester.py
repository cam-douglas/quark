"""
Morphogen Integration Tester

Tests integration between morphogen gradients and validation systems.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Any

from .integration_test_types import IntegrationTestResult, IntegrationTestStatus
from .proliferation_rate_validator import ProliferationRateValidator
from .neuroepithelial_cells import NeuroepithelialCell
from .neuroepithelial_cell_types import NeuroepithelialCellType


class MorphogenIntegrationTester:
    """Tests morphogen gradient integration with validation systems"""
    
    def __init__(self):
        """Initialize morphogen integration tester"""
        self.proliferation_validator = ProliferationRateValidator()
    
    def test_morphogen_proliferation_integration(self, 
                                               morphogen_concentrations: Dict[str, float],
                                               developmental_stage: str) -> IntegrationTestResult:
        """
        Test integration between morphogen gradients and proliferation validation
        
        Args:
            morphogen_concentrations: Morphogen levels (SHH, BMP, WNT, FGF)
            developmental_stage: E8.5, E9.5, E10.5, etc.
        
        Returns:
            Integration test result
        """
        try:
            # Create test cells with morphogen exposure
            test_cells = {}
            cell_count = 50
            
            for i in range(cell_count):
                cell_id = f'morphogen_cell_{i+1}'
                # Position cells along morphogen gradient
                position = (float(i) / cell_count, 0.0, 0.5)
                dev_time = float(developmental_stage.replace('E', ''))
                
                cell = NeuroepithelialCell(
                    cell_type=NeuroepithelialCellType.EARLY_MULTIPOTENT,
                    position=position,
                    developmental_time=dev_time
                )
                
                # Apply morphogen exposure effects on proliferation
                shh_effect = morphogen_concentrations.get('SHH', 0.5)
                # Higher SHH typically reduces proliferation in ventral regions
                proliferation_modifier = 1.0 - (shh_effect * 0.3)
                cell.proliferation_rate *= proliferation_modifier
                
                test_cells[cell_id] = cell
            
            # Run proliferation validation
            validation_results = self.proliferation_validator.validate_proliferation_rates(
                test_cells, developmental_stage
            )
            
            # Calculate integration metrics
            passed_validations = sum(1 for r in validation_results if r.validation_status.value == "passed")
            total_validations = len(validation_results)
            experimental_accuracy = passed_validations / total_validations if total_validations > 0 else 0.0
            
            # Integration score based on morphogen-proliferation consistency
            integration_score = self._calculate_morphogen_proliferation_consistency(
                morphogen_concentrations, validation_results
            )
            
            status = IntegrationTestStatus.PASSED if experimental_accuracy >= 0.7 else IntegrationTestStatus.FAILED
            
            return IntegrationTestResult(
                test_name="morphogen_proliferation_integration",
                foundation_component="morphogen_gradients",
                validation_component="proliferation_rate_validator",
                test_status=status,
                experimental_accuracy=experimental_accuracy,
                integration_score=integration_score
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="morphogen_proliferation_integration",
                foundation_component="morphogen_gradients",
                validation_component="proliferation_rate_validator",
                test_status=IntegrationTestStatus.FAILED,
                experimental_accuracy=0.0,
                integration_score=0.0,
                error_message=str(e)
            )
    
    def _calculate_morphogen_proliferation_consistency(self,
                                                     morphogen_concentrations: Dict[str, float],
                                                     validation_results: List[Any]) -> float:
        """Calculate consistency between morphogen levels and proliferation validation"""
        # Expected relationships from literature:
        # - High SHH reduces proliferation in ventral regions
        # - FGF promotes proliferation
        # - BMP can inhibit proliferation
        
        shh_level = morphogen_concentrations.get('SHH', 0.5)
        fgf_level = morphogen_concentrations.get('FGF', 0.5)
        
        # Calculate expected proliferation modifier
        expected_modifier = (1.0 - shh_level * 0.3) + (fgf_level * 0.2)
        
        # Compare with actual validation results
        proliferation_metrics = [r for r in validation_results if 'proliferation' in r.metric_type.value]
        
        if not proliferation_metrics:
            return 0.5  # Neutral score if no proliferation metrics
        
        # Calculate consistency score
        consistency_scores = []
        for metric in proliferation_metrics:
            expected_value = metric.experimental_value * expected_modifier
            actual_value = metric.simulated_value
            consistency = 1.0 - abs(expected_value - actual_value) / max(expected_value, actual_value)
            consistency_scores.append(max(0.0, consistency))
        
        return np.mean(consistency_scores)
