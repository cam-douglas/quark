"""
Spatial Integration Tester

Tests integration between ventricular topology and spatial validation systems.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Any

from .integration_test_types import IntegrationTestResult, IntegrationTestStatus
from .spatial_organization_validator import SpatialOrganizationValidator
from .neuroepithelial_cells import NeuroepithelialCell
from .neuroepithelial_cell_types import NeuroepithelialCellType


class SpatialIntegrationTester:
    """Tests spatial organization integration with validation systems"""
    
    def __init__(self):
        """Initialize spatial integration tester"""
        self.spatial_validator = SpatialOrganizationValidator()
    
    def test_ventricular_spatial_integration(self,
                                           ventricular_topology: Dict[str, Any],
                                           developmental_stage: str) -> IntegrationTestResult:
        """
        Test integration between ventricular topology and spatial validation
        
        Args:
            ventricular_topology: VZ organization parameters
            developmental_stage: E8.5, E9.5, E10.5, etc.
        
        Returns:
            Integration test result
        """
        try:
            # Create test cells organized according to ventricular topology
            test_cells = {}
            vz_thickness = ventricular_topology.get('vz_thickness', 40.0)
            cell_density = ventricular_topology.get('cell_density', 200000.0)
            
            # Calculate number of cells based on density and volume
            estimated_volume = vz_thickness * 100 * 100  # μm³
            estimated_cells = int((cell_density / 1e9) * estimated_volume)  # Convert to actual count
            
            for i in range(min(estimated_cells, 100)):  # Cap for performance
                cell_id = f'vz_cell_{i+1}'
                # Position cells within VZ boundaries
                apical_distance = (float(i) / estimated_cells) * vz_thickness
                position = (0.5, 0.5, apical_distance / 100.0)  # Normalize to [0,1]
                
                dev_time = float(developmental_stage.replace('E', ''))
                
                cell = NeuroepithelialCell(
                    cell_type=NeuroepithelialCellType.EARLY_MULTIPOTENT,
                    position=position,
                    developmental_time=dev_time
                )
                
                test_cells[cell_id] = cell
            
            # Run spatial validation
            validation_results = self.spatial_validator.validate_spatial_organization(
                test_cells, developmental_stage
            )
            
            # Calculate integration metrics
            passed_validations = sum(1 for r in validation_results if r.validation_passed)
            total_validations = len(validation_results)
            experimental_accuracy = passed_validations / total_validations if total_validations > 0 else 0.0
            
            # Integration score based on topology-spatial consistency
            integration_score = self._calculate_ventricular_spatial_consistency(
                ventricular_topology, validation_results
            )
            
            status = IntegrationTestStatus.PASSED if experimental_accuracy >= 0.8 else IntegrationTestStatus.FAILED
            
            return IntegrationTestResult(
                test_name="ventricular_spatial_integration",
                foundation_component="ventricular_topology",
                validation_component="spatial_organization_validator",
                test_status=status,
                experimental_accuracy=experimental_accuracy,
                integration_score=integration_score
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="ventricular_spatial_integration",
                foundation_component="ventricular_topology",
                validation_component="spatial_organization_validator",
                test_status=IntegrationTestStatus.FAILED,
                experimental_accuracy=0.0,
                integration_score=0.0,
                error_message=str(e)
            )
    
    def test_meninges_spatial_integration(self,
                                        meninges_scaffold: Dict[str, Any],
                                        developmental_stage: str) -> IntegrationTestResult:
        """Test integration between meninges scaffold and spatial validation"""
        try:
            # Meninges provide boundary constraints for spatial organization
            boundary_integrity = meninges_scaffold.get('boundary_integrity', 0.95)
            
            # Create test cells within meninges boundaries
            test_cells = {}
            for i in range(30):
                cell_id = f'meninges_cell_{i+1}'
                # Position cells within meninges-defined boundaries
                position = (float(i) * 0.03, 0.0, 0.5)
                dev_time = float(developmental_stage.replace('E', ''))
                
                cell = NeuroepithelialCell(
                    cell_type=NeuroepithelialCellType.EARLY_MULTIPOTENT,
                    position=position,
                    developmental_time=dev_time
                )
                test_cells[cell_id] = cell
            
            # Run spatial validation
            validation_results = self.spatial_validator.validate_spatial_organization(
                test_cells, developmental_stage
            )
            
            # Calculate metrics
            passed_validations = sum(1 for r in validation_results if r.validation_passed)
            total_validations = len(validation_results)
            experimental_accuracy = passed_validations / total_validations if total_validations > 0 else 0.0
            
            # Integration score based on boundary integrity
            integration_score = boundary_integrity * experimental_accuracy
            
            status = IntegrationTestStatus.PASSED if integration_score >= 0.8 else IntegrationTestStatus.FAILED
            
            return IntegrationTestResult(
                test_name="meninges_spatial_integration",
                foundation_component="meninges_scaffold",
                validation_component="spatial_organization_validator",
                test_status=status,
                experimental_accuracy=experimental_accuracy,
                integration_score=integration_score
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="meninges_spatial_integration",
                foundation_component="meninges_scaffold",
                validation_component="spatial_organization_validator",
                test_status=IntegrationTestStatus.FAILED,
                experimental_accuracy=0.0,
                integration_score=0.0,
                error_message=str(e)
            )
    
    def _calculate_ventricular_spatial_consistency(self,
                                                 ventricular_topology: Dict[str, Any],
                                                 validation_results: List[Any]) -> float:
        """Calculate consistency between ventricular topology and spatial validation"""
        # Expected relationships from literature:
        # - VZ thickness affects cell density
        # - Apical-basal organization affects cell distribution
        
        vz_thickness = ventricular_topology.get('vz_thickness', 40.0)
        cell_density = ventricular_topology.get('cell_density', 200000.0)
        
        # Calculate expected spatial metrics
        expected_density_score = min(1.0, cell_density / 250000.0)  # Normalize to max expected
        expected_thickness_score = min(1.0, vz_thickness / 60.0)   # Normalize to max expected
        
        # Compare with actual validation results
        spatial_metrics = [r for r in validation_results if hasattr(r, 'region')]
        
        if not spatial_metrics:
            return 0.5  # Neutral score if no spatial metrics
        
        # Calculate consistency score
        consistency_scores = []
        for metric in spatial_metrics:
            if 'thickness' in metric.metric_type.value:
                expected_value = metric.experimental_value * expected_thickness_score
                actual_value = metric.simulated_value
                consistency = 1.0 - abs(expected_value - actual_value) / max(expected_value, actual_value, 1.0)
                consistency_scores.append(max(0.0, consistency))
            elif 'density' in metric.metric_type.value:
                expected_value = metric.experimental_value * expected_density_score
                actual_value = metric.simulated_value
                consistency = 1.0 - abs(expected_value - actual_value) / max(expected_value, actual_value, 1.0)
                consistency_scores.append(max(0.0, consistency))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
