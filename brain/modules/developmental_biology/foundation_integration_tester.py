"""
Foundation Layer Integration Tester

Tests integration between experimental data validation systems and 
the completed foundation layer (morphogen gradients, ventricular topology).

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, List, Any

from .integration_test_types import IntegrationTestResult, IntegrationTestStatus
from .morphogen_integration_tester import MorphogenIntegrationTester
from .spatial_integration_tester import SpatialIntegrationTester
from .proliferation_rate_validator import ProliferationRateValidator
from .lineage_fate_validator import LineageFateValidator
from .spatial_organization_validator import SpatialOrganizationValidator


class FoundationIntegrationTester:
    """
    Tests integration between foundation layer and experimental validation systems
    """
    
    def __init__(self):
        """Initialize foundation integration tester"""
        self.morphogen_tester = MorphogenIntegrationTester()
        self.spatial_tester = SpatialIntegrationTester()
        self.proliferation_validator = ProliferationRateValidator()
        self.fate_validator = LineageFateValidator()
        self.spatial_validator = SpatialOrganizationValidator()
        self.integration_results: List[IntegrationTestResult] = []
    
    def test_comprehensive_foundation_validation(self,
                                                foundation_state: Dict[str, Any],
                                                developmental_stage: str) -> Dict[str, IntegrationTestResult]:
        """
        Comprehensive test of all foundation-validation integrations
        
        Args:
            foundation_state: Complete foundation layer state
            developmental_stage: E8.5, E9.5, E10.5, etc.
        
        Returns:
            Dictionary of all integration test results
        """
        results = {}
        
        # Test morphogen-proliferation integration
        if 'morphogen_concentrations' in foundation_state:
            results['morphogen_proliferation'] = self.morphogen_tester.test_morphogen_proliferation_integration(
                foundation_state['morphogen_concentrations'],
                developmental_stage
            )
        
        # Test ventricular-spatial integration
        if 'ventricular_topology' in foundation_state:
            results['ventricular_spatial'] = self.spatial_tester.test_ventricular_spatial_integration(
                foundation_state['ventricular_topology'],
                developmental_stage
            )
        
        # Test meninges-spatial integration (if available)
        if 'meninges_scaffold' in foundation_state:
            results['meninges_spatial'] = self.spatial_tester.test_meninges_spatial_integration(
                foundation_state['meninges_scaffold'],
                developmental_stage
            )
        
        return results
    
    def run_full_integration_test_suite(self, 
                                      foundation_state: Dict[str, Any],
                                      developmental_stages: List[str]) -> Dict[str, Any]:
        """
        Run complete integration test suite across multiple developmental stages
        
        Args:
            foundation_state: Complete foundation layer state
            developmental_stages: List of stages to test (e.g., ['E8.5', 'E9.5', 'E10.5'])
        
        Returns:
            Comprehensive test results
        """
        all_results = {}
        
        for stage in developmental_stages:
            stage_results = self.test_comprehensive_foundation_validation(
                foundation_state, stage
            )
            all_results[stage] = stage_results
        
        # Calculate overall integration metrics
        overall_metrics = self._calculate_overall_integration_metrics(all_results)
        
        return {
            'stage_results': all_results,
            'overall_metrics': overall_metrics,
            'test_summary': self._generate_test_summary(all_results)
        }
    
    def _calculate_overall_integration_metrics(self, 
                                             all_results: Dict[str, Dict[str, IntegrationTestResult]]) -> Dict[str, float]:
        """Calculate overall integration metrics across all tests"""
        all_scores = []
        all_accuracies = []
        passed_tests = 0
        total_tests = 0
        
        for stage_results in all_results.values():
            for test_result in stage_results.values():
                all_scores.append(test_result.integration_score)
                all_accuracies.append(test_result.experimental_accuracy)
                if test_result.test_status == IntegrationTestStatus.PASSED:
                    passed_tests += 1
                total_tests += 1
        
        return {
            'overall_integration_score': np.mean(all_scores) if all_scores else 0.0,
            'overall_experimental_accuracy': np.mean(all_accuracies) if all_accuracies else 0.0,
            'test_pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'total_tests_run': total_tests
        }
    
    def _generate_test_summary(self, 
                             all_results: Dict[str, Dict[str, IntegrationTestResult]]) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        summary = {
            'foundation_components_tested': set(),
            'validation_components_tested': set(),
            'experimental_data_sources': {
                'Calegari & Huttner 2005 J Neurosci',
                'Bocanegra-Moreno et al. 2023 Nat Physics',
                'Delás et al. 2022 Dev Cell',
                'Chen et al. 2017 Toxicol Pathol'
            },
            'integration_quality': 'PASSED',
            'recommendations': []
        }
        
        for stage_results in all_results.values():
            for test_result in stage_results.values():
                summary['foundation_components_tested'].add(test_result.foundation_component)
                summary['validation_components_tested'].add(test_result.validation_component)
                
                if test_result.test_status == IntegrationTestStatus.FAILED:
                    summary['integration_quality'] = 'NEEDS_IMPROVEMENT'
                    summary['recommendations'].append(
                        f"Improve {test_result.foundation_component} - {test_result.validation_component} integration"
                    )
        
        # Convert sets to lists for JSON serialization
        summary['foundation_components_tested'] = list(summary['foundation_components_tested'])
        summary['validation_components_tested'] = list(summary['validation_components_tested'])
        
        return summary
    
    def get_integration_report(self) -> Dict[str, Any]:
        """Get comprehensive integration test report"""
        return {
            'test_results': [
                {
                    'test_name': result.test_name,
                    'foundation_component': result.foundation_component,
                    'validation_component': result.validation_component,
                    'status': result.test_status.value,
                    'experimental_accuracy': result.experimental_accuracy,
                    'integration_score': result.integration_score,
                    'error_message': result.error_message
                }
                for result in self.integration_results
            ],
            'summary': {
                'total_tests': len(self.integration_results),
                'passed_tests': sum(1 for r in self.integration_results if r.test_status == IntegrationTestStatus.PASSED),
                'average_accuracy': np.mean([r.experimental_accuracy for r in self.integration_results]) if self.integration_results else 0.0,
                'average_integration_score': np.mean([r.integration_score for r in self.integration_results]) if self.integration_results else 0.0
            }
        }
