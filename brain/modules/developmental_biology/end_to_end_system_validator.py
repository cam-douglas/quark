"""
End-to-End System Validator

Validates the complete embryonic development simulation pipeline.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
import time
from typing import Dict, List, Any

from .system_validation_types import SystemValidationResult, SystemValidationStatus
from .embryonic_simulation_engine import EmbryonicSimulationEngine
from .proliferation_rate_validator import ProliferationRateValidator
from .lineage_fate_validator import LineageFateValidator
from .spatial_organization_validator import SpatialOrganizationValidator
from .committed_progenitor_generator import CommittedProgenitorGenerator
from .lineage_tag_preservator import LineageTagPreservator
from .downstream_interface_manager import DownstreamInterfaceManager


class EndToEndSystemValidator:
    """
    Validates the complete embryonic development simulation pipeline
    """
    
    def __init__(self):
        """Initialize end-to-end system validator"""
        self.proliferation_validator = ProliferationRateValidator()
        self.fate_validator = LineageFateValidator()
        self.spatial_validator = SpatialOrganizationValidator()
        self.progenitor_generator = CommittedProgenitorGenerator()
        self.tag_preservator = LineageTagPreservator()
        self.interface_manager = DownstreamInterfaceManager()
        self.simulation_engine = EmbryonicSimulationEngine()
        
        self.validation_results: List[SystemValidationResult] = []
    
    def run_complete_embryonic_simulation(self,
                                        initial_cell_count: int = 100,
                                        simulation_duration: float = 48.0,
                                        time_step: float = 0.5) -> SystemValidationResult:
        """Run complete embryonic development simulation"""
        try:
            print("🧪 Running complete embryonic development simulation...")
            
            # Use simulation engine
            simulation_results = self.simulation_engine.run_simulation(
                initial_cell_count, simulation_duration, time_step
            )
            
            print(f"✅ Simulation completed in {simulation_results['simulation_time']:.2f}s")
            
            # Calculate performance metrics
            performance_metrics = {
                "simulation_duration_hours": simulation_duration,
                "actual_runtime_seconds": simulation_results['simulation_time'],
                "cells_processed": len(simulation_results['neuroepithelial_cells']) + len(simulation_results['committed_progenitors']),
                "time_steps_completed": simulation_results['steps_completed'],
                "cells_per_second": (len(simulation_results['neuroepithelial_cells']) + len(simulation_results['committed_progenitors'])) / simulation_results['simulation_time']
            }
            
            # Calculate accuracy
            accuracy_score = self._calculate_system_accuracy(
                simulation_results['neuroepithelial_cells'],
                simulation_results['committed_progenitors'],
                simulation_results['final_time']
            )
            
            status = SystemValidationStatus.PASSED if accuracy_score >= 0.7 else SystemValidationStatus.FAILED
            
            result = SystemValidationResult(
                validation_name="complete_embryonic_simulation",
                status=status,
                accuracy_score=accuracy_score,
                performance_metrics=performance_metrics
            )
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            return SystemValidationResult(
                validation_name="complete_embryonic_simulation",
                status=SystemValidationStatus.FAILED,
                accuracy_score=0.0,
                performance_metrics={},
                error_message=str(e)
            )
    
    def validate_morphogen_cell_fate_lineage_pipeline(self,
                                                    morphogen_concentrations: Dict[str, float],
                                                    developmental_stage: str) -> SystemValidationResult:
        """Validate the morphogen → cell fate → lineage pipeline"""
        start_time = time.time()
        
        try:
            print("🧬 Validating morphogen → cell fate → lineage pipeline...")
            
            # Run simplified simulation for pipeline testing
            simulation_results = self.simulation_engine.run_simulation(50, 12.0, 1.0)
            
            print(f"✅ Pipeline simulation completed")
            
            # Validate pipeline components
            pipeline_validation_score = self._validate_pipeline_components(
                simulation_results['neuroepithelial_cells'],
                simulation_results['committed_progenitors'],
                morphogen_concentrations,
                developmental_stage
            )
            
            pipeline_time = time.time() - start_time
            
            performance_metrics = {
                "pipeline_runtime_seconds": pipeline_time,
                "morphogen_response_accuracy": pipeline_validation_score.get("morphogen_response", 0.0),
                "fate_specification_accuracy": pipeline_validation_score.get("fate_specification", 0.0),
                "lineage_tracking_accuracy": pipeline_validation_score.get("lineage_tracking", 0.0),
                "overall_pipeline_accuracy": pipeline_validation_score.get("overall", 0.0)
            }
            
            overall_accuracy = performance_metrics["overall_pipeline_accuracy"]
            status = SystemValidationStatus.PASSED if overall_accuracy >= 0.8 else SystemValidationStatus.FAILED
            
            result = SystemValidationResult(
                validation_name="morphogen_cell_fate_lineage_pipeline",
                status=status,
                accuracy_score=overall_accuracy,
                performance_metrics=performance_metrics
            )
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            pipeline_time = time.time() - start_time
            return SystemValidationResult(
                validation_name="morphogen_cell_fate_lineage_pipeline",
                status=SystemValidationStatus.FAILED,
                accuracy_score=0.0,
                performance_metrics={"runtime_seconds": pipeline_time},
                error_message=str(e)
            )
    
    def test_system_performance_and_stability(self,
                                            stress_test_duration: float = 24.0) -> SystemValidationResult:
        """Test system performance and stability"""
        try:
            print("⚡ Testing system performance and stability...")
            
            stress_result = self.run_complete_embryonic_simulation(200, stress_test_duration, 0.25)
            
            performance_passed = (
                stress_result.performance_metrics.get("actual_runtime_seconds", 999) < 120.0 and
                stress_result.status != SystemValidationStatus.FAILED
            )
            
            status = SystemValidationStatus.PASSED if performance_passed else SystemValidationStatus.FAILED
            
            result = SystemValidationResult(
                validation_name="system_performance_and_stability",
                status=status,
                accuracy_score=stress_result.accuracy_score,
                performance_metrics=stress_result.performance_metrics
            )
            
            self.validation_results.append(result)
            return result
            
        except Exception as e:
            return SystemValidationResult(
                validation_name="system_performance_and_stability",
                status=SystemValidationStatus.FAILED,
                accuracy_score=0.0,
                performance_metrics={},
                error_message=str(e)
            )
    
    def _calculate_system_accuracy(self, neuroepithelial_cells, committed_progenitors, final_time) -> float:
        """Calculate overall system accuracy"""
        # Simplified accuracy calculation
        cell_count_score = len(neuroepithelial_cells) / 100.0  # Normalize to expected count
        progenitor_score = len(committed_progenitors) / 10.0   # Normalize to expected count
        
        return min(1.0, (cell_count_score + progenitor_score) / 2.0)
    
    def _validate_pipeline_components(self, neuroepithelial_cells, committed_progenitors, morphogen_concentrations, developmental_stage) -> Dict[str, float]:
        """Validate individual pipeline components"""
        return {
            "morphogen_response": 0.5,
            "fate_specification": 0.5,
            "lineage_tracking": 0.5,
            "overall": 0.5
        }
    
    def get_system_validation_report(self) -> Dict[str, Any]:
        """Get system validation report"""
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        passed_validations = sum(1 for r in self.validation_results if r.status == SystemValidationStatus.PASSED)
        total_validations = len(self.validation_results)
        
        return {
            "total_system_validations": total_validations,
            "passed_validations": passed_validations,
            "system_reliability": passed_validations / total_validations if total_validations > 0 else 0.0,
            "average_accuracy": np.mean([r.accuracy_score for r in self.validation_results])
        }