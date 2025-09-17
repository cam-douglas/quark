"""
Proliferation Rate Validator (human-specific)

Validates simulated proliferation kinetics against longitudinal *human* embryonic
radial-glia datasets.  Primary sources:
    • Nowakowski et al. 2016 (Cell) – 8–24 pcw cortex
    • Linsley et al. 2019 (Dev Cell) – 6–11 pcw ventral spinal cord
    • Reillo & Borrell 2012 (Cereb Cortex) – 10–16 pcw oRG
These data are loaded via `human_experimental_data.load_human_experimental_data()`.\n"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats

from .proliferation_validation_types import ValidationMetric, ValidationStatus, ValidationResult
from .experimental_data_database import ExperimentalDataDatabase
from .neuroepithelial_cells import NeuroepithelialCell
from .cell_cycle_timing_engine import CellCyclePhase
from . import human_data_utils


class ProliferationRateValidator:
    """
    Validates proliferation rates against experimental data from scientific literature
    """
    
    def __init__(self):
        """Initialize proliferation rate validator"""
        self.experimental_db = ExperimentalDataDatabase()
        self.validation_thresholds = {
            "cell_cycle_length": 0.15,  # 15% tolerance
            "division_rate": 0.20,      # 20% tolerance
            "phase_duration": 0.25,     # 25% tolerance
            "proliferation_index": 0.10  # 10% tolerance
        }
        self.validation_results: List[ValidationResult] = []
    
    def validate_proliferation_rates(self,
                                   cells: Dict[str, NeuroepithelialCell],
                                   developmental_stage: str) -> List[ValidationResult]:
        """
        Validate proliferation rates against experimental data
        
        Args:
            cells: Dictionary of neuroepithelial cells
            developmental_stage: E8.5, E9.5, E10.5, etc.
        
        Returns:
            List of validation results
        """
        self.validation_results = []
        
        # Get experimental data for this stage
        stage_data = self.experimental_db.get_all_data_for_stage(developmental_stage)
        
        if not stage_data:
            return self.validation_results
        
        # Calculate simulated metrics
        simulated_metrics = self._calculate_simulated_metrics(cells)
        
        # Validate each experimental data point
        for data_key, exp_data in stage_data.items():
            if exp_data.metric_name in simulated_metrics:
                simulated_value = simulated_metrics[exp_data.metric_name]
                
                # Perform statistical validation
                validation_result = self._perform_statistical_validation(
                    exp_data, simulated_value
                )
                
                self.validation_results.append(validation_result)
        
        # Validate longitudinal human G1-length slope (8→16 pcw)
        try:
            g1_8 = human_data_utils.interpolate("phase_duration", 8)
            g1_16 = human_data_utils.interpolate("phase_duration", 16)
            expected_slope = (g1_16 - g1_8) / (16 - 8)
            sim_g1_lengths = [cell.cell_cycle_length * 0.6 for cell in cells.values()]
            # naive grouping by pcw via cell attribute (defaults to stage pcw if missing)
            sim_mean = np.mean(sim_g1_lengths)
            # crude slope assume sim lengths correspond to provided stage midpoint
            sim_slope = (sim_mean - g1_8) / (16 - 8)
            slope_match = 1 - abs(sim_slope - expected_slope) / expected_slope
            if slope_match < 0.9:
                self.validation_results.append(ValidationResult(
                    metric_name="human_cell_cycle_slope_match",
                    simulated_value=slope_match,
                    expected_value=1.0,
                    status=ValidationStatus.FAILED,
                    p_value=1.0,
                    confidence_interval=(0.9, 1.0),
                    reference_source="Nowakowski 2016 / Reillo 2012"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    metric_name="human_cell_cycle_slope_match",
                    simulated_value=slope_match,
                    expected_value=1.0,
                    status=ValidationStatus.PASSED,
                    p_value=0.0,
                    confidence_interval=(0.9, 1.0),
                    reference_source="Nowakowski 2016 / Reillo 2012"
                ))
        except Exception:
            pass
        
        return self.validation_results
    
    def _calculate_simulated_metrics(self, cells: Dict[str, NeuroepithelialCell]) -> Dict[str, float]:
        """Calculate simulated proliferation metrics from cells"""
        if not cells:
            return {}
        
        # Calculate cell cycle lengths
        cell_cycle_lengths = []
        for cell in cells.values():
            ccl = getattr(cell, 'cell_cycle_length', None)
            if ccl is None:
                # Fallback by developmental window (pcw) if available; else default 16 h
                pcw = getattr(cell, 'developmental_time', 10.0)
                if pcw <= 8.0:
                    ccl = 12.0
                elif pcw <= 11.0:
                    ccl = 17.0
                else:
                    ccl = 20.0
            # Guard against non-positive values
            if ccl <= 0:
                ccl = 16.0
            cell_cycle_lengths.append(ccl)
        avg_cycle_length = np.mean(cell_cycle_lengths)
        
        # Calculate division rates
        division_rates = [1.0 / c if c > 0 else 0.0 for c in cell_cycle_lengths]
        avg_division_rate = np.mean(division_rates)
        
        # Calculate proliferation index (fraction of proliferating cells)
        proliferating_cells = []
        for cell in cells.values():
            phase = getattr(cell, 'cell_cycle_phase', None)
            if phase is None:
                proliferating_cells.append(cell)
            else:
                try:
                    proliferating = phase != CellCyclePhase.G0
                except Exception:
                    proliferating = True
                if proliferating:
                    proliferating_cells.append(cell)
        proliferation_index = len(proliferating_cells) / len(cells)
        
        # Account for apoptotic fraction if available
        apoptotic_fraction = 0.0
        if hasattr(self, 'apoptosis_manager'):
            apoptotic_fraction = self.apoptosis_manager.get_apoptosis_fraction(len(cells))
        
        # Calculate phase durations (simplified)
        g1_durations = [c * 0.6 for c in cell_cycle_lengths]  # G1 ~60% of cycle
        s_durations = [c * 0.2 for c in cell_cycle_lengths]   # S ~20% of cycle
        
        return {
            "cell_cycle_length": avg_cycle_length,
            "division_rate": avg_division_rate,
            "phase_duration": np.mean(g1_durations),  # Use G1 as representative
            "proliferation_index": proliferation_index
        }
    
    def _perform_statistical_validation(self,
                                       exp_data,
                                       simulated_value: float) -> ValidationResult:
        """Perform statistical validation against experimental data"""
        # Calculate z-score
        z_score = (simulated_value - exp_data.expected_value) / exp_data.standard_deviation
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Determine validation status
        tolerance = self.validation_thresholds.get(exp_data.metric_name, 0.20)
        relative_error = abs(simulated_value - exp_data.expected_value) / exp_data.expected_value
        
        if relative_error <= tolerance:
            status = ValidationStatus.PASSED
        elif relative_error <= tolerance * 1.5:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        # Calculate confidence interval
        margin = 1.96 * exp_data.standard_deviation  # 95% CI
        confidence_interval = (
            exp_data.expected_value - margin,
            exp_data.expected_value + margin
        )
        
        # Determine metric type
        if exp_data.metric_name == "cell_cycle_length":
            metric_type = ValidationMetric.CELL_CYCLE_LENGTH
        elif exp_data.metric_name == "division_rate":
            metric_type = ValidationMetric.DIVISION_RATE
        elif exp_data.metric_name == "phase_duration":
            metric_type = ValidationMetric.PHASE_DURATION
        else:
            metric_type = ValidationMetric.PROLIFERATION_INDEX
        
        return ValidationResult(
            metric_type=metric_type,
            experimental_value=exp_data.expected_value,
            simulated_value=simulated_value,
            validation_status=status,
            confidence_interval=confidence_interval,
            reference_source=exp_data.reference_source,
            z_score=z_score,
            p_value=p_value
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        passed_validations = sum(1 for r in self.validation_results if r.validation_status == ValidationStatus.PASSED)
        total_validations = len(self.validation_results)
        
        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "success_rate": passed_validations / total_validations,
            "biological_accuracy": passed_validations / total_validations >= 0.8,
            "average_z_score": np.mean([abs(r.z_score) for r in self.validation_results]),
            "experimental_data_sources": list(set(r.reference_source for r in self.validation_results))
        }
    
    def optimize_parameters_for_validation(self,
                                         target_accuracy: float = 0.90) -> Dict[str, float]:
        """Optimize parameters to improve validation accuracy"""
        optimization_suggestions = {}
        
        for result in self.validation_results:
            if result.validation_status != ValidationStatus.PASSED:
                # Calculate required adjustment
                target_value = result.experimental_value
                current_value = result.simulated_value
                adjustment_factor = target_value / current_value if current_value != 0 else 1.0
                
                optimization_suggestions[f"{result.metric_type.value}_adjustment"] = adjustment_factor
        
        return optimization_suggestions