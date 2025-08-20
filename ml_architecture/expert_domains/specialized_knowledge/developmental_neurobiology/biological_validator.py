#!/usr/bin/env python3
"""
🧠 Biological Validation Framework for Pillar 1: Basic Neural Dynamics
Validates neural dynamics against biological benchmarks and neuroscience literature

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Ensure neural simulation matches biological reality
**Validation Level:** Biological accuracy verification
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, field
from enum import Enum
import json

class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"    # Must pass for biological realism
    WARNING = "warning"      # Should be within biological ranges
    INFO = "info"           # Informational validation

@dataclass
class BiologicalBenchmark:
    """Biological benchmark with validation ranges"""
    name: str
    min_value: float
    max_value: float
    unit: str
    source: str
    validation_level: ValidationLevel
    description: str

@dataclass
class ValidationResult:
    """Result of biological validation"""
    benchmark: BiologicalBenchmark
    actual_value: float
    passed: bool
    deviation: float
    message: str

class BiologicalValidator:
    """Validates neural dynamics against biological benchmarks"""
    
    def __init__(self):
        # Biological benchmarks from neuroscience literature
        self.benchmarks = self._load_biological_benchmarks()
        
        # Validation results storage
        self.validation_history = []
        self.current_results = []
        
    def _load_biological_benchmarks(self) -> Dict[str, BiologicalBenchmark]:
        """Load biological benchmarks from neuroscience literature"""
        
        benchmarks = {
            # Firing rate benchmarks (Hz)
            "pfc_firing_rate": BiologicalBenchmark(
                name="PFC Firing Rate",
                min_value=0.1,
                max_value=50.0,
                unit="Hz",
                source="Goldman-Rakic et al. (1999)",
                validation_level=ValidationLevel.CRITICAL,
                description="PFC neurons typically fire at 0.1-50 Hz"
            ),
            
            "bg_firing_rate": BiologicalBenchmark(
                name="Basal Ganglia Firing Rate", 
                min_value=1.0,
                max_value=100.0,
                unit="Hz",
                source="Wilson (1993)",
                validation_level=ValidationLevel.CRITICAL,
                description="BG neurons fire at 1-100 Hz, with fast-spiking interneurons"
            ),
            
            "thalamus_firing_rate": BiologicalBenchmark(
                name="Thalamus Firing Rate",
                min_value=0.5,
                max_value=200.0,
                unit="Hz",
                source="Sherman & Guillery (2006)",
                validation_level=ValidationLevel.CRITICAL,
                description="Thalamic neurons show burst and tonic firing modes"
            ),
            
            # Synchrony benchmarks (0-1)
            "pfc_synchrony": BiologicalBenchmark(
                name="PFC Synchrony",
                min_value=0.0,
                max_value=0.8,
                unit="correlation",
                source="Buzsaki (2006)",
                validation_level=ValidationLevel.WARNING,
                description="PFC shows moderate synchrony during cognitive tasks"
            ),
            
            "bg_synchrony": BiologicalBenchmark(
                name="BG Synchrony",
                min_value=0.0,
                max_value=0.9,
                unit="correlation", 
                source="Graybiel (2008)",
                validation_level=ValidationLevel.WARNING,
                description="BG shows high synchrony in movement control"
            ),
            
            # Oscillation power benchmarks
            "alpha_oscillation": BiologicalBenchmark(
                name="Alpha Oscillation Power (8-13 Hz)",
                min_value=0.0,
                max_value=1.0,
                unit="power",
                source="Klimesch (1999)",
                validation_level=ValidationLevel.INFO,
                description="Alpha oscillations prominent in relaxed states"
            ),
            
            "beta_oscillation": BiologicalBenchmark(
                name="Beta Oscillation Power (13-30 Hz)",
                min_value=0.0,
                max_value=1.0,
                unit="power",
                source="Engel & Fries (2010)",
                validation_level=ValidationLevel.INFO,
                description="Beta oscillations in motor control and attention"
            ),
            
            "gamma_oscillation": BiologicalBenchmark(
                name="Gamma Oscillation Power (30-100 Hz)",
                min_value=0.0,
                max_value=1.0,
                unit="power",
                source="Fries (2009)",
                validation_level=ValidationLevel.INFO,
                description="Gamma oscillations in attention and binding"
            ),
            
            # Loop stability benchmarks
            "loop_stability": BiologicalBenchmark(
                name="Cortical-Subcortical Loop Stability",
                min_value=0.3,
                max_value=1.0,
                unit="stability",
                source="Alexander et al. (1986)",
                validation_level=ValidationLevel.CRITICAL,
                description="Loop should maintain stable dynamics"
            ),
            
            # Plasticity benchmarks
            "stdp_weight_change": BiologicalBenchmark(
                name="STDP Weight Change",
                min_value=-0.5,
                max_value=0.5,
                unit="weight_change",
                source="Bi & Poo (2001)",
                validation_level=ValidationLevel.WARNING,
                description="STDP weight changes should be reasonable"
            ),
            
            # Population size benchmarks
            "pfc_population_size": BiologicalBenchmark(
                name="PFC Population Size",
                min_value=100,
                max_value=10000,
                unit="neurons",
                source="Elston (2003)",
                validation_level=ValidationLevel.INFO,
                description="PFC has large pyramidal cell populations"
            ),
            
            "bg_population_size": BiologicalBenchmark(
                name="BG Population Size",
                min_value=50,
                max_value=5000,
                unit="neurons",
                source="Tepper et al. (2004)",
                validation_level=ValidationLevel.INFO,
                description="BG has diverse neuron types and sizes"
            )
        }
        
        return benchmarks
    
    def validate_firing_rates(self, firing_rates: Dict[str, float]) -> List[ValidationResult]:
        """Validate firing rates against biological benchmarks"""
        results = []
        
        for population, rate in firing_rates.items():
            benchmark_key = f"{population}_firing_rate"
            if benchmark_key in self.benchmarks:
                benchmark = self.benchmarks[benchmark_key]
                passed = benchmark.min_value <= rate <= benchmark.max_value
                deviation = abs(rate - (benchmark.max_value + benchmark.min_value) / 2)
                
                result = ValidationResult(
                    benchmark=benchmark,
                    actual_value=rate,
                    passed=passed,
                    deviation=deviation,
                    message=f"{population} firing rate: {rate:.2f} Hz (expected {benchmark.min_value}-{benchmark.max_value} Hz)"
                )
                results.append(result)
        
        return results
    
    def validate_synchrony(self, synchrony: Dict[str, float]) -> List[ValidationResult]:
        """Validate synchrony measures against biological benchmarks"""
        results = []
        
        for population, sync in synchrony.items():
            benchmark_key = f"{population}_synchrony"
            if benchmark_key in self.benchmarks:
                benchmark = self.benchmarks[benchmark_key]
                passed = benchmark.min_value <= sync <= benchmark.max_value
                deviation = abs(sync - (benchmark.max_value + benchmark.min_value) / 2)
                
                result = ValidationResult(
                    benchmark=benchmark,
                    actual_value=sync,
                    passed=passed,
                    deviation=deviation,
                    message=f"{population} synchrony: {sync:.3f} (expected {benchmark.min_value}-{benchmark.max_value})"
                )
                results.append(result)
        
        return results
    
    def validate_oscillations(self, oscillation_power: Dict[str, float]) -> List[ValidationResult]:
        """Validate oscillation power against biological benchmarks"""
        results = []
        
        # Check if oscillation power is within reasonable bounds
        for population, power in oscillation_power.items():
            # General oscillation power validation
            passed = 0.0 <= power <= 1.0
            deviation = abs(power - 0.5)
            
            result = ValidationResult(
                benchmark=self.benchmarks["alpha_oscillation"],  # Use alpha as reference
                actual_value=power,
                passed=passed,
                deviation=deviation,
                message=f"{population} oscillation power: {power:.3f} (should be 0-1)"
            )
            results.append(result)
        
        return results
    
    def validate_loop_stability(self, stability: float) -> List[ValidationResult]:
        """Validate cortical-subcortical loop stability"""
        benchmark = self.benchmarks["loop_stability"]
        passed = benchmark.min_value <= stability <= benchmark.max_value
        deviation = abs(stability - (benchmark.max_value + benchmark.min_value) / 2)
        
        result = ValidationResult(
            benchmark=benchmark,
            actual_value=stability,
            passed=passed,
            deviation=deviation,
            message=f"Loop stability: {stability:.3f} (expected {benchmark.min_value}-{benchmark.max_value})"
        )
        
        return [result]
    
    def validate_plasticity(self, weight_changes: Dict[str, float]) -> List[ValidationResult]:
        """Validate synaptic plasticity changes"""
        results = []
        benchmark = self.benchmarks["stdp_weight_change"]
        
        for synapse, change in weight_changes.items():
            passed = benchmark.min_value <= change <= benchmark.max_value
            deviation = abs(change - (benchmark.max_value + benchmark.min_value) / 2)
            
            result = ValidationResult(
                benchmark=benchmark,
                actual_value=change,
                passed=passed,
                deviation=deviation,
                message=f"{synapse} weight change: {change:.4f} (expected {benchmark.min_value}-{benchmark.max_value})"
            )
            results.append(result)
        
        return results
    
    def validate_population_sizes(self, sizes: Dict[str, int]) -> List[ValidationResult]:
        """Validate neural population sizes"""
        results = []
        
        for population, size in sizes.items():
            benchmark_key = f"{population}_population_size"
            if benchmark_key in self.benchmarks:
                benchmark = self.benchmarks[benchmark_key]
                passed = benchmark.min_value <= size <= benchmark.max_value
                deviation = abs(size - (benchmark.max_value + benchmark.min_value) / 2)
                
                result = ValidationResult(
                    benchmark=benchmark,
                    actual_value=size,
                    passed=passed,
                    deviation=deviation,
                    message=f"{population} population size: {size} neurons (expected {benchmark.min_value}-{benchmark.max_value})"
                )
                results.append(result)
        
        return results
    
    def validate_neural_dynamics(self, biological_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive validation of neural dynamics"""
        
        all_results = []
        
        # Validate firing rates
        if "firing_rates" in biological_metrics:
            firing_results = self.validate_firing_rates(biological_metrics["firing_rates"])
            all_results.extend(firing_results)
        
        # Validate synchrony
        if "synchrony" in biological_metrics:
            synchrony_results = self.validate_synchrony(biological_metrics["synchrony"])
            all_results.extend(synchrony_results)
        
        # Validate oscillations
        if "oscillation_power" in biological_metrics:
            oscillation_results = self.validate_oscillations(biological_metrics["oscillation_power"])
            all_results.extend(oscillation_results)
        
        # Validate loop stability
        if "loop_stability" in biological_metrics:
            stability_results = self.validate_loop_stability(biological_metrics["loop_stability"])
            all_results.extend(stability_results)
        
        # Validate plasticity
        if "plasticity_changes" in biological_metrics:
            plasticity_results = self.validate_plasticity(biological_metrics["plasticity_changes"])
            all_results.extend(plasticity_results)
        
        # Validate population sizes
        if "population_sizes" in biological_metrics:
            size_results = self.validate_population_sizes(biological_metrics["population_sizes"])
            all_results.extend(size_results)
        
        # Calculate validation summary
        critical_results = [r for r in all_results if r.benchmark.validation_level == ValidationLevel.CRITICAL]
        warning_results = [r for r in all_results if r.benchmark.validation_level == ValidationLevel.WARNING]
        info_results = [r for r in all_results if r.benchmark.validation_level == ValidationLevel.INFO]
        
        critical_passed = sum(1 for r in critical_results if r.passed)
        warning_passed = sum(1 for r in warning_results if r.passed)
        
        validation_summary = {
            "total_validations": len(all_results),
            "critical_validations": len(critical_results),
            "critical_passed": critical_passed,
            "critical_failed": len(critical_results) - critical_passed,
            "warning_validations": len(warning_results),
            "warning_passed": warning_passed,
            "warning_failed": len(warning_results) - warning_passed,
            "info_validations": len(info_results),
            "overall_score": (critical_passed + 0.5 * warning_passed) / (len(critical_results) + 0.5 * len(warning_results)) if (len(critical_results) + len(warning_results)) > 0 else 0.0,
            "biological_realism": critical_passed == len(critical_results),  # Must pass all critical
            "results": all_results
        }
        
        # Store validation history
        self.validation_history.append({
            "timestamp": np.datetime64('now'),
            "summary": validation_summary,
            "metrics": biological_metrics
        })
        
        self.current_results = all_results
        
        return validation_summary
    
    def get_validation_report(self) -> str:
        """Generate a human-readable validation report"""
        if not self.current_results:
            return "No validation results available."
        
        report = "🧠 Biological Validation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Group by validation level
        critical_results = [r for r in self.current_results if r.benchmark.validation_level == ValidationLevel.CRITICAL]
        warning_results = [r for r in self.current_results if r.benchmark.validation_level == ValidationLevel.WARNING]
        info_results = [r for r in self.current_results if r.benchmark.validation_level == ValidationLevel.INFO]
        
        # Critical validations
        if critical_results:
            report += "🔴 CRITICAL VALIDATIONS:\n"
            for result in critical_results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                report += f"  {status} {result.message}\n"
            report += "\n"
        
        # Warning validations
        if warning_results:
            report += "🟡 WARNING VALIDATIONS:\n"
            for result in warning_results:
                status = "✅ PASS" if result.passed else "⚠️  WARN"
                report += f"  {status} {result.message}\n"
            report += "\n"
        
        # Info validations
        if info_results:
            report += "🔵 INFO VALIDATIONS:\n"
            for result in info_results:
                status = "✅ PASS" if result.passed else "ℹ️  INFO"
                report += f"  {status} {result.message}\n"
            report += "\n"
        
        # Summary
        critical_passed = sum(1 for r in critical_results if r.passed)
        warning_passed = sum(1 for r in warning_results if r.passed)
        
        report += "📊 SUMMARY:\n"
        report += f"  Critical: {critical_passed}/{len(critical_results)} passed\n"
        report += f"  Warnings: {warning_passed}/{len(warning_results)} passed\n"
        report += f"  Overall Score: {((critical_passed + 0.5 * warning_passed) / (len(critical_results) + 0.5 * len(warning_results)) * 100):.1f}%\n"
        
        if critical_passed == len(critical_results):
            report += "  🎉 BIOLOGICAL REALISM ACHIEVED!\n"
        else:
            report += "  ⚠️  CRITICAL VALIDATIONS FAILED - NOT BIOLOGICALLY REALISTIC\n"
        
        return report
    
    def export_validation_data(self, filename: str):
        """Export validation data to JSON file"""
        data = {
            "validation_history": [
                {
                    "timestamp": str(h["timestamp"]),
                    "summary": h["summary"],
                    "metrics": h["metrics"]
                }
                for h in self.validation_history
            ],
            "benchmarks": {
                name: {
                    "min_value": b.min_value,
                    "max_value": b.max_value,
                    "unit": b.unit,
                    "source": b.source,
                    "validation_level": b.validation_level.value,
                    "description": b.description
                }
                for name, b in self.benchmarks.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
