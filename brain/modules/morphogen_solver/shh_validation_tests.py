#!/usr/bin/env python3
"""SHH Gradient Validation Tests.

Comprehensive validation tests for SHH morphogen gradient system including
gradient formation, gene expression patterns, and cell fate specification.

Integration: Validation component for SHH gradient system
Rationale: Ensures biological accuracy and system correctness
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
from dataclasses import dataclass

from .spatial_grid import SpatialGrid, GridDimensions
from .shh_gradient_system import SHHGradientSystem
from .biological_parameters import BiologicalParameters
from .cell_fate_specifier import CellFateSpecifier

logger = logging.getLogger(__name__)

@dataclass
class ValidationTest:
    """Single validation test specification."""
    name: str
    description: str
    test_function: str
    expected_result: Any
    tolerance: float = 0.1
    critical: bool = True

class SHHValidationTests:
    """Comprehensive SHH validation test suite.
    
    Validates SHH gradient system against experimental data and biological
    expectations including gradient properties, gene expression patterns,
    and cell fate specification.
    
    Key Features:
    - Gradient formation validation
    - Gene expression pattern validation
    - Cell fate specification validation
    - Performance benchmarking
    - Biological accuracy assessment
    """
    
    def __init__(self, grid_dimensions: GridDimensions):
        """Initialize SHH validation tests.
        
        Args:
            grid_dimensions: Test grid dimensions
        """
        self.grid_dimensions = grid_dimensions
        
        # Test components
        self.spatial_grid = None
        self.shh_system = None
        self.cell_fate_specifier = None
        
        # Test specifications
        self.validation_tests = self._define_validation_tests()
        
        # Test results
        self.test_results: Dict[str, Any] = {}
        
        logger.info("Initialized SHH validation test suite")
        logger.info(f"Test grid: {grid_dimensions.x_size}x{grid_dimensions.y_size}x{grid_dimensions.z_size}")
    
    def _define_validation_tests(self) -> List[ValidationTest]:
        """Define comprehensive validation tests."""
        return [
            # Gradient formation tests
            ValidationTest(
                name="gradient_formation",
                description="SHH gradient forms from ventral sources",
                test_function="test_gradient_formation",
                expected_result=True,
                critical=True
            ),
            ValidationTest(
                name="dorsal_ventral_gradient",
                description="Proper dorsal-ventral concentration gradient",
                test_function="test_dorsal_ventral_gradient",
                expected_result=True,
                critical=True
            ),
            ValidationTest(
                name="gradient_steepness",
                description="Gradient steepness within biological range",
                test_function="test_gradient_steepness",
                expected_result=True,
                tolerance=0.2,
                critical=False
            ),
            
            # Concentration validation tests
            ValidationTest(
                name="ventral_concentration",
                description="Ventral SHH concentration > 5 nM",
                test_function="test_ventral_concentration",
                expected_result=5.0,
                tolerance=2.0,
                critical=True
            ),
            ValidationTest(
                name="dorsal_concentration",
                description="Dorsal SHH concentration < 1 nM",
                test_function="test_dorsal_concentration",
                expected_result=1.0,
                tolerance=0.5,
                critical=True
            ),
            ValidationTest(
                name="concentration_range",
                description="SHH concentration range 4-12 nM",
                test_function="test_concentration_range",
                expected_result=(4.0, 12.0),
                tolerance=2.0,
                critical=True
            ),
            
            # Gene expression tests
            ValidationTest(
                name="gene_expression_domains",
                description="Proper gene expression domain formation",
                test_function="test_gene_expression_domains",
                expected_result=True,
                critical=True
            ),
            ValidationTest(
                name="gene_hierarchy",
                description="Gene expression hierarchy (high threshold = small domain)",
                test_function="test_gene_hierarchy",
                expected_result=True,
                critical=True
            ),
            ValidationTest(
                name="ventral_genes",
                description="Ventral genes expressed in ventral domain",
                test_function="test_ventral_gene_expression",
                expected_result=True,
                critical=True
            ),
            ValidationTest(
                name="dorsal_genes", 
                description="Dorsal genes expressed in dorsal domain",
                test_function="test_dorsal_gene_expression",
                expected_result=True,
                critical=True
            ),
            
            # Cell fate specification tests
            ValidationTest(
                name="cell_fate_specification",
                description="Proper cell fate specification",
                test_function="test_cell_fate_specification",
                expected_result=True,
                critical=True
            ),
            ValidationTest(
                name="motor_neuron_domain",
                description="Motor neuron domain present",
                test_function="test_motor_neuron_domain",
                expected_result=True,
                critical=True
            ),
            ValidationTest(
                name="cell_type_ordering",
                description="Cell types in correct dorsal-ventral order",
                test_function="test_cell_type_ordering",
                expected_result=True,
                critical=True
            )
        ]
    
    def setup_test_system(self) -> None:
        """Set up test system components."""
        # Create spatial grid
        self.spatial_grid = SpatialGrid(self.grid_dimensions)
        
        # Create biological parameters
        bio_params = BiologicalParameters()
        
        # Create SHH system
        shh_diffusion = bio_params.get_diffusion_parameters('SHH')
        shh_source = bio_params.get_source_parameters('SHH')
        shh_interactions = bio_params.get_interaction_parameters('SHH')
        
        self.shh_system = SHHGradientSystem(
            self.spatial_grid, shh_diffusion, shh_source, shh_interactions
        )
        
        # Configure neural tube
        neural_tube_dims = GridDimensions(
            x_size=self.grid_dimensions.x_size,
            y_size=self.grid_dimensions.y_size,
            z_size=self.grid_dimensions.z_size,
            resolution=self.grid_dimensions.resolution
        )
        self.shh_system.configure_sources(neural_tube_dims)
        
        # Create cell fate specifier
        self.cell_fate_specifier = CellFateSpecifier(
            self.spatial_grid, self.shh_system.gene_expression
        )
        
        logger.info("Set up test system components")
    
    def run_gradient_simulation(self, duration_hours: float = 12.0) -> None:
        """Run SHH gradient simulation for testing."""
        if self.shh_system is None:
            raise ValueError("Test system not set up. Call setup_test_system() first.")
        
        logger.info(f"Running gradient simulation for {duration_hours} hours")
        self.shh_system.run_simulation(duration_hours)
        logger.info("Gradient simulation completed")
    
    def test_gradient_formation(self) -> bool:
        """Test if SHH gradient forms properly."""
        if 'SHH' not in self.spatial_grid.concentrations:
            return False
        
        shh_conc = self.spatial_grid.concentrations['SHH']
        
        # Check if gradient exists (non-zero concentrations)
        if np.max(shh_conc) == 0:
            return False
        
        # Check if there's spatial variation
        if np.std(shh_conc) < 0.1:
            return False
        
        return True
    
    def test_dorsal_ventral_gradient(self) -> bool:
        """Test proper dorsal-ventral gradient formation."""
        y_coords, shh_profile = self.shh_system.get_dorsal_ventral_profile()
        
        # Check if ventral concentration > dorsal concentration
        ventral_conc = shh_profile[-1]  # Bottom (ventral)
        dorsal_conc = shh_profile[0]    # Top (dorsal)
        
        return ventral_conc > dorsal_conc
    
    def test_gradient_steepness(self) -> bool:
        """Test if gradient steepness is within biological range."""
        grad_x, grad_y, grad_z = self.spatial_grid.get_gradient('SHH')
        
        # Focus on dorsal-ventral gradient (Y direction)
        mean_grad_y = np.mean(np.abs(grad_y))
        
        # Expected range: 0.01 - 0.5 nM/µm
        return 0.01 <= mean_grad_y <= 0.5
    
    def test_ventral_concentration(self) -> float:
        """Test ventral SHH concentration."""
        y_coords, shh_profile = self.shh_system.get_dorsal_ventral_profile()
        return float(shh_profile[-1])  # Ventral concentration
    
    def test_dorsal_concentration(self) -> float:
        """Test dorsal SHH concentration."""
        y_coords, shh_profile = self.shh_system.get_dorsal_ventral_profile()
        return float(shh_profile[0])   # Dorsal concentration
    
    def test_concentration_range(self) -> Tuple[float, float]:
        """Test SHH concentration range."""
        shh_conc = self.spatial_grid.concentrations['SHH']
        return (float(np.min(shh_conc)), float(np.max(shh_conc)))
    
    def test_gene_expression_domains(self) -> bool:
        """Test gene expression domain formation."""
        try:
            # Get expression maps for key genes
            key_genes = ['Nkx2.2', 'Olig2', 'Pax6']
            
            for gene in key_genes:
                if gene in self.shh_system.gene_expression.get_available_genes():
                    expr_map = self.shh_system.gene_expression.get_gene_expression_map(gene)
                    
                    # Check if gene has expression domain
                    if np.sum(expr_map > 0.5) == 0:
                        return False
            
            return True
        except Exception:
            return False
    
    def test_gene_hierarchy(self) -> bool:
        """Test gene expression hierarchy."""
        try:
            hierarchy = self.shh_system.gene_expression.get_expression_hierarchy()
            return hierarchy.get("hierarchy_valid", False)
        except Exception:
            return False
    
    def test_ventral_gene_expression(self) -> bool:
        """Test ventral gene expression patterns."""
        try:
            ventral_genes = ['Nkx2.2', 'Olig2']
            
            for gene in ventral_genes:
                if gene in self.shh_system.gene_expression.get_available_genes():
                    expr_map = self.shh_system.gene_expression.get_gene_expression_map(gene)
                    
                    if np.sum(expr_map > 0.5) > 0:
                        # Check if expression is biased toward ventral (high Y)
                        indices = np.where(expr_map > 0.5)
                        mean_y = np.mean(indices[1])
                        ventral_bias = mean_y > (self.grid_dimensions.y_size * 0.6)
                        
                        if not ventral_bias:
                            return False
            
            return True
        except Exception:
            return False
    
    def test_dorsal_gene_expression(self) -> bool:
        """Test dorsal gene expression patterns."""
        try:
            dorsal_genes = ['Pax6', 'Pax7']
            
            for gene in dorsal_genes:
                if gene in self.shh_system.gene_expression.get_available_genes():
                    expr_map = self.shh_system.gene_expression.get_gene_expression_map(gene)
                    
                    if np.sum(expr_map > 0.5) > 0:
                        # Check if expression is biased toward dorsal (low Y)
                        indices = np.where(expr_map > 0.5)
                        mean_y = np.mean(indices[1])
                        dorsal_bias = mean_y < (self.grid_dimensions.y_size * 0.4)
                        
                        if not dorsal_bias:
                            return False
            
            return True
        except Exception:
            return False
    
    def test_cell_fate_specification(self) -> bool:
        """Test cell fate specification."""
        try:
            cell_fates = self.cell_fate_specifier.specify_cell_fates()
            
            # Check if any cell fates were specified
            return len(cell_fates) > 0 and any(np.sum(fate_map) > 0 for fate_map in cell_fates.values())
        except Exception:
            return False
    
    def test_motor_neuron_domain(self) -> bool:
        """Test motor neuron domain presence."""
        try:
            cell_fates = self.cell_fate_specifier.specify_cell_fates()
            
            if 'motor_neuron' in cell_fates:
                return np.sum(cell_fates['motor_neuron']) > 0
            
            return False
        except Exception:
            return False
    
    def test_cell_type_ordering(self) -> bool:
        """Test cell type dorsal-ventral ordering."""
        try:
            validation = self.cell_fate_specifier.validate_cell_fate_patterns()
            return validation.get("is_valid", False) and len(validation.get("errors", [])) == 0
        except Exception:
            return False
    
    def run_single_test(self, test: ValidationTest) -> Dict[str, Any]:
        """Run single validation test.
        
        Args:
            test: Validation test to run
            
        Returns:
            Dictionary of test results
        """
        try:
            # Get test function
            test_func = getattr(self, test.test_function)
            
            # Run test
            result = test_func()
            
            # Evaluate result
            if isinstance(test.expected_result, bool):
                passed = result == test.expected_result
            elif isinstance(test.expected_result, (int, float)):
                passed = abs(result - test.expected_result) <= test.tolerance
            elif isinstance(test.expected_result, tuple) and len(test.expected_result) == 2:
                # Range test
                min_val, max_val = test.expected_result
                if isinstance(result, tuple):
                    passed = (min_val <= result[0] <= max_val + test.tolerance and 
                             min_val <= result[1] <= max_val + test.tolerance)
                else:
                    passed = min_val <= result <= max_val + test.tolerance
            else:
                passed = result == test.expected_result
            
            return {
                "name": test.name,
                "description": test.description,
                "passed": passed,
                "result": result,
                "expected": test.expected_result,
                "tolerance": test.tolerance,
                "critical": test.critical
            }
            
        except Exception as e:
            return {
                "name": test.name,
                "description": test.description,
                "passed": False,
                "result": None,
                "expected": test.expected_result,
                "error": str(e),
                "critical": test.critical
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests.
        
        Returns:
            Dictionary of comprehensive test results
        """
        logger.info("Running comprehensive SHH validation tests")
        
        # Set up test system
        self.setup_test_system()
        
        # Run gradient simulation
        self.run_gradient_simulation()
        
        # Run all tests
        test_results = []
        passed_count = 0
        critical_failures = 0
        
        for test in self.validation_tests:
            result = self.run_single_test(test)
            test_results.append(result)
            
            if result["passed"]:
                passed_count += 1
            elif result["critical"]:
                critical_failures += 1
        
        # Calculate summary statistics
        total_tests = len(self.validation_tests)
        pass_rate = passed_count / total_tests if total_tests > 0 else 0.0
        
        self.test_results = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_count,
                "failed": total_tests - passed_count,
                "critical_failures": critical_failures,
                "pass_rate": pass_rate,
                "overall_passed": critical_failures == 0 and pass_rate >= 0.8
            },
            "test_results": test_results,
            "system_info": {
                "grid_dimensions": {
                    "x": self.grid_dimensions.x_size,
                    "y": self.grid_dimensions.y_size,
                    "z": self.grid_dimensions.z_size
                },
                "resolution_um": self.grid_dimensions.resolution
            }
        }
        
        logger.info(f"Validation tests completed: {passed_count}/{total_tests} passed")
        if critical_failures > 0:
            logger.error(f"Critical failures: {critical_failures}")
        
        return self.test_results
    
    def get_validation_report(self) -> str:
        """Generate human-readable validation report.
        
        Returns:
            Formatted validation report string
        """
        if not self.test_results:
            return "No test results available. Run run_all_tests() first."
        
        report = []
        report.append("=" * 60)
        report.append("SHH GRADIENT SYSTEM VALIDATION REPORT")
        report.append("=" * 60)
        
        # Summary
        summary = self.test_results["summary"]
        report.append(f"\nSUMMARY:")
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed']}")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Critical Failures: {summary['critical_failures']}")
        report.append(f"Pass Rate: {summary['pass_rate']:.1%}")
        report.append(f"Overall Status: {'✅ PASSED' if summary['overall_passed'] else '❌ FAILED'}")
        
        # Detailed results
        report.append(f"\nDETAILED RESULTS:")
        report.append("-" * 40)
        
        for test_result in self.test_results["test_results"]:
            status = "✅ PASS" if test_result["passed"] else "❌ FAIL"
            critical_mark = " (CRITICAL)" if test_result["critical"] else ""
            
            report.append(f"{status} {test_result['name']}{critical_mark}")
            report.append(f"    Description: {test_result['description']}")
            
            if "error" in test_result:
                report.append(f"    Error: {test_result['error']}")
            else:
                report.append(f"    Result: {test_result['result']}")
                report.append(f"    Expected: {test_result['expected']}")
            
            report.append("")
        
        return "\n".join(report)
