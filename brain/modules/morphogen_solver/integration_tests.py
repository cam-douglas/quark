#!/usr/bin/env python3
"""SHH Integration Tests.

Complete integration tests for the SHH morphogen gradient system including
end-to-end workflows, system integration, and performance validation.

Integration: Integration testing for complete SHH system
Rationale: Ensures all components work together correctly
"""

from typing import Dict, Any, Optional, List
import numpy as np
import time
import logging

from .spatial_grid import SpatialGrid, GridDimensions
from .morphogen_solver import MorphogenSolver
from .shh_validation_tests import SHHValidationTests
from .biological_parameters import BiologicalParameters

logger = logging.getLogger(__name__)

class SHHIntegrationTests:
    """Complete SHH system integration tests.
    
    Tests end-to-end workflows, component integration, and system performance
    for the complete SHH morphogen gradient system.
    
    Key Features:
    - End-to-end workflow testing
    - Component integration validation
    - Performance benchmarking
    - System robustness testing
    """
    
    def __init__(self):
        """Initialize SHH integration tests."""
        self.test_results: Dict[str, Any] = {}
        
        # Test configurations
        self.test_grid_small = GridDimensions(x_size=50, y_size=40, z_size=30, resolution=2.0)
        self.test_grid_medium = GridDimensions(x_size=100, y_size=80, z_size=60, resolution=1.0)
        self.test_grid_large = GridDimensions(x_size=200, y_size=160, z_size=120, resolution=0.5)
        
        logger.info("Initialized SHH integration test suite")
    
    def test_complete_workflow(self, grid_dims: Optional[GridDimensions] = None) -> Dict[str, Any]:
        """Test complete SHH workflow from initialization to cell fate specification.
        
        Args:
            grid_dims: Grid dimensions for testing (uses medium if None)
            
        Returns:
            Dictionary of workflow test results
        """
        if grid_dims is None:
            grid_dims = self.test_grid_medium
        
        workflow_results = {
            "test_name": "complete_workflow",
            "grid_dimensions": {
                "x": grid_dims.x_size,
                "y": grid_dims.y_size,
                "z": grid_dims.z_size,
                "resolution": grid_dims.resolution
            },
            "steps": {},
            "overall_success": True,
            "execution_time_s": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Initialize morphogen solver
            logger.info("Step 1: Initializing morphogen solver")
            step_start = time.time()
            
            solver = MorphogenSolver(grid_dims)
            
            workflow_results["steps"]["1_initialization"] = {
                "success": True,
                "execution_time_s": time.time() - step_start,
                "details": "Morphogen solver initialized successfully"
            }
            
            # Step 2: Configure neural tube
            logger.info("Step 2: Configuring neural tube")
            step_start = time.time()
            
            solver.configure_neural_tube(
                neural_tube_length=grid_dims.x_size * grid_dims.resolution,
                neural_tube_height=grid_dims.y_size * grid_dims.resolution,
                neural_tube_width=grid_dims.z_size * grid_dims.resolution
            )
            
            workflow_results["steps"]["2_configuration"] = {
                "success": True,
                "execution_time_s": time.time() - step_start,
                "details": "Neural tube configured successfully"
            }
            
            # Step 3: Generate SHH gradient
            logger.info("Step 3: Generating SHH gradient")
            step_start = time.time()
            
            gradient_results = solver.generate_shh_gradient(
                simulation_hours=8.0  # Shorter simulation for testing
            )
            
            workflow_results["steps"]["3_gradient_generation"] = {
                "success": True,
                "execution_time_s": time.time() - step_start,
                "details": gradient_results["gradient_properties"],
                "time_steps": gradient_results.get("time_steps", 0)
            }
            
            # Step 4: Validate gradient
            logger.info("Step 4: Validating gradient formation")
            step_start = time.time()
            
            validation_results = solver.validate_gradients()
            
            workflow_results["steps"]["4_gradient_validation"] = {
                "success": validation_results.get("SHH", {}).get("concentration_stats", {}).get("max_concentration_nM", 0) > 1.0,
                "execution_time_s": time.time() - step_start,
                "details": validation_results
            }
            
            # Step 5: Test gene expression mapping
            logger.info("Step 5: Testing gene expression mapping")
            step_start = time.time()
            
            if 'SHH' in solver.morphogen_systems:
                shh_system = solver.morphogen_systems['SHH']
                gene_maps = shh_system.gene_expression.get_all_expression_maps('binary')
                
                workflow_results["steps"]["5_gene_expression"] = {
                    "success": len(gene_maps) > 0,
                    "execution_time_s": time.time() - step_start,
                    "details": {
                        "genes_mapped": len(gene_maps),
                        "available_genes": shh_system.gene_expression.get_available_genes()
                    }
                }
            else:
                workflow_results["steps"]["5_gene_expression"] = {
                    "success": False,
                    "execution_time_s": time.time() - step_start,
                    "error": "SHH system not found"
                }
            
            # Step 6: Test cell fate specification
            logger.info("Step 6: Testing cell fate specification")
            step_start = time.time()
            
            if 'SHH' in solver.morphogen_systems:
                from .cell_fate_specifier import CellFateSpecifier
                
                shh_system = solver.morphogen_systems['SHH']
                cell_fate_specifier = CellFateSpecifier(solver.spatial_grid, shh_system.gene_expression)
                
                cell_fates = cell_fate_specifier.specify_cell_fates()
                
                workflow_results["steps"]["6_cell_fate_specification"] = {
                    "success": len(cell_fates) > 0,
                    "execution_time_s": time.time() - step_start,
                    "details": {
                        "cell_types_specified": len(cell_fates),
                        "cell_types": list(cell_fates.keys()),
                        "total_cells": sum(int(np.sum(fate_map > 0.5)) for fate_map in cell_fates.values())
                    }
                }
            else:
                workflow_results["steps"]["6_cell_fate_specification"] = {
                    "success": False,
                    "execution_time_s": time.time() - step_start,
                    "error": "SHH system not found"
                }
        
        except Exception as e:
            logger.error(f"Workflow test failed: {e}")
            workflow_results["overall_success"] = False
            workflow_results["error"] = str(e)
        
        # Check if all steps succeeded
        for step_name, step_result in workflow_results["steps"].items():
            if not step_result.get("success", False):
                workflow_results["overall_success"] = False
                break
        
        workflow_results["execution_time_s"] = time.time() - start_time
        
        return workflow_results
    
    def test_system_performance(self) -> Dict[str, Any]:
        """Test system performance across different grid sizes.
        
        Returns:
            Dictionary of performance test results
        """
        performance_results = {
            "test_name": "system_performance",
            "grid_tests": {},
            "performance_summary": {}
        }
        
        test_grids = [
            ("small", self.test_grid_small),
            ("medium", self.test_grid_medium),
            ("large", self.test_grid_large)
        ]
        
        for grid_name, grid_dims in test_grids:
            logger.info(f"Performance test: {grid_name} grid")
            
            try:
                # Run workflow test for this grid size
                workflow_result = self.test_complete_workflow(grid_dims)
                
                performance_results["grid_tests"][grid_name] = {
                    "grid_dimensions": workflow_result["grid_dimensions"],
                    "total_voxels": grid_dims.x_size * grid_dims.y_size * grid_dims.z_size,
                    "execution_time_s": workflow_result["execution_time_s"],
                    "success": workflow_result["overall_success"],
                    "time_per_voxel_ms": (workflow_result["execution_time_s"] * 1000) / (grid_dims.x_size * grid_dims.y_size * grid_dims.z_size),
                    "step_timings": {
                        step_name: step_data.get("execution_time_s", 0)
                        for step_name, step_data in workflow_result["steps"].items()
                    }
                }
                
            except Exception as e:
                performance_results["grid_tests"][grid_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Calculate performance summary
        successful_tests = [test for test in performance_results["grid_tests"].values() if test.get("success", False)]
        
        if successful_tests:
            performance_results["performance_summary"] = {
                "successful_tests": len(successful_tests),
                "total_tests": len(test_grids),
                "avg_time_per_voxel_ms": sum(test["time_per_voxel_ms"] for test in successful_tests) / len(successful_tests),
                "fastest_grid": min(successful_tests, key=lambda x: x["time_per_voxel_ms"]),
                "slowest_grid": max(successful_tests, key=lambda x: x["time_per_voxel_ms"])
            }
        
        return performance_results
    
    def test_component_integration(self) -> Dict[str, Any]:
        """Test integration between system components.
        
        Returns:
            Dictionary of component integration test results
        """
        integration_results = {
            "test_name": "component_integration",
            "component_tests": {},
            "overall_success": True
        }
        
        try:
            # Initialize system
            solver = MorphogenSolver(self.test_grid_medium)
            solver.configure_neural_tube()
            
            # Test 1: Spatial grid integration
            logger.info("Testing spatial grid integration")
            
            grid_test = {
                "success": True,
                "details": {}
            }
            
            # Check if SHH was added to spatial grid
            if 'SHH' not in solver.spatial_grid.concentrations:
                grid_test["success"] = False
                grid_test["error"] = "SHH not added to spatial grid"
            else:
                grid_test["details"]["shh_grid_shape"] = solver.spatial_grid.concentrations['SHH'].shape
                grid_test["details"]["grid_info"] = solver.spatial_grid.get_grid_info()
            
            integration_results["component_tests"]["spatial_grid"] = grid_test
            
            # Test 2: Biological parameters integration
            logger.info("Testing biological parameters integration")
            
            bio_params_test = {
                "success": True,
                "details": {}
            }
            
            try:
                shh_params = solver.bio_params.get_diffusion_parameters('SHH')
                bio_params_test["details"]["shh_diffusion_coeff"] = shh_params.diffusion_coefficient
                bio_params_test["details"]["shh_degradation_rate"] = shh_params.degradation_rate
            except Exception as e:
                bio_params_test["success"] = False
                bio_params_test["error"] = str(e)
            
            integration_results["component_tests"]["biological_parameters"] = bio_params_test
            
            # Test 3: SHH system integration
            logger.info("Testing SHH system integration")
            
            shh_system_test = {
                "success": True,
                "details": {}
            }
            
            if 'SHH' in solver.morphogen_systems:
                shh_system = solver.morphogen_systems['SHH']
                
                # Test source manager
                source_stats = shh_system.source_manager.get_source_statistics()
                shh_system_test["details"]["source_configured"] = source_stats.get("configured", False)
                
                # Test gene expression system
                available_genes = shh_system.gene_expression.get_available_genes()
                shh_system_test["details"]["gene_count"] = len(available_genes)
                
            else:
                shh_system_test["success"] = False
                shh_system_test["error"] = "SHH system not found"
            
            integration_results["component_tests"]["shh_system"] = shh_system_test
            
            # Test 4: End-to-end data flow
            logger.info("Testing end-to-end data flow")
            
            data_flow_test = {
                "success": True,
                "details": {}
            }
            
            try:
                # Run short simulation
                gradient_results = solver.generate_shh_gradient(simulation_hours=2.0)
                
                # Check if concentration gradients were generated
                shh_concentrations = solver.get_shh_concentration_map()
                max_conc = np.max(shh_concentrations)
                
                data_flow_test["details"]["max_concentration"] = float(max_conc)
                data_flow_test["details"]["simulation_successful"] = max_conc > 0
                
                if max_conc <= 0:
                    data_flow_test["success"] = False
                    data_flow_test["error"] = "No SHH concentration generated"
                
            except Exception as e:
                data_flow_test["success"] = False
                data_flow_test["error"] = str(e)
            
            integration_results["component_tests"]["data_flow"] = data_flow_test
        
        except Exception as e:
            integration_results["overall_success"] = False
            integration_results["error"] = str(e)
        
        # Check overall success
        for test_name, test_result in integration_results["component_tests"].items():
            if not test_result.get("success", False):
                integration_results["overall_success"] = False
                break
        
        return integration_results
    
    def test_system_robustness(self) -> Dict[str, Any]:
        """Test system robustness with edge cases and error conditions.
        
        Returns:
            Dictionary of robustness test results
        """
        robustness_results = {
            "test_name": "system_robustness",
            "robustness_tests": {},
            "overall_success": True
        }
        
        # Test 1: Very small grid
        logger.info("Testing very small grid")
        try:
            tiny_grid = GridDimensions(x_size=5, y_size=5, z_size=5, resolution=5.0)
            solver = MorphogenSolver(tiny_grid)
            solver.configure_neural_tube()
            
            robustness_results["robustness_tests"]["tiny_grid"] = {
                "success": True,
                "details": "System handles very small grid"
            }
        except Exception as e:
            robustness_results["robustness_tests"]["tiny_grid"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 2: Zero simulation time
        logger.info("Testing zero simulation time")
        try:
            solver = MorphogenSolver(self.test_grid_small)
            solver.configure_neural_tube()
            result = solver.generate_shh_gradient(simulation_hours=0.0)
            
            robustness_results["robustness_tests"]["zero_time"] = {
                "success": True,
                "details": "System handles zero simulation time"
            }
        except Exception as e:
            robustness_results["robustness_tests"]["zero_time"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test 3: Missing configuration
        logger.info("Testing missing configuration")
        try:
            solver = MorphogenSolver(self.test_grid_small)
            # Don't configure neural tube
            result = solver.generate_shh_gradient(simulation_hours=1.0)
            
            # Should fail gracefully
            robustness_results["robustness_tests"]["missing_config"] = {
                "success": False,
                "details": "Should have failed but didn't"
            }
        except ValueError:
            robustness_results["robustness_tests"]["missing_config"] = {
                "success": True,
                "details": "Correctly failed with missing configuration"
            }
        except Exception as e:
            robustness_results["robustness_tests"]["missing_config"] = {
                "success": False,
                "error": f"Wrong exception type: {str(e)}"
            }
        
        # Check overall robustness
        successful_tests = sum(1 for test in robustness_results["robustness_tests"].values() 
                              if test.get("success", False))
        total_tests = len(robustness_results["robustness_tests"])
        
        robustness_results["overall_success"] = successful_tests >= (total_tests * 0.8)  # 80% success rate
        robustness_results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        return robustness_results
    
    def run_comprehensive_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration test suite.
        
        Returns:
            Dictionary of comprehensive test results
        """
        logger.info("Starting comprehensive SHH integration tests")
        
        comprehensive_results = {
            "test_suite": "comprehensive_integration",
            "start_time": time.time(),
            "test_results": {},
            "summary": {}
        }
        
        # Run all integration tests
        test_functions = [
            ("workflow", self.test_complete_workflow),
            ("performance", self.test_system_performance),
            ("component_integration", self.test_component_integration),
            ("robustness", self.test_system_robustness)
        ]
        
        for test_name, test_function in test_functions:
            logger.info(f"Running {test_name} tests")
            try:
                test_result = test_function()
                comprehensive_results["test_results"][test_name] = test_result
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                comprehensive_results["test_results"][test_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Calculate summary
        total_tests = len(test_functions)
        successful_tests = sum(1 for result in comprehensive_results["test_results"].values() 
                              if result.get("overall_success", result.get("success", False)))
        
        comprehensive_results["summary"] = {
            "total_test_categories": total_tests,
            "successful_categories": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "overall_success": successful_tests == total_tests,
            "execution_time_s": time.time() - comprehensive_results["start_time"]
        }
        
        logger.info(f"Integration tests completed: {successful_tests}/{total_tests} categories passed")
        
        return comprehensive_results
    
    def generate_integration_report(self, test_results: Dict[str, Any]) -> str:
        """Generate human-readable integration test report.
        
        Args:
            test_results: Results from run_comprehensive_integration_tests()
            
        Returns:
            Formatted integration test report
        """
        report = []
        report.append("=" * 70)
        report.append("SHH SYSTEM COMPREHENSIVE INTEGRATION TEST REPORT")
        report.append("=" * 70)
        
        # Summary
        summary = test_results.get("summary", {})
        report.append(f"\nSUMMARY:")
        report.append(f"Total Test Categories: {summary.get('total_test_categories', 0)}")
        report.append(f"Successful Categories: {summary.get('successful_categories', 0)}")
        report.append(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        report.append(f"Overall Status: {'✅ PASSED' if summary.get('overall_success', False) else '❌ FAILED'}")
        report.append(f"Total Execution Time: {summary.get('execution_time_s', 0):.1f} seconds")
        
        # Detailed results
        report.append(f"\nDETAILED RESULTS:")
        report.append("-" * 50)
        
        for test_category, test_result in test_results.get("test_results", {}).items():
            success = test_result.get("overall_success", test_result.get("success", False))
            status = "✅ PASS" if success else "❌ FAIL"
            
            report.append(f"\n{status} {test_category.upper()}")
            
            if "error" in test_result:
                report.append(f"    Error: {test_result['error']}")
            
            if "execution_time_s" in test_result:
                report.append(f"    Execution Time: {test_result['execution_time_s']:.2f}s")
            
            # Add specific details for each test type
            if test_category == "workflow" and "steps" in test_result:
                report.append("    Workflow Steps:")
                for step_name, step_data in test_result["steps"].items():
                    step_status = "✅" if step_data.get("success", False) else "❌"
                    report.append(f"      {step_status} {step_name}")
            
            elif test_category == "performance" and "performance_summary" in test_result:
                perf_summary = test_result["performance_summary"]
                if perf_summary:
                    report.append(f"    Avg Time per Voxel: {perf_summary.get('avg_time_per_voxel_ms', 0):.3f}ms")
            
            elif test_category == "component_integration" and "component_tests" in test_result:
                report.append("    Component Tests:")
                for comp_name, comp_result in test_result["component_tests"].items():
                    comp_status = "✅" if comp_result.get("success", False) else "❌"
                    report.append(f"      {comp_status} {comp_name}")
            
            elif test_category == "robustness" and "success_rate" in test_result:
                report.append(f"    Robustness Success Rate: {test_result['success_rate']:.1%}")
        
        return "\n".join(report)
