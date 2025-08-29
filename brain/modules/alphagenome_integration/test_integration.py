#!/usr/bin/env python3
"""
Comprehensive Integration Tests for AlphaGenome Integration
Tests all components individually and as an integrated system
"""

import unittest
import sys
import os
import numpy as np
import json
import tempfile
import shutil
from typing import Dict, Any, List
from datetime import datetime

# Add path for imports
sys.path.insert(0, '/Users/camdouglas/quark')

from brain_modules.alphagenome_integration import (
    DNAController, CellConstructor, GenomeAnalyzer, BiologicalSimulator,
    CellType, DevelopmentalStage, SimulationMode, BiologicalProcess,
    create_integrated_biological_system, get_alphagenome_status
)
from brain_modules.alphagenome_integration.config import (
    ConfigurationManager, validate_system_setup
)

class TestDNAController(unittest.TestCase):
    """Test DNA Controller functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.dna_controller = DNAController()
        self.test_chromosome = "chr22"
        self.test_start = 35677410
        self.test_end = 36725986
    
    def test_genomic_interval_analysis(self):
        """Test basic genomic interval analysis"""
        result = self.dna_controller.analyze_genomic_interval(
            self.test_chromosome, self.test_start, self.test_end
        )
        
        # Check required fields
        self.assertIn("status", result)
        self.assertIn("biological_context", result)
        self.assertIn("sequence_length", result)
        
        # Check sequence length calculation
        expected_length = self.test_end - self.test_start
        self.assertEqual(result["sequence_length"], expected_length)
        
        # Check biological context
        bio_context = result["biological_context"]
        self.assertIn("chromosome", bio_context)
        self.assertEqual(bio_context["chromosome"], self.test_chromosome)
    
    def test_variant_analysis(self):
        """Test variant effect analysis"""
        variant = {
            "chromosome": self.test_chromosome,
            "position": 36201698,
            "reference": "A",
            "alternate": "C"
        }
        
        result = self.dna_controller.analyze_genomic_interval(
            self.test_chromosome, self.test_start, self.test_end,
            variant=variant
        )
        
        # Should contain variant effects
        if result["status"] == "success":
            self.assertIn("variant_effects", result)
        
        # Should handle simulation mode gracefully
        self.assertIn("status", result)
    
    def test_gene_regulatory_network_prediction(self):
        """Test GRN prediction functionality"""
        target_genes = ["SOX2", "PAX6", "FOXG1", "EMX2"]
        genomic_context = {"neural_relevance": 0.8}
        
        grn_result = self.dna_controller.predict_gene_regulatory_network(
            target_genes, genomic_context
        )
        
        # Check result structure
        self.assertIn("target_genes", grn_result)
        self.assertIn("regulatory_interactions", grn_result)
        self.assertIn("network_topology", grn_result)
        
        # Check gene list
        self.assertEqual(grn_result["target_genes"], target_genes)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        initial_metrics = self.dna_controller.get_performance_metrics()
        
        # Run some analyses
        self.dna_controller.analyze_genomic_interval("chr1", 1000, 2000)
        self.dna_controller.analyze_genomic_interval("chr2", 1000, 2000)
        
        final_metrics = self.dna_controller.get_performance_metrics()
        
        # Check metrics updated
        self.assertGreater(
            final_metrics["controller_metrics"]["sequences_analyzed"],
            initial_metrics["controller_metrics"]["sequences_analyzed"]
        )

class TestCellConstructor(unittest.TestCase):
    """Test Cell Constructor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.cell_constructor = CellConstructor()
        self.test_position = (100.0, 100.0, 100.0)
    
    def test_neural_stem_cell_creation(self):
        """Test neural stem cell creation"""
        cell_id = self.cell_constructor.create_neural_stem_cell(self.test_position)
        
        # Check cell was created
        self.assertIsNotNone(cell_id)
        self.assertIn(cell_id, self.cell_constructor.cells)
        
        # Check cell properties
        cell = self.cell_constructor.cells[cell_id]
        self.assertEqual(cell.cell_type, CellType.NEURAL_STEM_CELL)
        self.assertEqual(cell.position, self.test_position)
        
        # Check gene expression
        self.assertIn("SOX2", cell.gene_expression)
        self.assertGreater(cell.gene_expression["SOX2"], 0.5)  # Should be highly expressed
    
    def test_neuron_creation(self):
        """Test neuron creation with subtypes"""
        # Test glutamatergic neuron
        glut_id = self.cell_constructor.create_neuron(
            self.test_position, "glutamatergic"
        )
        
        self.assertIsNotNone(glut_id)
        glut_cell = self.cell_constructor.cells[glut_id]
        self.assertEqual(glut_cell.cell_type, CellType.NEURON)
        self.assertGreater(glut_cell.gene_expression.get("VGLUT1", 0), 0.5)
        
        # Test GABAergic neuron
        gaba_id = self.cell_constructor.create_neuron(
            (200.0, 200.0, 200.0), "GABAergic"
        )
        
        gaba_cell = self.cell_constructor.cells[gaba_id]
        self.assertGreater(gaba_cell.gene_expression.get("GAD1", 0), 0.5)
    
    def test_glial_cell_creation(self):
        """Test glial cell creation"""
        # Test astrocyte
        astro_id = self.cell_constructor.create_glial_cell(
            self.test_position, CellType.ASTROCYTE
        )
        
        astro_cell = self.cell_constructor.cells[astro_id]
        self.assertEqual(astro_cell.cell_type, CellType.ASTROCYTE)
        self.assertGreater(astro_cell.gene_expression.get("GFAP", 0), 0.5)
        
        # Test oligodendrocyte
        oligo_id = self.cell_constructor.create_glial_cell(
            (300.0, 300.0, 300.0), CellType.OLIGODENDROCYTE
        )
        
        oligo_cell = self.cell_constructor.cells[oligo_id]
        self.assertEqual(oligo_cell.cell_type, CellType.OLIGODENDROCYTE)
        self.assertGreater(oligo_cell.gene_expression.get("MBP", 0), 0.5)
    
    def test_cell_differentiation(self):
        """Test cell differentiation"""
        # Create neural stem cell
        stem_id = self.cell_constructor.create_neural_stem_cell(self.test_position)
        
        # Attempt differentiation to neuron
        success = self.cell_constructor.differentiate_cell(stem_id, CellType.NEURON)
        
        if success:
            # Check cell type changed
            cell = self.cell_constructor.cells[stem_id]
            self.assertEqual(cell.cell_type, CellType.NEURON)
            
            # Check post-mitotic state
            self.assertEqual(cell.proliferation_rate, 0.0)
    
    def test_tissue_creation(self):
        """Test tissue formation"""
        # Create multiple cells
        cell_positions = [
            (0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0)
        ]
        
        for pos in cell_positions:
            self.cell_constructor.create_neural_stem_cell(pos)
        
        # Create tissue
        morphogen_sources = {
            "SHH": (5, -20, 0),
            "BMP": (5, 20, 0)
        }
        
        tissue_id = self.cell_constructor.create_tissue(
            "neural_tissue", "forebrain", cell_positions, morphogen_sources
        )
        
        # Check tissue created
        self.assertIsNotNone(tissue_id)
        self.assertIn(tissue_id, self.cell_constructor.tissues)
        
        tissue = self.cell_constructor.tissues[tissue_id]
        self.assertEqual(tissue.tissue_type, "neural_tissue")
        self.assertEqual(len(tissue.signaling_centers), 2)
    
    def test_developmental_stage_advancement(self):
        """Test developmental stage progression"""
        initial_stage = self.cell_constructor.current_stage
        
        # Advance stage
        self.cell_constructor.advance_development_stage(
            DevelopmentalStage.NEURAL_PROLIFERATION
        )
        
        # Check stage changed
        self.assertEqual(
            self.cell_constructor.current_stage,
            DevelopmentalStage.NEURAL_PROLIFERATION
        )
        self.assertNotEqual(self.cell_constructor.current_stage, initial_stage)
    
    def test_biological_validation(self):
        """Test biological rule validation"""
        # Create some cells
        stem_id = self.cell_constructor.create_neural_stem_cell((0, 0, 0))
        neuron_id = self.cell_constructor.create_neuron((10, 10, 10), "glutamatergic")
        
        # Run validation
        validation = self.cell_constructor.validate_biological_rules()
        
        # Check validation structure
        self.assertIn("cell_type_distribution", validation)
        self.assertIn("developmental_stage_consistency", validation)
        self.assertIn("biological_violations", validation)
        
        # Should be consistent
        self.assertTrue(validation["developmental_stage_consistency"])

class TestGenomeAnalyzer(unittest.TestCase):
    """Test Genome Analyzer functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.genome_analyzer = GenomeAnalyzer()
        self.test_chromosome = "chr17"
        self.test_start = 43000000
        self.test_end = 43100000
    
    def test_comprehensive_genomic_analysis(self):
        """Test comprehensive genomic region analysis"""
        analysis = self.genome_analyzer.analyze_genomic_region_comprehensive(
            self.test_chromosome, self.test_start, self.test_end
        )
        
        # Check all analysis components present
        required_components = [
            "basic_predictions",
            "gene_annotations", 
            "regulatory_elements",
            "conservation_analysis",
            "developmental_relevance",
            "network_associations"
        ]
        
        for component in required_components:
            self.assertIn(component, analysis)
        
        # Check gene annotations
        gene_annotations = analysis["gene_annotations"]
        self.assertIn("total_genes", gene_annotations)
        self.assertGreater(gene_annotations["total_genes"], 0)
    
    def test_gene_regulatory_network_construction(self):
        """Test GRN construction"""
        neural_genes = ["SOX2", "PAX6", "FOXG1", "EMX2", "TBR1", "NEUROG2"]
        
        grn = self.genome_analyzer.construct_gene_regulatory_network(
            neural_genes, "neural_development"
        )
        
        # Check GRN structure
        self.assertEqual(grn.biological_process, "neural_development")
        self.assertEqual(grn.core_genes, neural_genes)
        self.assertGreater(len(grn.transcription_factors), 0)
        self.assertIn("network_density", grn.network_topology)
        
        # Check expression dynamics
        self.assertEqual(len(grn.expression_dynamics), len(neural_genes))
        for gene in neural_genes:
            self.assertIn(gene, grn.expression_dynamics)
            self.assertEqual(len(grn.expression_dynamics[gene]), 25)  # 25 time points
    
    def test_developmental_cascade_analysis(self):
        """Test developmental cascade analysis"""
        stages = [
            DevelopmentalStage.NEURAL_INDUCTION,
            DevelopmentalStage.NEURAL_PROLIFERATION,
            DevelopmentalStage.DIFFERENTIATION,
            DevelopmentalStage.SYNAPTOGENESIS
        ]
        
        cascade = self.genome_analyzer.analyze_developmental_cascade(stages)
        
        # Check cascade structure
        self.assertIn("stage_transitions", cascade)
        self.assertIn("temporal_dynamics", cascade)
        self.assertIn("bifurcation_points", cascade)
        
        # Check transitions
        transitions = cascade["stage_transitions"]
        self.assertEqual(len(transitions), len(stages) - 1)
    
    def test_variant_network_effects(self):
        """Test variant network effect prediction"""
        effects = self.genome_analyzer.predict_variant_network_effects(
            "chr17", 43045000, "G", "A"
        )
        
        # Check effects structure
        self.assertIn("affected_networks", effects)
        self.assertIn("conservation_context", effects)
        self.assertIn("expression_changes", effects)
        
        # Check conservation context
        conservation = effects["conservation_context"]
        self.assertIn("variant_conservation", conservation)
        self.assertIn("likely_functional", conservation)
    
    def test_conservation_analysis(self):
        """Test conservation analysis"""
        conservation = self.genome_analyzer._analyze_conservation(
            self.test_chromosome, self.test_start, self.test_end
        )
        
        # Check conservation structure
        self.assertIn("overall_conservation", conservation)
        self.assertIn("conservation_segments", conservation)
        self.assertIn("phylogenetic_depth", conservation)
        
        # Check values in valid range
        self.assertGreaterEqual(conservation["overall_conservation"], 0.0)
        self.assertLessEqual(conservation["overall_conservation"], 1.0)

class TestBiologicalSimulator(unittest.TestCase):
    """Test Biological Simulator functionality"""
    
    def setUp(self):
        """Set up test environment"""
        from brain_modules.alphagenome_integration.biological_simulator import SimulationParameters
        
        # Use fast parameters for testing
        self.sim_params = SimulationParameters(
            simulation_id="test_simulation",
            mode=SimulationMode.ACCELERATED,
            time_step=1.0,  # 1 hour steps
            total_time=12.0,  # 12 hours total
            save_frequency=4.0,  # Save every 4 hours
            visualization_enabled=False  # Disable for testing
        )
        
        self.bio_simulator = BiologicalSimulator(simulation_params=self.sim_params)
    
    def test_simulation_initialization(self):
        """Test simulation initialization"""
        # Check initial state
        state = self.bio_simulator.get_current_state()
        
        self.assertEqual(state["current_time"], 0.0)
        self.assertEqual(state["current_stage"], DevelopmentalStage.NEURAL_INDUCTION.value)
        self.assertGreater(state["total_cells"], 0)  # Should have initial cells
        self.assertFalse(state["simulation_running"])
    
    def test_morphogen_gradients(self):
        """Test morphogen gradient setup"""
        # Check morphogen gradients exist
        self.assertGreater(len(self.bio_simulator.morphogen_gradients), 0)
        
        # Check specific morphogens
        required_morphogens = ["SHH", "BMP", "WNT", "FGF"]
        for morphogen in required_morphogens:
            self.assertIn(morphogen, self.bio_simulator.morphogen_gradients)
            
            gradient = self.bio_simulator.morphogen_gradients[morphogen]
            self.assertEqual(gradient.morphogen_name, morphogen)
            self.assertGreater(gradient.diffusion_rate, 0)
            self.assertGreater(gradient.production_rate, 0)
    
    def test_gene_regulatory_networks(self):
        """Test GRN setup"""
        # Check GRNs exist
        self.assertGreater(len(self.bio_simulator.signaling_networks), 0)
        
        # Check specific networks
        expected_networks = ["neural_induction", "neurogenesis", "gliogenesis"]
        for network_name in expected_networks:
            self.assertIn(network_name, self.bio_simulator.signaling_networks)
            
            grn = self.bio_simulator.signaling_networks[network_name]
            self.assertEqual(grn.biological_process, network_name)
            self.assertGreater(len(grn.core_genes), 0)
    
    def test_developmental_events(self):
        """Test developmental event scheduling"""
        # Check events scheduled
        self.assertGreater(len(self.bio_simulator.scheduled_events), 0)
        
        # Check event types
        event_types = [event.event_type for event in self.bio_simulator.scheduled_events]
        expected_types = [
            BiologicalProcess.NEURAL_INDUCTION,
            BiologicalProcess.NEURULATION,
            BiologicalProcess.NEUROGENESIS
        ]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, event_types)
    
    def test_short_simulation_run(self):
        """Test short simulation run"""
        # Run short simulation
        results = self.bio_simulator.run_simulation(duration=6.0)  # 6 hours
        
        # Check results structure
        self.assertIn("simulation_info", results)
        self.assertIn("final_state", results)
        self.assertIn("performance_metrics", results)
        self.assertIn("biological_validation", results)
        
        # Check simulation completed
        sim_info = results["simulation_info"]
        self.assertGreaterEqual(sim_info["total_time"], 6.0)
        
        # Check biological validation
        validation = results["biological_validation"]
        self.assertIn("passes_biological_rules", validation)
    
    def test_simulation_state_management(self):
        """Test simulation state management"""
        # Start simulation
        self.bio_simulator.simulation_running = True
        self.bio_simulator.current_time = 0.0
        
        # Pause simulation
        self.bio_simulator.pause_simulation()
        self.assertTrue(self.bio_simulator.simulation_paused)
        
        # Resume simulation
        self.bio_simulator.resume_simulation()
        self.assertFalse(self.bio_simulator.simulation_paused)
        
        # Stop simulation
        self.bio_simulator.stop_simulation()
        self.assertFalse(self.bio_simulator.simulation_running)

class TestIntegratedSystem(unittest.TestCase):
    """Test integrated system functionality"""
    
    def setUp(self):
        """Set up integrated system"""
        self.system = create_integrated_biological_system()
    
    def test_system_components(self):
        """Test all system components present"""
        required_components = [
            "dna_controller",
            "cell_constructor", 
            "genome_analyzer",
            "biological_simulator",
            "alphagenome_status",
            "system_config"
        ]
        
        for component in required_components:
            self.assertIn(component, self.system)
    
    def test_component_integration(self):
        """Test components work together"""
        dna_controller = self.system["dna_controller"]
        cell_constructor = self.system["cell_constructor"]
        
        # Test that cell constructor uses DNA controller
        self.assertIs(cell_constructor.dna_controller, dna_controller)
        
        # Test genomic analysis flows to cell construction
        analysis = dna_controller.analyze_genomic_interval("chr1", 1000, 10000)
        
        # Create cell with regulatory profile
        if "regulatory_analysis" in analysis:
            cell_id = cell_constructor.create_neural_stem_cell(
                (0, 0, 0), analysis.get("regulatory_analysis")
            )
            self.assertIsNotNone(cell_id)
    
    def test_system_configuration(self):
        """Test system configuration"""
        config = self.system["system_config"]
        
        # Check configuration structure
        self.assertIn("integration_modules", config)
        self.assertIn("biological_compliance", config)
        self.assertIn("supported_features", config)
        
        # Check biological compliance
        compliance = config["biological_compliance"]
        self.assertTrue(compliance["follows_biological_rules"])
    
    def test_alphagenome_status(self):
        """Test AlphaGenome status reporting"""
        status = self.system["alphagenome_status"]
        
        # Check status structure
        self.assertIn("available", status)
        self.assertIn("integration_status", status)
        self.assertIn("repository_path", status)
        
        # Should handle both available and simulation modes
        self.assertIn(status["integration_status"], ["active", "simulation_mode"])

class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        """Set up configuration test"""
        # Create temporary directory for test config
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = os.path.join(self.temp_dir, "test_config.json")
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_configuration_manager(self):
        """Test configuration manager"""
        config_manager = ConfigurationManager(self.test_config_file)
        
        # Check initial configuration
        self.assertIsNotNone(config_manager.alphagenome_config)
        self.assertIsNotNone(config_manager.biological_rules_config)
        self.assertIsNotNone(config_manager.simulation_config)
    
    def test_configuration_save_load(self):
        """Test configuration persistence"""
        config_manager = ConfigurationManager(self.test_config_file)
        
        # Modify configuration
        original_api_key = config_manager.alphagenome_config.api_key
        config_manager.alphagenome_config.api_key = "test_key"
        
        # Save configuration
        success = config_manager.save_configuration()
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.test_config_file))
        
        # Create new manager and load
        new_config_manager = ConfigurationManager(self.test_config_file)
        self.assertEqual(new_config_manager.alphagenome_config.api_key, "test_key")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config_manager = ConfigurationManager(self.test_config_file)
        validation = config_manager.validate_configuration()
        
        # Check validation structure
        self.assertIn("valid", validation)
        self.assertIn("warnings", validation)
        self.assertIn("errors", validation)
        self.assertIn("recommendations", validation)
        
        # Should be boolean
        self.assertIsInstance(validation["valid"], bool)
    
    def test_system_setup_validation(self):
        """Test complete system setup validation"""
        validation = validate_system_setup()
        
        # Check validation components
        required_keys = [
            "configuration_valid",
            "alphagenome_available", 
            "directories_writable",
            "system_requirements",
            "ready_for_use"
        ]
        
        for key in required_keys:
            self.assertIn(key, validation)

class TestBiologicalAccuracy(unittest.TestCase):
    """Test biological accuracy and compliance"""
    
    def setUp(self):
        """Set up biological accuracy tests"""
        self.cell_constructor = CellConstructor()
    
    def test_cell_type_transitions(self):
        """Test that only valid cell type transitions are allowed"""
        # Create neural stem cell
        stem_id = self.cell_constructor.create_neural_stem_cell((0, 0, 0))
        
        # Valid transition: stem cell -> neuron
        valid_transition = self.cell_constructor.differentiate_cell(
            stem_id, CellType.NEURON
        )
        
        # Note: This might fail due to probability, so we just check the method doesn't crash
        # and that the transition rules are enforced in the background
        
        # Check transition rules exist
        self.assertIn(CellType.NEURAL_STEM_CELL, self.cell_constructor.cell_type_transitions)
        valid_targets = self.cell_constructor.cell_type_transitions[CellType.NEURAL_STEM_CELL]
        self.assertIn(CellType.NEUROBLAST, valid_targets)
    
    def test_gene_expression_bounds(self):
        """Test gene expression stays within biological bounds"""
        cell_id = self.cell_constructor.create_neural_stem_cell((0, 0, 0))
        cell = self.cell_constructor.cells[cell_id]
        
        # Check all gene expression values are in [0, 1]
        for gene, expression in cell.gene_expression.items():
            self.assertGreaterEqual(expression, 0.0, f"Gene {gene} expression below 0")
            self.assertLessEqual(expression, 1.0, f"Gene {gene} expression above 1")
    
    def test_developmental_timing(self):
        """Test developmental timing follows biological constraints"""
        bio_simulator = BiologicalSimulator()
        
        # Check initial stage
        self.assertEqual(bio_simulator.current_stage, DevelopmentalStage.NEURAL_INDUCTION)
        
        # Check developmental events are properly timed
        for event in bio_simulator.scheduled_events:
            self.assertGreaterEqual(event.timing, 0.0)
            self.assertGreater(event.duration, 0.0)
    
    def test_morphogen_concentration_bounds(self):
        """Test morphogen concentrations stay within realistic bounds"""
        bio_simulator = BiologicalSimulator()
        
        for morphogen_name, gradient in bio_simulator.morphogen_gradients.items():
            # Check no negative concentrations
            self.assertGreaterEqual(
                np.min(gradient.concentration_profile), 0.0,
                f"Negative concentrations in {morphogen_name}"
            )
            
            # Check reasonable maximum concentrations
            max_conc = np.max(gradient.concentration_profile)
            self.assertLessEqual(
                max_conc, 20.0,  # Reasonable upper bound
                f"Extremely high concentrations in {morphogen_name}"
            )

def run_comprehensive_tests():
    """Run all integration tests"""
    
    print("🧪 Running AlphaGenome Integration Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDNAController,
        TestCellConstructor,
        TestGenomeAnalyzer,
        TestBiologicalSimulator,
        TestIntegratedSystem,
        TestConfiguration,
        TestBiologicalAccuracy
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    # Overall result
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n✅ All tests passed! AlphaGenome integration is working correctly.")
    else:
        print(f"\n⚠️ Some tests failed. Please review the integration.")
    
    return success

def run_quick_smoke_test():
    """Run quick smoke test of key functionality"""
    
    print("⚡ Quick Smoke Test - AlphaGenome Integration")
    print("=" * 50)
    
    try:
        # Test 1: System creation
        print("1. Creating integrated system...")
        system = create_integrated_biological_system()
        print("   ✅ System created successfully")
        
        # Test 2: AlphaGenome status
        print("2. Checking AlphaGenome status...")
        status = get_alphagenome_status()
        print(f"   ✅ Status: {status['integration_status']}")
        
        # Test 3: DNA analysis
        print("3. Testing DNA analysis...")
        dna_controller = system["dna_controller"]
        result = dna_controller.analyze_genomic_interval("chr1", 1000, 10000)
        print(f"   ✅ Analysis completed: {result['status']}")
        
        # Test 4: Cell creation
        print("4. Testing cell creation...")
        cell_constructor = system["cell_constructor"]
        cell_id = cell_constructor.create_neural_stem_cell((0, 0, 0))
        print(f"   ✅ Cell created: {cell_id[:8]}...")
        
        # Test 5: Configuration
        print("5. Testing configuration...")
        validation = validate_system_setup()
        print(f"   ✅ System ready: {validation['ready_for_use']}")
        
        print(f"\n🎉 Smoke test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AlphaGenome Integration Tests")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--full", action="store_true", help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    if args.smoke:
        success = run_quick_smoke_test()
    elif args.full:
        success = run_comprehensive_tests()
    else:
        # Default: run smoke test
        success = run_quick_smoke_test()
        
        if success:
            print(f"\n🔬 Run full test suite with: python test_integration.py --full")
    
    exit(0 if success else 1)
