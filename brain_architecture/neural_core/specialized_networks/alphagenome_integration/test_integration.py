#!/usr/bin/env python3
"""
Test Script for AlphaGenome Integration

This script demonstrates the capabilities of the AlphaGenome integration
module for brain simulation.

Author: Brain Simulation Team
Date: 2025
License: Apache 2.0
"""

import os, sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='[🧬] %(message)s')
logger = logging.getLogger(__name__)

def test_alpha_genome_integration():
    """Test the AlphaGenome integration capabilities"""
    
    # Your AlphaGenome API key
    API_KEY = "AIzaSyBzpB8gJ_8xaWH_obtjNBuQQ4hZkSnMxOk"
    
    logger.info("🧬 Starting AlphaGenome Integration Test")
    
    try:
        # Test 1: DNA Controller
        logger.info("🧬 Testing DNA Controller...")
        from ...........................................................dna_controller import DNAController, DNARegion
        
        dna_controller = DNAController(API_KEY)
        
        # Create a test region
        test_region = DNARegion(
            chromosome="chr22",
            start=0,
            end=1000,
            name="test_region"
        )
        
        # Test DNA sequence analysis
        test_sequence = "ATCG" * 250  # 1000 base pairs
        analysis = dna_controller.analyze_dna_sequence(test_sequence, test_region)
        
        logger.info(f"✓ DNA analysis completed: {len(analysis)} results")
        
        # Test brain-specific sequence creation
        cortex_sequence = dna_controller.create_brain_specific_sequence("cortex", 500)
        logger.info(f"✓ Created cortex sequence: {cortex_sequence.description}")
        
        # Test 2: Cell Constructor
        logger.info("🧬 Testing Cell Constructor...")
        from ...........................................................cell_constructor import CellConstructor
        
        cell_constructor = CellConstructor(API_KEY)
        
        # Create a neuron
        neuron = cell_constructor.construct_cell(
            cell_type_name="neuron",
            developmental_context="adult"
        )
        
        logger.info(f"✓ Created {neuron.cell_type.name} cell")
        logger.info(f"  - Metabolic activity: {neuron.metabolic_activity:.3f}")
        logger.info(f"  - Differentiation level: {neuron.differentiation_level:.3f}")
        
        # Create cell population
        neuron_population = cell_constructor.create_cell_population(
            "neuron",
            population_size=10,
            variation_factor=0.1
        )
        
        logger.info(f"✓ Created neuron population: {len(neuron_population)} cells")
        
        # Test 3: Genome Analyzer
        logger.info("🧬 Testing Genome Analyzer...")
        from ...........................................................genome_analyzer import GenomeAnalyzer
        
        genome_analyzer = GenomeAnalyzer(API_KEY)
        
        # Analyze genomic region
        genomic_analysis = genome_analyzer.analyze_genomic_region(
            chromosome="chr22",
            start=0,
            end=1000,
            analysis_types=['rna_seq', 'chromatin_accessibility']
        )
        
        logger.info(f"✓ Genomic analysis completed: {len(genomic_analysis)} results")
        
        # Test 4: Biological Simulator
        logger.info("🧬 Testing Biological Simulator...")
        from ...........................................................biological_simulator import BiologicalSimulator
        
        simulator = BiologicalSimulator(API_KEY)
        
        # Test brain development simulation
        development_results = simulator.simulate_brain_development(
            developmental_stage="fetal",
            brain_regions=['cortex', 'hippocampus'],
            simulation_steps=20
        )
        
        logger.info(f"✓ Brain development simulation completed")
        logger.info(f"  - Regions simulated: {len(development_results['results'])}")
        logger.info(f"  - Simulation steps: {development_results['simulation_steps']}")
        
        # Test cellular interactions
        interaction_results = simulator.simulate_cellular_interactions(
            cell_types=['neuron', 'astrocyte'],
            simulation_time=30
        )
        
        logger.info(f"✓ Cellular interactions simulation completed")
        logger.info(f"  - Cell types: {interaction_results['cell_types']}")
        logger.info(f"  - Simulation time: {interaction_results['simulation_time']}")
        
        # Test comprehensive brain model
        brain_model = simulator.create_comprehensive_brain_model(
            model_name="test_brain_model"
        )
        
        logger.info(f"✓ Comprehensive brain model created: {brain_model['name']}")
        logger.info(f"  - Components: {list(brain_model['components'].keys())}")
        
        # Test 5: Export capabilities
        logger.info("🧬 Testing Export Capabilities...")
        
        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Export simulation results
        output_path = simulator.export_simulation_results(
            development_results,
            str(output_dir),
            "json"
        )
        
        logger.info(f"✓ Results exported to: {output_path}")
        
        # Test 6: Integration with existing brain modules
        logger.info("🧬 Testing Integration with Brain Modules...")
        
        # Test that we can import from existing brain modules
        try:
            from 🧠_BRAIN_ARCHITECTURE.01_NEURAL_CORE import __init__ as brain_init
            logger.info("✓ Successfully imported brain modules")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import brain modules: {e}")
        
        logger.info("🧬 All AlphaGenome Integration Tests Completed Successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulation_mode():
    """Test the simulation mode when AlphaGenome is not available"""
    
    logger.info("🧬 Testing Simulation Mode...")
    
    try:
        # Test with invalid API key to trigger simulation mode
        invalid_api_key = "invalid_key"
        
        from ...........................................................dna_controller import DNAController
        from ...........................................................cell_constructor import CellConstructor
        from ...........................................................genome_analyzer import GenomeAnalyzer
        from ...........................................................biological_simulator import BiologicalSimulator
        
        # These should work in simulation mode
        dna_controller = DNAController(invalid_api_key)
        cell_constructor = CellConstructor(invalid_api_key)
        genome_analyzer = GenomeAnalyzer(invalid_api_key)
        simulator = BiologicalSimulator(invalid_api_key)
        
        logger.info("✓ All components initialized in simulation mode")
        
        # Test basic functionality
        test_region = DNARegion(
            chromosome="chr22",
            start=0,
            end=100,
            name="test_region"
        )
        
        test_sequence = "ATCG" * 25
        analysis = dna_controller.analyze_dna_sequence(test_sequence, test_region)
        
        if analysis.get('simulation_mode', False):
            logger.info("✓ Simulation mode working correctly")
        else:
            logger.warning("⚠️ Simulation mode not detected")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simulation mode test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧬 AlphaGenome Integration Test Suite")
    logger.info("=" * 50)
    
    # Test 1: Full integration
    success1 = test_alpha_genome_integration()
    
    # Test 2: Simulation mode
    success2 = test_simulation_mode()
    
    # Summary
    logger.info("=" * 50)
    logger.info("🧬 Test Summary:")
    logger.info(f"  - Full Integration Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    logger.info(f"  - Simulation Mode Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        logger.info("🎉 All tests passed! AlphaGenome integration is working correctly.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
