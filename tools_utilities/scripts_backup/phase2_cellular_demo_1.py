#!/usr/bin/env python3
"""
Phase 2: Cellular and Tissue Modeling Demo
Fetal Brain Development - Cellular Processes and Morphogen Gradients

This demo showcases the cellular modeling capabilities implemented in Phase 2,
including morphogen gradients, cell population dynamics, and tissue patterning.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append('.')

from development.src.neurodata.fetal_cellular_modeling import FetalCellularModeler, CellPopulation
from development.src.neurodata.fetal_anatomical_simulation import FetalAnatomicalSimulator
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_phase2_cellular_modeling():
    """Demonstrate Phase 2 cellular modeling capabilities"""
    print("🧬 Phase 2: Cellular and Tissue Modeling Demo")
    print("=" * 60)
    
    # Initialize cellular modeler
    modeler = FetalCellularModeler()
    
    print("\n1. 📊 Morphogen System Overview")
    print("-" * 40)
    for name, info in modeler.morphogens.items():
        print(f"  • {info['name']} ({name.upper()})")
        print(f"    Function: {info['function']}")
        print(f"    Diffusion Rate: {info['diffusion_rate']}")
        print(f"    Decay Rate: {info['decay_rate']}")
        print(f"    Target Receptors: {', '.join(info['target_receptors'])}")
        print()
    
    print("\n2. 🧫 Cell Type Properties")
    print("-" * 40)
    for cell_type, props in modeler.cell_types.items():
        print(f"  • {cell_type.replace('_', ' ').title()}")
        print(f"    Proliferation Rate: {props['proliferation_rate']}")
        print(f"    Differentiation Rate: {props['differentiation_rate']}")
        print(f"    Migration Rate: {props['migration_rate']}")
        print(f"    Morphogen Sensitivity: {props['morphogen_sensitivity']}")
        print()
    
    print("\n3. 🧬 Gene Regulatory Network")
    print("-" * 40)
    for gene, network in modeler.gene_network.items():
        print(f"  • {gene.upper()}")
        print(f"    Activators: {', '.join(network['activators'])}")
        print(f"    Inhibitors: {', '.join(network['inhibitors'])}")
        print(f"    Targets: {', '.join(network['targets'])}")
        print(f"    Function: {network['function']}")
        print()
    
    return modeler

def demo_morphogen_gradients(modeler):
    """Demonstrate morphogen gradient creation and visualization"""
    print("\n4. 🌊 Morphogen Gradient Simulation")
    print("-" * 40)
    
    # Create sample volume
    volume_shape = (32, 32, 32)  # Smaller for demo
    
    # Create different morphogen gradients
    morphogen_gradients = []
    
    # SHH from ventral midline
    print("  Creating SHH gradient (ventral midline)...")
    shh_gradient = modeler.create_morphogen_gradient(
        'shh', volume_shape, 
        [(16, 16, 10), (16, 16, 15)]
    )
    morphogen_gradients.append(shh_gradient)
    
    # WNT from dorsal midline
    print("  Creating WNT gradient (dorsal midline)...")
    wnt_gradient = modeler.create_morphogen_gradient(
        'wnt', volume_shape,
        [(16, 16, 20), (16, 16, 25)]
    )
    morphogen_gradients.append(wnt_gradient)
    
    # FGF from anterior region
    print("  Creating FGF gradient (anterior region)...")
    fgf_gradient = modeler.create_morphogen_gradient(
        'fgf', volume_shape,
        [(25, 16, 16), (30, 16, 16)]
    )
    morphogen_gradients.append(fgf_gradient)
    
    print(f"  ✅ Created {len(morphogen_gradients)} morphogen gradients")
    
    # Show gradient statistics
    for gradient in morphogen_gradients:
        print(f"    {gradient.name.upper()}: max={np.max(gradient.concentration):.3f}, "
              f"mean={np.mean(gradient.concentration):.3f}")
    
    return morphogen_gradients

def demo_cell_populations(modeler):
    """Demonstrate cell population creation and properties"""
    print("\n5. 🧫 Cell Population Modeling")
    print("-" * 40)
    
    initial_populations = {}
    
    # Neural progenitor cells
    print("  Creating neural progenitor population...")
    progenitor_positions = np.random.uniform(10, 22, (50, 3))
    initial_populations['neural_progenitor'] = CellPopulation(
        cell_type='neural_progenitor',
        positions=progenitor_positions,
        properties={'size': 1.0, 'age': 0.0, 'metabolic_activity': 1.0},
        gene_expression={'pax6': 0.8, 'olig2': 0.2, 'neurogenin2': 0.1},
        morphogen_sensitivity={'shh': 0.8, 'wnt': 0.6, 'bmp': 0.4, 'fgf': 0.7}
    )
    
    # Radial glia cells
    print("  Creating radial glia population...")
    glia_positions = np.random.uniform(12, 20, (25, 3))
    initial_populations['radial_glia'] = CellPopulation(
        cell_type='radial_glia',
        positions=glia_positions,
        properties={'size': 1.2, 'age': 0.0, 'metabolic_activity': 1.1},
        gene_expression={'pax6': 0.9, 'olig2': 0.1, 'neurogenin2': 0.0},
        morphogen_sensitivity={'shh': 0.6, 'wnt': 0.8, 'bmp': 0.3, 'fgf': 0.9}
    )
    
    print(f"  ✅ Created {len(initial_populations)} cell populations")
    for cell_type, population in initial_populations.items():
        print(f"    {cell_type}: {len(population.positions)} cells")
    
    return initial_populations

def demo_cellular_simulation(modeler, morphogen_gradients, initial_populations):
    """Demonstrate cellular simulation and dynamics"""
    print("\n6. ⚡ Cellular Dynamics Simulation")
    print("-" * 40)
    
    print("  Running cellular simulation...")
    print("  (This may take a moment for realistic cell counts)")
    
    # Run simulation with fewer time steps for demo
    population_history = modeler.simulate_cell_population_dynamics(
        initial_populations, morphogen_gradients, time_steps=20
    )
    
    final_populations = population_history[-1]
    
    print("  ✅ Simulation complete!")
    print(f"    Time steps: {len(population_history)}")
    print(f"    Final populations:")
    
    total_cells = 0
    for cell_type, population in final_populations.items():
        cell_count = len(population.positions)
        total_cells += cell_count
        print(f"      {cell_type}: {cell_count} cells")
    
    print(f"    Total cells: {total_cells}")
    
    return population_history

def demo_phase_integration():
    """Demonstrate integration between Phase 1 and Phase 2"""
    print("\n7. 🔗 Phase 1 + Phase 2 Integration")
    print("-" * 40)
    
    print("  Initializing Phase 1 anatomical simulator...")
    anatomical_simulator = FetalAnatomicalSimulator()
    
    print("  Creating anatomical development timeline...")
    anatomical_simulator.create_development_timeline_visualization(
        output_path=Path("outputs/phase2_cellular/phase1_2_integration_timeline.png")
    )
    
    print("  ✅ Phase 1-2 integration complete!")
    print("    - Anatomical structures provide spatial context for cellular models")
    print("    - Cellular processes drive anatomical development")
    print("    - Multi-scale simulation pipeline established")
    
    return anatomical_simulator

def demo_outputs_and_visualization(modeler, morphogen_gradients, final_populations):
    """Demonstrate output generation and visualization"""
    print("\n8. 📊 Output Generation and Visualization")
    print("-" * 40)
    
    output_dir = Path("outputs/phase2_cellular")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("  Creating tissue patterning visualization...")
    modeler.create_tissue_patterning_visualization(
        morphogen_gradients, final_populations,
        output_path=output_dir / "demo_tissue_patterning.png"
    )
    
    print("  Generating cellular simulation report...")
    report = modeler.create_cellular_simulation_report(
        morphogen_gradients, final_populations,
        output_path=output_dir / "demo_cellular_report.json"
    )
    
    print("  ✅ Outputs generated successfully!")
    print(f"    Visualization: {output_dir / 'demo_tissue_patterning.png'}")
    print(f"    Report: {output_dir / 'demo_cellular_report.json'}")
    print(f"    Total cells: {report['cell_populations']['total_cells']}")
    print(f"    Morphogen gradients: {len(morphogen_gradients)}")
    
    return report

def main():
    """Main demo function"""
    print("🚀 SmallMind Fetal Brain Development Pipeline")
    print("   Phase 2: Cellular and Tissue Modeling Demo")
    print("=" * 70)
    
    try:
        # Demo Phase 2 cellular modeling
        modeler = demo_phase2_cellular_modeling()
        
        # Demo morphogen gradients
        morphogen_gradients = demo_morphogen_gradients(modeler)
        
        # Demo cell populations
        initial_populations = demo_cell_populations(modeler)
        
        # Demo cellular simulation
        population_history = demo_cellular_simulation(
            modeler, morphogen_gradients, initial_populations
        )
        
        # Demo phase integration
        anatomical_simulator = demo_phase_integration()
        
        # Demo outputs and visualization
        final_populations = population_history[-1]
        report = demo_outputs_and_visualization(
            modeler, morphogen_gradients, final_populations
        )
        
        # Summary
        print("\n🎉 Phase 2 Demo Complete!")
        print("=" * 40)
        print("✅ Morphogen gradient simulation")
        print("✅ Cell population modeling")
        print("✅ Cellular dynamics simulation")
        print("✅ Phase 1-2 integration")
        print("✅ Output generation and visualization")
        print("\n🚀 Ready for Phase 3: Neural Network Development!")
        
        return modeler, morphogen_gradients, population_history, report
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
