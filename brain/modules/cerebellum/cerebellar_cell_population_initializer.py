#!/usr/bin/env python3
"""
Cerebellar Cell Population Initializer

Initializes all cerebellar cell populations based on lineage specifications
and morphogen gradient responses. Creates progenitor pools for Math1/Atoh1+
granule lineage, Ptf1a+ GABAergic lineage, Olig2+ glial lineage, and
neural stem cell populations.

Cell populations:
- Math1/Atoh1+ rhombic lip progenitors: 10⁶ cells at E10.5
- Ptf1a+ ventricular zone progenitors: 10⁵ cells for GABAergic lineages
- Olig2+ Bergmann glia precursors: 10⁴ cells
- Nestin+ neural stem cells: white matter zones
- Lhx1/5+ interneuron precursors: cerebellar VZ

Author: Quark Brain Development Team
Date: 2025-09-17
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# Import existing cell construction components
sys.path.insert(0, str(Path(__file__).parent.parent / "alphagenome_integration" / "cell_construction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "developmental_biology"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from cell_types import CellType, DevelopmentalStage, CellularParameters
    from neuroepithelial_cells import NeuroepithelialCell
except ImportError:
    logger.warning("Using simplified cell types without full integration")
    
    class CellType:
        NEURAL_PROGENITOR = "neural_progenitor"
        NEURON = "neuron"
        GLIAL_CELL = "glial_cell"
    
    class DevelopmentalStage:
        NEURAL_PROLIFERATION = "neural_proliferation"
        DIFFERENTIATION = "differentiation"
    
    @dataclass
    class CellularParameters:
        cell_id: str
        cell_type: str
        developmental_stage: str
        position: Tuple[float, float, float]
        diameter: float
        division_probability: float = 0.5
        molecular_markers: List[str] = None
        differentiation_potential: Dict[str, float] = None

@dataclass
class CellPopulationDefinition:
    """Definition of cerebellar cell population."""
    population_name: str
    progenitor_marker: str
    source_location: str
    target_cell_count: int
    spatial_distribution: Dict[str, Any]
    molecular_profile: List[str]
    division_rate: float  # divisions per hour
    differentiation_schedule: Dict[str, float]  # stage -> probability
    morphogen_responsiveness: Dict[str, Tuple[float, float]]  # morphogen -> (threshold, saturation)


class CerebellarCellPopulationInitializer:
    """Initializes all cerebellar cell populations."""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize cell population initializer."""
        self.data_dir = Path(data_dir)
        self.cell_pop_dir = self.data_dir / "cell_populations"
        self.populations_dir = self.cell_pop_dir / "population_definitions"
        self.initialization_dir = self.cell_pop_dir / "initialization_data"
        self.metadata_dir = self.cell_pop_dir / "metadata"
        
        for directory in [self.cell_pop_dir, self.populations_dir, self.initialization_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized cerebellar cell population initializer")
        logger.info(f"Data directory: {self.cell_pop_dir}")
    
    def define_math1_rhombic_lip_population(self) -> CellPopulationDefinition:
        """Define Math1/Atoh1+ rhombic lip progenitor population."""
        logger.info("Defining Math1/Atoh1+ rhombic lip progenitor population")
        
        math1_population = CellPopulationDefinition(
            population_name="Math1_Atoh1_rhombic_lip_progenitors",
            progenitor_marker="Atoh1",
            source_location="rhombic_lip",
            target_cell_count=1000000,  # 10⁶ cells
            spatial_distribution={
                "location_type": "rhombic_lip_territory",
                "coordinates": {
                    "anteroposterior_range": [0.44, 0.46],
                    "dorsoventral_position": 0.9,
                    "mediolateral_extent": [0.3, 0.7],
                    "thickness_um": 50.0
                },
                "distribution_pattern": "uniform_with_proliferative_zones",
                "density_per_mm3": 2000000  # 2M cells/mm³
            },
            molecular_profile=[
                "Atoh1", "Math1", "Pax6", "NeuroD1", "Zic1", "Gdf7", "Tbr2"
            ],
            division_rate=0.5,  # Every 2 hours
            differentiation_schedule={
                "E10.5": 0.1,  # 10% differentiate immediately
                "E11.5": 0.3,  # 30% differentiate
                "E12.5": 0.5,  # 50% differentiate
                "E13.5": 0.8,  # 80% differentiate
                "E14.5": 0.9   # 90% differentiate
            },
            morphogen_responsiveness={
                "FGF8": (2.5, 12.5),    # 50-250 ng/ml → 2.5-12.5 μM
                "SHH": (0.5, 5.0),      # 0.5-5 nM
                "BMP": (0.1, 1.0),      # Low BMP for dorsal specification
                "Reelin": (50.0, 200.0)  # ng/ml for migration guidance
            }
        )
        
        logger.info(f"Defined Math1+ population: {math1_population.target_cell_count:,} cells")
        return math1_population
    
    def define_ptf1a_ventricular_population(self) -> CellPopulationDefinition:
        """Define Ptf1a+ ventricular zone GABAergic progenitors."""
        logger.info("Defining Ptf1a+ ventricular zone progenitor population")
        
        ptf1a_population = CellPopulationDefinition(
            population_name="Ptf1a_ventricular_zone_GABAergic_progenitors",
            progenitor_marker="Ptf1a",
            source_location="ventricular_zone",
            target_cell_count=100000,  # 10⁵ cells
            spatial_distribution={
                "location_type": "ventricular_zone_territory",
                "coordinates": {
                    "anteroposterior_range": [0.42, 0.52],
                    "dorsoventral_position": 0.6,
                    "mediolateral_extent": [0.2, 0.8],
                    "thickness_um": 100.0
                },
                "distribution_pattern": "ventricular_surface_aligned",
                "density_per_mm3": 500000  # 500K cells/mm³
            },
            molecular_profile=[
                "Ptf1a", "Pax2", "Lhx1", "Lhx5", "Gad1", "Gad2", "Foxp2"
            ],
            division_rate=0.33,  # Every 3 hours
            differentiation_schedule={
                "E10.5": 0.05,  # 5% early differentiation
                "E11.5": 0.2,   # 20% differentiate
                "E12.5": 0.4,   # 40% differentiate
                "E13.5": 0.7,   # 70% differentiate
                "E14.5": 0.85   # 85% differentiate
            },
            morphogen_responsiveness={
                "FGF8": (1.0, 8.0),      # Lower sensitivity than Math1+
                "Wnt1": (0.5, 5.0),     # ng/ml → μM conversion
                "SHH": (0.1, 2.0),      # Low SHH for GABAergic fate
                "BMP_antagonist": (5.0, 50.0)  # ng/ml
            }
        )
        
        logger.info(f"Defined Ptf1a+ population: {ptf1a_population.target_cell_count:,} cells")
        return ptf1a_population
    
    def define_olig2_bergmann_glia_population(self) -> CellPopulationDefinition:
        """Define Olig2+ Bergmann glia precursor population."""
        logger.info("Defining Olig2+ Bergmann glia precursor population")
        
        olig2_population = CellPopulationDefinition(
            population_name="Olig2_Bergmann_glia_precursors",
            progenitor_marker="Olig2",
            source_location="ventricular_zone",
            target_cell_count=10000,  # 10⁴ cells
            spatial_distribution={
                "location_type": "ventricular_zone_glial_domain",
                "coordinates": {
                    "anteroposterior_range": [0.43, 0.51],
                    "dorsoventral_position": 0.65,
                    "mediolateral_extent": [0.25, 0.75],
                    "thickness_um": 75.0
                },
                "distribution_pattern": "scattered_progenitor_domains",
                "density_per_mm3": 100000  # 100K cells/mm³
            },
            molecular_profile=[
                "Olig2", "Sox9", "Nfib", "Gfap", "S100b", "Hopx"
            ],
            division_rate=0.25,  # Every 4 hours
            differentiation_schedule={
                "E11.5": 0.1,   # 10% early
                "E12.5": 0.3,   # 30%
                "E13.5": 0.6,   # 60%
                "E14.5": 0.8,   # 80%
                "E15.5": 0.95   # 95%
            },
            morphogen_responsiveness={
                "FGF8": (0.5, 4.0),      # Moderate FGF8 sensitivity
                "SHH": (0.2, 3.0),      # Moderate SHH for gliogenesis
                "BMP_antagonist": (2.0, 20.0),  # BMP antagonism promotes glia
                "Reelin": (10.0, 100.0)  # Reelin for radial guidance
            }
        )
        
        logger.info(f"Defined Olig2+ population: {olig2_population.target_cell_count:,} cells")
        return olig2_population
    
    def define_nestin_neural_stem_population(self) -> CellPopulationDefinition:
        """Define Nestin+ neural stem cell population."""
        logger.info("Defining Nestin+ neural stem cell population")
        
        nestin_population = CellPopulationDefinition(
            population_name="Nestin_neural_stem_cells",
            progenitor_marker="Nestin",
            source_location="prospective_white_matter",
            target_cell_count=5000,  # 5×10³ cells
            spatial_distribution={
                "location_type": "white_matter_zones",
                "coordinates": {
                    "anteroposterior_range": [0.42, 0.54],
                    "dorsoventral_position": 0.5,
                    "mediolateral_extent": [0.3, 0.7],
                    "thickness_um": 200.0
                },
                "distribution_pattern": "white_matter_tracts",
                "density_per_mm3": 50000  # 50K cells/mm³
            },
            molecular_profile=[
                "Nestin", "Sox2", "Pax6", "Hes1", "Id1"
            ],
            division_rate=0.2,  # Every 5 hours (slower)
            differentiation_schedule={
                "E12.5": 0.05,  # 5% early
                "E13.5": 0.15,  # 15%
                "E14.5": 0.3,   # 30%
                "E15.5": 0.5,   # 50%
                "P0": 0.7       # 70%
            },
            morphogen_responsiveness={
                "FGF8": (0.2, 2.0),      # Low FGF8 for stem maintenance
                "Wnt1": (0.1, 1.0),     # Low Wnt for proliferation
                "SHH": (0.05, 1.0),     # Very low SHH
                "BMP_antagonist": (1.0, 10.0)
            }
        )
        
        logger.info(f"Defined Nestin+ population: {nestin_population.target_cell_count:,} cells")
        return nestin_population
    
    def define_lhx_interneuron_population(self) -> CellPopulationDefinition:
        """Define Lhx1/5+ interneuron precursor population."""
        logger.info("Defining Lhx1/5+ interneuron precursor population")
        
        lhx_population = CellPopulationDefinition(
            population_name="Lhx1_5_interneuron_precursors",
            progenitor_marker="Lhx1_Lhx5",
            source_location="cerebellar_ventricular_zone",
            target_cell_count=25000,  # 2.5×10⁴ cells
            spatial_distribution={
                "location_type": "ventricular_zone_interneuron_domain",
                "coordinates": {
                    "anteroposterior_range": [0.43, 0.50],
                    "dorsoventral_position": 0.6,
                    "mediolateral_extent": [0.3, 0.7],
                    "thickness_um": 80.0
                },
                "distribution_pattern": "ventricular_patches",
                "density_per_mm3": 200000  # 200K cells/mm³
            },
            molecular_profile=[
                "Lhx1", "Lhx5", "Pax2", "Gad1", "Gad2", "Pvalb", "Sst"
            ],
            division_rate=0.4,  # Every 2.5 hours
            differentiation_schedule={
                "E11.0": 0.1,   # 10% early
                "E12.0": 0.3,   # 30%
                "E13.0": 0.6,   # 60%
                "E14.0": 0.8,   # 80%
                "P0": 0.95      # 95%
            },
            morphogen_responsiveness={
                "FGF8": (1.0, 6.0),      # Moderate FGF8 sensitivity
                "Wnt1": (0.5, 3.0),     # Wnt1 for proliferation
                "SHH": (0.1, 1.5),      # Low SHH for interneuron fate
                "BMP_antagonist": (3.0, 30.0)
            }
        )
        
        logger.info(f"Defined Lhx1/5+ population: {lhx_population.target_cell_count:,} cells")
        return lhx_population
    
    def create_population_initialization_map(self, populations: List[CellPopulationDefinition]) -> Dict[str, Any]:
        """Create 3D population initialization map."""
        logger.info("Creating 3D population initialization map")
        
        grid_size = (100, 100, 50)  # X (M-L), Y (A-P), Z (D-V)
        
        # Initialize population density maps
        population_maps = {}
        total_cells_map = np.zeros(grid_size)
        
        for population in populations:
            pop_map = np.zeros(grid_size)
            
            # Get spatial distribution
            coords = population.spatial_distribution["coordinates"]
            
            # Convert coordinates to grid indices
            ap_start = int(coords["anteroposterior_range"][0] * grid_size[1])
            ap_end = int(coords["anteroposterior_range"][1] * grid_size[1])
            dv_pos = int(coords["dorsoventral_position"] * grid_size[2])
            ml_start = int(coords["mediolateral_extent"][0] * grid_size[0])
            ml_end = int(coords["mediolateral_extent"][1] * grid_size[0])
            
            # Calculate thickness in voxels
            thickness_voxels = max(1, int(coords["thickness_um"] / 50.0))  # 50μm per voxel
            dv_start = max(0, dv_pos - thickness_voxels // 2)
            dv_end = min(grid_size[2], dv_pos + thickness_voxels // 2)
            
            # Fill population region
            region_volume = (ap_end - ap_start) * (ml_end - ml_start) * (dv_end - dv_start)
            if region_volume > 0:
                cell_density = population.target_cell_count / region_volume
                pop_map[ml_start:ml_end, ap_start:ap_end, dv_start:dv_end] = cell_density
            
            population_maps[population.population_name] = pop_map
            total_cells_map += pop_map
        
        # Save population maps
        for pop_name, pop_map in population_maps.items():
            map_file = self.initialization_dir / f"{pop_name}_density_map.npy"
            np.save(map_file, pop_map)
        
        total_map_file = self.initialization_dir / "total_cell_density_map.npy"
        np.save(total_map_file, total_cells_map)
        
        map_results = {
            "creation_date": datetime.now().isoformat(),
            "grid_dimensions": grid_size,
            "voxel_size_um": 50.0,
            "populations_mapped": len(populations),
            "total_target_cells": sum(p.target_cell_count for p in populations),
            "population_statistics": {
                pop.population_name: {
                    "target_cells": pop.target_cell_count,
                    "peak_density": float(np.max(population_maps[pop.population_name])),
                    "spatial_extent_voxels": int(np.sum(population_maps[pop.population_name] > 0))
                } for pop in populations
            },
            "density_map_files": {
                pop.population_name: str(self.initialization_dir / f"{pop.population_name}_density_map.npy")
                for pop in populations
            }
        }
        
        logger.info("Created 3D population initialization map")
        return map_results
    
    def execute_initialization(self) -> Dict[str, Any]:
        """Execute all cerebellar cell population initialization."""
        logger.info("Executing cerebellar cell population initialization")
        
        initialization_results = {
            "initialization_date": datetime.now().isoformat(),
            "batch_phase": "Phase_1_Batch_B_Steps_B2.1_to_B2.5",
            "populations_initialized": [],
            "total_cells_initialized": 0,
            "initialization_status": "completed"
        }
        
        # Define all cell populations
        logger.info("=== Defining All Cell Populations ===")
        populations = [
            self.define_math1_rhombic_lip_population(),
            self.define_ptf1a_ventricular_population(),
            self.define_olig2_bergmann_glia_population(),
            self.define_nestin_neural_stem_population(),
            self.define_lhx_interneuron_population()
        ]
        
        initialization_results["populations_initialized"] = [p.population_name for p in populations]
        initialization_results["total_cells_initialized"] = sum(p.target_cell_count for p in populations)
        
        # Create population initialization map
        logger.info("=== Creating Population Initialization Map ===")
        population_map = self.create_population_initialization_map(populations)
        initialization_results["population_map_details"] = population_map
        
        # Save population definitions
        populations_file = self.metadata_dir / "cell_population_definitions.json"
        populations_data = [
            {
                "population_name": pop.population_name,
                "progenitor_marker": pop.progenitor_marker,
                "source_location": pop.source_location,
                "target_cell_count": pop.target_cell_count,
                "spatial_distribution": pop.spatial_distribution,
                "molecular_profile": pop.molecular_profile,
                "division_rate": pop.division_rate,
                "differentiation_schedule": pop.differentiation_schedule,
                "morphogen_responsiveness": pop.morphogen_responsiveness
            } for pop in populations
        ]
        
        with open(populations_file, 'w') as f:
            json.dump(populations_data, f, indent=2)
        
        # Save complete results
        results_file = self.metadata_dir / "cell_population_initialization_results.json"
        with open(results_file, 'w') as f:
            json.dump(initialization_results, f, indent=2)
        
        logger.info(f"Cell population initialization completed. Results saved to {results_file}")
        return initialization_results


def main():
    """Execute cerebellar cell population initialization."""
    
    print("🧬 CEREBELLAR CELL POPULATION INITIALIZATION")
    print("=" * 60)
    print("Phase 1 ▸ Batch B ▸ Steps B2.1-B2.5")
    print("Math1+ + Ptf1a+ + Olig2+ + Nestin+ + Lhx1/5+ Populations")
    print()
    
    # Initialize cell population initializer
    initializer = CerebellarCellPopulationInitializer()
    
    # Execute initialization
    results = initializer.execute_initialization()
    
    # Print initialization summary
    print(f"✅ Cell population initialization completed")
    print(f"🧬 Populations initialized: {len(results['populations_initialized'])}")
    print(f"📊 Total cells initialized: {results['total_cells_initialized']:,}")
    print(f"🎯 Initialization status: {results['initialization_status']}")
    print()
    
    # Display population details
    print("📥 Cell Populations Initialized:")
    if 'population_map_details' in results:
        pop_stats = results['population_map_details']['population_statistics']
        for pop_name, stats in pop_stats.items():
            print(f"  • {pop_name.replace('_', ' ')}")
            print(f"    Target cells: {stats['target_cells']:,}")
            print(f"    Peak density: {stats['peak_density']:.0f} cells/voxel")
            print(f"    Spatial extent: {stats['spatial_extent_voxels']} voxels")
            print()
    
    print("📁 Data Locations:")
    print(f"  • Population definitions: {initializer.populations_dir}")
    print(f"  • Initialization data: {initializer.initialization_dir}")
    print(f"  • Metadata: {initializer.metadata_dir}")
    
    print()
    print("🎯 Next Steps:")
    print("- Validate cell population densities against literature")
    print("- Integrate populations with morphogen gradient responses")
    print("- Proceed to B3: Structural Scaffold Construction")
    print("- Begin B4: Migration Path Implementation")


if __name__ == "__main__":
    main()
