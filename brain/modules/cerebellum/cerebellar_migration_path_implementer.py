#!/usr/bin/env python3
"""
Cerebellar Migration Path Implementer

Implements all migration pathways for cerebellar development including
tangential migration of granule precursors, radial migration along Bergmann
glia, Purkinje cell nuclear transposition, stellate/basket cell migration,
and deep nuclei neuron migration.

Migration pathways:
- Tangential migration: granule precursors along pial surface (30 μm/hour)
- Radial migration: granule cells along Bergmann glia (20 μm/hour)
- Nuclear transposition: Purkinje cells (inside-out, 15 μm/hour)
- Tangential molecular layer: stellate/basket cells
- Deep nuclei radial: early neurons from VZ

Author: Quark Brain Development Team
Date: 2025-09-17
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MigrationPathway:
    """Definition of cellular migration pathway."""
    pathway_name: str
    cell_type: str
    migration_type: str  # "tangential", "radial", "nuclear_transposition"
    origin_coordinates: Tuple[float, float, float]
    destination_coordinates: Tuple[float, float, float]
    migration_speed_um_hour: float
    guidance_molecules: List[str]
    migration_duration_hours: float
    cell_count_per_pathway: int
    developmental_window: str


@dataclass
class BergmannGliaScaffold:
    """Definition of Bergmann glia radial scaffold."""
    scaffold_id: int
    origin_position: Tuple[float, float, float]  # VZ position
    target_position: Tuple[float, float, float]  # Pial surface
    fiber_length_um: float
    guidance_capacity: int  # Cells that can migrate per fiber
    molecular_markers: List[str]


class CerebellarMigrationPathImplementer:
    """Implements all cerebellar migration pathways."""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize migration path implementer."""
        self.data_dir = Path(data_dir)
        self.migration_dir = self.data_dir / "migration_paths"
        self.pathways_dir = self.migration_dir / "pathway_definitions"
        self.scaffolds_dir = self.migration_dir / "bergmann_scaffolds"
        self.maps_dir = self.migration_dir / "migration_maps"
        self.metadata_dir = self.migration_dir / "metadata"
        
        for directory in [self.migration_dir, self.pathways_dir, self.scaffolds_dir, self.maps_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized migration path implementer")
        logger.info(f"Data directory: {self.migration_dir}")
    
    def implement_tangential_migration_routes(self) -> List[MigrationPathway]:
        """Implement tangential migration routes for granule precursors."""
        logger.info("Implementing tangential migration routes for granule precursors")
        
        tangential_pathways = [
            MigrationPathway(
                pathway_name="rhombic_lip_to_rostral_EGL",
                cell_type="granule_cell_precursor",
                migration_type="tangential",
                origin_coordinates=(0.5, 0.45, 0.9),  # Rhombic lip
                destination_coordinates=(0.3, 0.35, 0.88),  # Rostral EGL
                migration_speed_um_hour=30.0,
                guidance_molecules=["Netrin1", "Slit2", "Ephrin_B2", "Semaphorin3A"],
                migration_duration_hours=18.0,
                cell_count_per_pathway=200000,  # 200K cells per stream
                developmental_window="E11.0-E13.5"
            ),
            MigrationPathway(
                pathway_name="rhombic_lip_to_caudal_EGL",
                cell_type="granule_cell_precursor",
                migration_type="tangential",
                origin_coordinates=(0.5, 0.45, 0.9),  # Rhombic lip
                destination_coordinates=(0.7, 0.55, 0.88),  # Caudal EGL
                migration_speed_um_hour=30.0,
                guidance_molecules=["Netrin1", "Slit2", "Ephrin_B2", "Semaphorin3A"],
                migration_duration_hours=18.0,
                cell_count_per_pathway=200000,
                developmental_window="E11.0-E13.5"
            ),
            MigrationPathway(
                pathway_name="EGL_tangential_spreading",
                cell_type="granule_cell_precursor",
                migration_type="tangential",
                origin_coordinates=(0.5, 0.45, 0.88),  # Central EGL
                destination_coordinates=(0.8, 0.45, 0.88),  # Lateral EGL
                migration_speed_um_hour=25.0,
                guidance_molecules=["Ephrin_A5", "EphA4", "Wnt3"],
                migration_duration_hours=24.0,
                cell_count_per_pathway=300000,
                developmental_window="E12.0-E15.0"
            )
        ]
        
        logger.info(f"Implemented {len(tangential_pathways)} tangential migration routes")
        return tangential_pathways
    
    def model_radial_migration_bergmann_glia(self) -> Tuple[List[MigrationPathway], List[BergmannGliaScaffold]]:
        """Model radial migration along Bergmann glial fibers."""
        logger.info("Modeling radial migration along Bergmann glial fibers")
        
        # Create Bergmann glia scaffold
        bergmann_scaffolds = []
        radial_pathways = []
        
        # Generate Bergmann glia fibers across cerebellar surface
        ml_positions = np.linspace(0.1, 0.9, 40)  # 40 radial positions
        ap_positions = np.linspace(0.35, 0.55, 30)  # 30 A-P positions
        
        scaffold_id = 1
        pathway_id = 1
        
        for ml_pos in ml_positions:
            for ap_pos in ap_positions:
                # Bergmann glia fiber from VZ to pial surface
                vz_position = (ml_pos, ap_pos, 0.6)   # Ventricular zone
                pial_position = (ml_pos, ap_pos, 0.9)  # Pial surface
                
                fiber_length = np.sqrt(sum((p - v)**2 for p, v in zip(pial_position, vz_position))) * 2500  # Convert to μm
                
                scaffold = BergmannGliaScaffold(
                    scaffold_id=scaffold_id,
                    origin_position=vz_position,
                    target_position=pial_position,
                    fiber_length_um=fiber_length,
                    guidance_capacity=50,  # 50 granule cells per fiber
                    molecular_markers=["Gfap", "S100b", "Aqp4", "Aldh1l1"]
                )
                bergmann_scaffolds.append(scaffold)
                
                # Corresponding radial migration pathway
                pathway = MigrationPathway(
                    pathway_name=f"radial_migration_fiber_{scaffold_id}",
                    cell_type="granule_cell",
                    migration_type="radial",
                    origin_coordinates=(ml_pos, ap_pos, 0.88),  # EGL position
                    destination_coordinates=(ml_pos, ap_pos, 0.4),   # IGL position
                    migration_speed_um_hour=20.0,
                    guidance_molecules=["Reelin", "Bergmann_glia_contact", "BDNF", "Neuregulin"],
                    migration_duration_hours=24.0,
                    cell_count_per_pathway=50,
                    developmental_window="E14.5-P21"
                )
                radial_pathways.append(pathway)
                
                scaffold_id += 1
                pathway_id += 1
        
        logger.info(f"Modeled {len(bergmann_scaffolds)} Bergmann glia scaffolds")
        logger.info(f"Created {len(radial_pathways)} radial migration pathways")
        return radial_pathways, bergmann_scaffolds
    
    def create_purkinje_transposition_patterns(self) -> List[MigrationPathway]:
        """Create nuclear transposition patterns for Purkinje cells."""
        logger.info("Creating Purkinje cell nuclear transposition patterns")
        
        transposition_pathways = [
            MigrationPathway(
                pathway_name="purkinje_nuclear_transposition",
                cell_type="Purkinje_cell",
                migration_type="nuclear_transposition",
                origin_coordinates=(0.5, 0.45, 0.6),   # VZ position
                destination_coordinates=(0.5, 0.45, 0.65),  # Final Purkinje layer
                migration_speed_um_hour=15.0,
                guidance_molecules=["Reelin", "Dab1", "ApoER2", "VLDLR"],
                migration_duration_hours=36.0,
                cell_count_per_pathway=19200,  # All Purkinje cells
                developmental_window="E11.5-E15.5"
            )
        ]
        
        logger.info(f"Created {len(transposition_pathways)} Purkinje transposition patterns")
        return transposition_pathways
    
    def define_stellate_basket_migration(self) -> List[MigrationPathway]:
        """Define stellate and basket cell tangential migration."""
        logger.info("Defining stellate/basket cell tangential migration")
        
        interneuron_pathways = [
            MigrationPathway(
                pathway_name="basket_cell_tangential_migration",
                cell_type="basket_cell",
                migration_type="tangential",
                origin_coordinates=(0.4, 0.45, 0.6),   # VZ source
                destination_coordinates=(0.4, 0.45, 0.75),  # Lower molecular layer
                migration_speed_um_hour=25.0,
                guidance_molecules=["Ephrin_A5", "EphA4", "Sema3F", "PlexinA3"],
                migration_duration_hours=30.0,
                cell_count_per_pathway=5000,
                developmental_window="E12.5-P7"
            ),
            MigrationPathway(
                pathway_name="stellate_cell_tangential_migration",
                cell_type="stellate_cell",
                migration_type="tangential",
                origin_coordinates=(0.6, 0.45, 0.6),   # VZ source
                destination_coordinates=(0.6, 0.45, 0.8),   # Upper molecular layer
                migration_speed_um_hour=22.0,
                guidance_molecules=["Ephrin_A5", "EphA4", "Sema3F", "PlexinA3"],
                migration_duration_hours=32.0,
                cell_count_per_pathway=3000,
                developmental_window="E13.0-P14"
            )
        ]
        
        logger.info(f"Defined {len(interneuron_pathways)} stellate/basket migration pathways")
        return interneuron_pathways
    
    def implement_deep_nuclei_migration(self) -> List[MigrationPathway]:
        """Implement deep nuclei neuron radial migration."""
        logger.info("Implementing deep nuclei neuron radial migration")
        
        deep_nuclei_pathways = [
            MigrationPathway(
                pathway_name="fastigial_neuron_migration",
                cell_type="fastigial_neuron",
                migration_type="radial",
                origin_coordinates=(0.5, 0.42, 0.6),   # Medial VZ
                destination_coordinates=(0.5, 0.43, 0.3),   # Fastigial nucleus
                migration_speed_um_hour=18.0,
                guidance_molecules=["Netrin1", "DCC", "Slit1", "Robo2"],
                migration_duration_hours=40.0,
                cell_count_per_pathway=50000,
                developmental_window="E10.5-E13.5"
            ),
            MigrationPathway(
                pathway_name="interposed_neuron_migration",
                cell_type="interposed_neuron",
                migration_type="radial",
                origin_coordinates=(0.65, 0.43, 0.6),  # Intermediate VZ
                destination_coordinates=(0.65, 0.44, 0.3),  # Interposed nucleus
                migration_speed_um_hour=18.0,
                guidance_molecules=["Netrin1", "DCC", "Slit1", "Robo2"],
                migration_duration_hours=40.0,
                cell_count_per_pathway=35000,
                developmental_window="E10.5-E13.5"
            ),
            MigrationPathway(
                pathway_name="dentate_neuron_migration",
                cell_type="dentate_neuron",
                migration_type="radial",
                origin_coordinates=(0.8, 0.44, 0.6),   # Lateral VZ
                destination_coordinates=(0.8, 0.45, 0.3),   # Dentate nucleus
                migration_speed_um_hour=18.0,
                guidance_molecules=["Netrin1", "DCC", "Slit1", "Robo2"],
                migration_duration_hours=40.0,
                cell_count_per_pathway=100000,
                developmental_window="E10.5-E13.5"
            )
        ]
        
        logger.info(f"Implemented {len(deep_nuclei_pathways)} deep nuclei migration pathways")
        return deep_nuclei_pathways
    
    def create_integrated_migration_map(self, all_pathways: List[MigrationPathway], 
                                      bergmann_scaffolds: List[BergmannGliaScaffold]) -> Dict[str, Any]:
        """Create integrated 3D migration map."""
        logger.info("Creating integrated 3D migration map")
        
        grid_size = (100, 100, 50)  # X (M-L), Y (A-P), Z (D-V)
        
        # Initialize migration maps
        tangential_map = np.zeros(grid_size)
        radial_map = np.zeros(grid_size)
        transposition_map = np.zeros(grid_size)
        migration_density_map = np.zeros(grid_size)
        
        for pathway in all_pathways:
            # Convert coordinates to grid indices
            origin_idx = tuple(int(coord * dim) for coord, dim in zip(pathway.origin_coordinates, grid_size))
            dest_idx = tuple(int(coord * dim) for coord, dim in zip(pathway.destination_coordinates, grid_size))
            
            # Create migration route
            if pathway.migration_type == "tangential":
                # Tangential migration along surface or within layer
                if "EGL" in pathway.pathway_name or "pial" in pathway.pathway_name:
                    # Surface tangential migration
                    y_path = np.linspace(origin_idx[1], dest_idx[1], 20)
                    z_surface = max(origin_idx[2], dest_idx[2])
                    
                    for y in y_path.astype(int):
                        if 0 <= y < grid_size[1]:
                            tangential_map[:, y, z_surface] = 1.0
                            migration_density_map[:, y, z_surface] += pathway.cell_count_per_pathway / 20
                
            elif pathway.migration_type == "radial":
                # Radial migration inward
                z_path = np.linspace(origin_idx[2], dest_idx[2], 25)
                x_pos, y_pos = origin_idx[0], origin_idx[1]
                
                for z in z_path.astype(int):
                    if 0 <= z < grid_size[2] and 0 <= x_pos < grid_size[0] and 0 <= y_pos < grid_size[1]:
                        radial_map[x_pos, y_pos, z] = 2.0
                        migration_density_map[x_pos, y_pos, z] += pathway.cell_count_per_pathway / 25
                        
            elif pathway.migration_type == "nuclear_transposition":
                # Nuclear transposition (short distance)
                z_path = np.linspace(origin_idx[2], dest_idx[2], 5)
                x_pos, y_pos = origin_idx[0], origin_idx[1]
                
                for z in z_path.astype(int):
                    if 0 <= z < grid_size[2]:
                        transposition_map[x_pos, y_pos, z] = 3.0
                        migration_density_map[x_pos, y_pos, z] += pathway.cell_count_per_pathway / 5
        
        # Map Bergmann glia scaffolds
        bergmann_map = np.zeros(grid_size)
        for scaffold in bergmann_scaffolds:
            origin_idx = tuple(int(coord * dim) for coord, dim in zip(scaffold.origin_position, grid_size))
            target_idx = tuple(int(coord * dim) for coord, dim in zip(scaffold.target_position, grid_size))
            
            # Create radial fiber
            z_path = np.linspace(origin_idx[2], target_idx[2], 15)
            x_pos, y_pos = origin_idx[0], origin_idx[1]
            
            for z in z_path.astype(int):
                if 0 <= z < grid_size[2] and 0 <= x_pos < grid_size[0] and 0 <= y_pos < grid_size[1]:
                    bergmann_map[x_pos, y_pos, z] = scaffold.scaffold_id
        
        # Save migration maps
        tangential_file = self.maps_dir / "tangential_migration_map.npy"
        radial_file = self.maps_dir / "radial_migration_map.npy"
        transposition_file = self.maps_dir / "nuclear_transposition_map.npy"
        density_file = self.maps_dir / "migration_density_map.npy"
        bergmann_file = self.maps_dir / "bergmann_glia_scaffold_map.npy"
        
        np.save(tangential_file, tangential_map)
        np.save(radial_file, radial_map)
        np.save(transposition_file, transposition_map)
        np.save(density_file, migration_density_map)
        np.save(bergmann_file, bergmann_map)
        
        map_results = {
            "creation_date": datetime.now().isoformat(),
            "grid_dimensions": grid_size,
            "voxel_size_um": 50.0,
            "migration_statistics": {
                "total_pathways": len(all_pathways),
                "bergmann_scaffolds": len(bergmann_scaffolds),
                "tangential_pathways": len([p for p in all_pathways if p.migration_type == "tangential"]),
                "radial_pathways": len([p for p in all_pathways if p.migration_type == "radial"]),
                "transposition_pathways": len([p for p in all_pathways if p.migration_type == "nuclear_transposition"]),
                "total_migrating_cells": sum(p.cell_count_per_pathway for p in all_pathways),
                "peak_migration_density": float(np.max(migration_density_map))
            },
            "map_files": {
                "tangential_migration": str(tangential_file),
                "radial_migration": str(radial_file),
                "nuclear_transposition": str(transposition_file),
                "migration_density": str(density_file),
                "bergmann_scaffolds": str(bergmann_file)
            }
        }
        
        logger.info("Created integrated 3D migration map")
        return map_results
    
    def execute_implementation(self) -> Dict[str, Any]:
        """Execute all migration path implementations."""
        logger.info("Executing cerebellar migration path implementation")
        
        implementation_results = {
            "implementation_date": datetime.now().isoformat(),
            "batch_phase": "Phase_1_Batch_B_Steps_B4.1_to_B4.5",
            "pathways_implemented": [],
            "total_pathways": 0,
            "implementation_status": "completed"
        }
        
        # 1. Implement tangential migration routes
        logger.info("=== Implementing Tangential Migration Routes ===")
        tangential_pathways = self.implement_tangential_migration_routes()
        implementation_results["pathways_implemented"].extend(["tangential_granule_migration"])
        
        # 2. Model radial migration along Bergmann glia
        logger.info("=== Modeling Radial Migration Along Bergmann Glia ===")
        radial_pathways, bergmann_scaffolds = self.model_radial_migration_bergmann_glia()
        implementation_results["pathways_implemented"].extend(["radial_bergmann_migration", "bergmann_glia_scaffolds"])
        
        # 3. Create Purkinje transposition patterns
        logger.info("=== Creating Purkinje Nuclear Transposition ===")
        transposition_pathways = self.create_purkinje_transposition_patterns()
        implementation_results["pathways_implemented"].extend(["purkinje_nuclear_transposition"])
        
        # 4. Define stellate/basket migration
        logger.info("=== Defining Stellate/Basket Cell Migration ===")
        interneuron_pathways = self.define_stellate_basket_migration()
        implementation_results["pathways_implemented"].extend(["stellate_basket_migration"])
        
        # 5. Implement deep nuclei migration
        logger.info("=== Implementing Deep Nuclei Migration ===")
        deep_nuclei_pathways = self.implement_deep_nuclei_migration()
        implementation_results["pathways_implemented"].extend(["deep_nuclei_migration"])
        
        # Combine all pathways
        all_pathways = (tangential_pathways + radial_pathways + transposition_pathways + 
                       interneuron_pathways + deep_nuclei_pathways)
        implementation_results["total_pathways"] = len(all_pathways)
        
        # 6. Create integrated migration map
        logger.info("=== Creating Integrated Migration Map ===")
        migration_map = self.create_integrated_migration_map(all_pathways, bergmann_scaffolds)
        implementation_results["migration_map_details"] = migration_map
        
        # Save pathway definitions
        pathways_file = self.metadata_dir / "migration_pathway_definitions.json"
        pathways_data = [
            {
                "pathway_name": pathway.pathway_name,
                "cell_type": pathway.cell_type,
                "migration_type": pathway.migration_type,
                "coordinates": {
                    "origin": pathway.origin_coordinates,
                    "destination": pathway.destination_coordinates
                },
                "migration_properties": {
                    "speed_um_hour": pathway.migration_speed_um_hour,
                    "duration_hours": pathway.migration_duration_hours,
                    "cell_count": pathway.cell_count_per_pathway
                },
                "guidance_molecules": pathway.guidance_molecules,
                "developmental_window": pathway.developmental_window
            } for pathway in all_pathways
        ]
        
        with open(pathways_file, 'w') as f:
            json.dump(pathways_data, f, indent=2)
        
        # Save Bergmann scaffolds
        scaffolds_file = self.metadata_dir / "bergmann_glia_scaffolds.json"
        scaffolds_data = [
            {
                "scaffold_id": scaffold.scaffold_id,
                "positions": {
                    "origin": scaffold.origin_position,
                    "target": scaffold.target_position
                },
                "properties": {
                    "fiber_length_um": scaffold.fiber_length_um,
                    "guidance_capacity": scaffold.guidance_capacity
                },
                "molecular_markers": scaffold.molecular_markers
            } for scaffold in bergmann_scaffolds
        ]
        
        with open(scaffolds_file, 'w') as f:
            json.dump(scaffolds_data, f, indent=2)
        
        # Save complete results
        results_file = self.metadata_dir / "migration_path_implementation_results.json"
        with open(results_file, 'w') as f:
            json.dump(implementation_results, f, indent=2)
        
        logger.info(f"Migration path implementation completed. Results saved to {results_file}")
        return implementation_results


def main():
    """Execute cerebellar migration path implementation."""
    
    print("🧬 CEREBELLAR MIGRATION PATH IMPLEMENTATION")
    print("=" * 60)
    print("Phase 1 ▸ Batch B ▸ Steps B4.1-B4.5")
    print("Tangential + Radial + Nuclear Transposition + Deep Nuclei Migration")
    print()
    
    # Initialize implementer
    implementer = CerebellarMigrationPathImplementer()
    
    # Execute implementation
    results = implementer.execute_implementation()
    
    # Print implementation summary
    print(f"✅ Migration path implementation completed")
    print(f"🧬 Pathways implemented: {len(results['pathways_implemented'])}")
    print(f"📊 Total pathways: {results['total_pathways']}")
    print(f"🎯 Implementation status: {results['implementation_status']}")
    print()
    
    # Display pathway details
    print("📥 Migration Pathways Implemented:")
    for pathway_type in results['pathways_implemented']:
        print(f"  • {pathway_type.replace('_', ' ')}")
    
    if 'migration_map_details' in results:
        migration_stats = results['migration_map_details']['migration_statistics']
        print(f"\n📊 Migration Statistics:")
        print(f"  • Total pathways: {migration_stats['total_pathways']}")
        print(f"  • Bergmann scaffolds: {migration_stats['bergmann_scaffolds']}")
        print(f"  • Tangential pathways: {migration_stats['tangential_pathways']}")
        print(f"  • Radial pathways: {migration_stats['radial_pathways']}")
        print(f"  • Total migrating cells: {migration_stats['total_migrating_cells']:,}")
    
    print()
    print("📁 Data Locations:")
    print(f"  • Pathway definitions: {implementer.pathways_dir}")
    print(f"  • Bergmann scaffolds: {implementer.scaffolds_dir}")
    print(f"  • Migration maps: {implementer.maps_dir}")
    print(f"  • Metadata: {implementer.metadata_dir}")
    
    print()
    print("🎯 Next Steps:")
    print("- Validate migration speeds against experimental data")
    print("- Integrate migration with cell population dynamics")
    print("- Proceed to Batch C: Validation & Testing")
    print("- Begin Batch D: Deployment & Monitoring")


if __name__ == "__main__":
    main()
