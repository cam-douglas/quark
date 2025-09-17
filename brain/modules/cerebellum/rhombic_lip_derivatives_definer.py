#!/usr/bin/env python3
"""
Rhombic Lip Derivatives and External Granular Layer Migration Routes Definer

Defines rhombic lip derivatives and external granular layer (EGL) migration
routes for granule cell precursors in cerebellar development. The rhombic lip
is the Math1/Atoh1+ source of all granule cells, which migrate tangentially
to form the EGL, then radially inward to form the internal granular layer.

Key anatomical features:
- Rhombic lip: Math1+ granule precursor source
- External granular layer: Proliferative tangential migration zone
- Radial migration routes: EGL → IGL along Bergmann glia
- Tangential migration: Rostral and caudal streams

Author: Quark Brain Development Team
Date: 2025-09-17
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RhombicLipDerivative:
    """Definition of rhombic lip derivative structure."""
    structure_name: str
    source_location: str
    target_location: str
    migration_type: str  # "tangential", "radial", "mixed"
    migration_speed_um_hour: float
    cell_types_generated: List[str]
    molecular_markers: List[str]
    developmental_window: str
    spatial_extent: Dict[str, float]


@dataclass
class MigrationRoute:
    """Definition of granule cell migration route."""
    route_name: str
    origin_coordinates: Tuple[float, float, float]  # (M-L, A-P, D-V)
    destination_coordinates: Tuple[float, float, float]
    pathway_type: str  # "tangential_rostral", "tangential_caudal", "radial_inward"
    guidance_molecules: List[str]
    migration_duration_hours: float
    cell_density_per_route: int


class RhombicLipDerivativesDefiner:
    """Defines rhombic lip derivatives and EGL migration routes."""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize rhombic lip derivatives definer."""
        self.data_dir = Path(data_dir)
        self.rhombic_lip_dir = self.data_dir / "rhombic_lip_derivatives"
        self.migration_dir = self.rhombic_lip_dir / "migration_routes"
        self.derivatives_dir = self.rhombic_lip_dir / "derivatives"
        self.metadata_dir = self.rhombic_lip_dir / "metadata"
        
        for directory in [self.rhombic_lip_dir, self.migration_dir, self.derivatives_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized rhombic lip derivatives definer")
        logger.info(f"Data directory: {self.rhombic_lip_dir}")
    
    def define_rhombic_lip_derivatives(self) -> List[RhombicLipDerivative]:
        """Define all rhombic lip derivative structures."""
        logger.info("Defining rhombic lip derivatives and structures")
        
        derivatives = [
            RhombicLipDerivative(
                structure_name="external_granular_layer",
                source_location="rhombic_lip",
                target_location="cerebellar_surface",
                migration_type="tangential",
                migration_speed_um_hour=30.0,
                cell_types_generated=["granule_cell_precursor", "unipolar_brush_cell"],
                molecular_markers=["Math1", "Atoh1", "NeuroD1", "Zic1", "Pax6"],
                developmental_window="E10.5-P14",
                spatial_extent={
                    "thickness_um": 150.0,
                    "mediolateral_extent_mm": 8.0,
                    "rostrocaudal_extent_mm": 6.0
                }
            ),
            RhombicLipDerivative(
                structure_name="internal_granular_layer",
                source_location="external_granular_layer",
                target_location="cerebellar_core",
                migration_type="radial",
                migration_speed_um_hour=20.0,
                cell_types_generated=["mature_granule_cell", "Golgi_cell"],
                molecular_markers=["NeuroD1", "Zic1", "Zic2", "Gabra6", "Nmdar1"],
                developmental_window="E12.5-P21",
                spatial_extent={
                    "thickness_um": 400.0,
                    "mediolateral_extent_mm": 8.0,
                    "rostrocaudal_extent_mm": 6.0
                }
            ),
            RhombicLipDerivative(
                structure_name="unipolar_brush_cells",
                source_location="rhombic_lip_dorsal",
                target_location="granular_layer_patches",
                migration_type="mixed",
                migration_speed_um_hour=15.0,
                cell_types_generated=["unipolar_brush_cell"],
                molecular_markers=["Tbr2", "Calb2", "mGluR1"],
                developmental_window="E12.5-P7",
                spatial_extent={
                    "patch_diameter_um": 100.0,
                    "patches_per_lobule": 5,
                    "total_patches": 50
                }
            )
        ]
        
        logger.info(f"Defined {len(derivatives)} rhombic lip derivatives")
        return derivatives
    
    def define_migration_routes(self) -> List[MigrationRoute]:
        """Define granule cell migration routes from rhombic lip."""
        logger.info("Defining granule cell migration routes")
        
        routes = [
            MigrationRoute(
                route_name="rostral_tangential_stream",
                origin_coordinates=(0.5, 0.45, 0.9),  # Rhombic lip rostral
                destination_coordinates=(0.3, 0.35, 0.85),  # Rostral EGL
                pathway_type="tangential_rostral",
                guidance_molecules=["Netrin1", "Slit2", "Ephrin_B2"],
                migration_duration_hours=24.0,
                cell_density_per_route=10000
            ),
            MigrationRoute(
                route_name="caudal_tangential_stream", 
                origin_coordinates=(0.5, 0.45, 0.9),  # Rhombic lip
                destination_coordinates=(0.7, 0.55, 0.85),  # Caudal EGL
                pathway_type="tangential_caudal",
                guidance_molecules=["Netrin1", "Slit2", "Ephrin_B2"],
                migration_duration_hours=24.0,
                cell_density_per_route=10000
            ),
            MigrationRoute(
                route_name="medial_radial_stream",
                origin_coordinates=(0.3, 0.4, 0.85),  # Medial EGL
                destination_coordinates=(0.3, 0.4, 0.4),  # Medial IGL
                pathway_type="radial_inward",
                guidance_molecules=["Reelin", "Bergmann_glia_contact", "BDNF"],
                migration_duration_hours=48.0,
                cell_density_per_route=15000
            ),
            MigrationRoute(
                route_name="lateral_radial_stream",
                origin_coordinates=(0.7, 0.4, 0.85),  # Lateral EGL
                destination_coordinates=(0.7, 0.4, 0.4),  # Lateral IGL
                pathway_type="radial_inward",
                guidance_molecules=["Reelin", "Bergmann_glia_contact", "BDNF"],
                migration_duration_hours=48.0,
                cell_density_per_route=15000
            ),
            MigrationRoute(
                route_name="deep_nuclei_stream",
                origin_coordinates=(0.5, 0.42, 0.8),  # Rostral rhombic lip
                destination_coordinates=(0.5, 0.43, 0.5),  # Deep nuclei region
                pathway_type="radial_early",
                guidance_molecules=["Netrin1", "Semaphorin3A", "PlexinA1"],
                migration_duration_hours=36.0,
                cell_density_per_route=5000
            )
        ]
        
        logger.info(f"Defined {len(routes)} migration routes")
        return routes
    
    def download_rhombic_lip_data(self) -> Dict[str, any]:
        """Download rhombic lip and migration-related expression data."""
        logger.info("Downloading rhombic lip and migration data")
        
        download_results = {
            "download_date": datetime.now().isoformat(),
            "genes_targeted": ["Math1", "Atoh1", "NeuroD1", "Reln", "Netrin1"],
            "files_downloaded": [],
            "download_status": {},
            "total_size_mb": 0
        }
        
        allen_dir = self.derivatives_dir / "allen_brain_map"
        allen_dir.mkdir(exist_ok=True)
        
        base_url = "http://api.brain-map.org/api/v2"
        
        queries = {
            "math1_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Atoh1'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))",
            "neurod1_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Neurod1'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))",
            "netrin1_experiments": f"{base_url}/data/query.json?criteria=model::SectionDataSet,rma::criteria,genes[acronym$eq'Ntn1'],products[abbreviation$eq'DevMouse']&include=specimen(donor(age))"
        }
        
        for query_name, query_url in queries.items():
            try:
                with urllib.request.urlopen(query_url, timeout=30) as response:
                    data = json.loads(response.read())
                
                output_file = allen_dir / f"{query_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                file_size_mb = output_file.stat().st_size / (1024*1024)
                download_results["files_downloaded"].append(str(output_file))
                download_results["download_status"][query_name] = "success"
                download_results["total_size_mb"] += file_size_mb
                
                logger.info(f"✅ Downloaded {query_name} ({file_size_mb:.2f}MB)")
                
            except Exception as e:
                download_results["download_status"][query_name] = f"failed: {str(e)}"
                logger.error(f"❌ Failed to download {query_name}: {e}")
        
        return download_results
    
    def create_migration_map(self, routes: List[MigrationRoute]) -> Dict[str, any]:
        """Create 3D migration route map."""
        logger.info("Creating 3D migration route map")
        
        grid_size = (100, 100, 50)  # X (M-L), Y (A-P), Z (D-V)
        migration_map = np.zeros(grid_size)
        route_density_map = np.zeros(grid_size)
        
        for route in routes:
            # Convert coordinates to grid indices
            origin_idx = tuple(int(coord * dim) for coord, dim in zip(route.origin_coordinates, grid_size))
            dest_idx = tuple(int(coord * dim) for coord, dim in zip(route.destination_coordinates, grid_size))
            
            # Create migration pathway
            if route.pathway_type == "tangential_rostral" or route.pathway_type == "tangential_caudal":
                # Tangential migration along surface
                y_path = np.linspace(origin_idx[1], dest_idx[1], 20)
                z_surface = int(0.85 * grid_size[2])  # Near surface
                
                for y in y_path.astype(int):
                    if 0 <= y < grid_size[1]:
                        migration_map[:, y, z_surface] = 1.0
                        route_density_map[:, y, z_surface] += route.cell_density_per_route
                        
            elif route.pathway_type == "radial_inward":
                # Radial migration inward
                z_path = np.linspace(origin_idx[2], dest_idx[2], 20)
                x_pos, y_pos = origin_idx[0], origin_idx[1]
                
                for z in z_path.astype(int):
                    if 0 <= z < grid_size[2]:
                        migration_map[x_pos, y_pos, z] = 2.0
                        route_density_map[x_pos, y_pos, z] += route.cell_density_per_route
        
        # Save migration maps
        migration_file = self.migration_dir / "migration_route_map.npy"
        density_file = self.migration_dir / "cell_density_map.npy"
        
        np.save(migration_file, migration_map)
        np.save(density_file, route_density_map)
        
        map_results = {
            "creation_date": datetime.now().isoformat(),
            "grid_dimensions": grid_size,
            "total_routes": len(routes),
            "migration_statistics": {
                "tangential_fraction": float(np.mean(migration_map == 1.0)),
                "radial_fraction": float(np.mean(migration_map == 2.0)),
                "total_migrating_cells": int(np.sum(route_density_map)),
                "peak_density": float(np.max(route_density_map))
            }
        }
        
        logger.info("Created 3D migration route map")
        return map_results
    
    def execute_definition(self) -> Dict[str, any]:
        """Execute rhombic lip derivatives definition."""
        logger.info("Executing rhombic lip derivatives definition")
        
        results = {
            "definition_date": datetime.now().isoformat(),
            "derivatives_defined": [],
            "migration_routes_defined": [],
            "total_data_mb": 0
        }
        
        # Define derivatives
        derivatives = self.define_rhombic_lip_derivatives()
        results["derivatives_defined"] = [d.structure_name for d in derivatives]
        
        # Define migration routes
        routes = self.define_migration_routes()
        results["migration_routes_defined"] = [r.route_name for r in routes]
        
        # Download related expression data
        download_results = self.download_rhombic_lip_data()
        results["total_data_mb"] = download_results["total_size_mb"]
        results["download_details"] = download_results
        
        # Create migration map
        migration_map = self.create_migration_map(routes)
        results["migration_map"] = migration_map
        
        # Save derivatives
        derivatives_file = self.metadata_dir / "rhombic_lip_derivatives.json"
        derivatives_data = [
            {
                "structure_name": d.structure_name,
                "migration_properties": {
                    "source": d.source_location,
                    "target": d.target_location,
                    "type": d.migration_type,
                    "speed_um_hour": d.migration_speed_um_hour
                },
                "cell_types": d.cell_types_generated,
                "molecular_markers": d.molecular_markers,
                "temporal_window": d.developmental_window,
                "spatial_extent": d.spatial_extent
            } for d in derivatives
        ]
        
        with open(derivatives_file, 'w') as f:
            json.dump(derivatives_data, f, indent=2)
        
        # Save migration routes
        routes_file = self.metadata_dir / "migration_routes.json"
        routes_data = [
            {
                "route_name": r.route_name,
                "coordinates": {
                    "origin": r.origin_coordinates,
                    "destination": r.destination_coordinates
                },
                "pathway_properties": {
                    "type": r.pathway_type,
                    "duration_hours": r.migration_duration_hours,
                    "cell_density": r.cell_density_per_route
                },
                "guidance_molecules": r.guidance_molecules
            } for r in routes
        ]
        
        with open(routes_file, 'w') as f:
            json.dump(routes_data, f, indent=2)
        
        # Save complete results
        results_file = self.metadata_dir / "rhombic_lip_definition_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Rhombic lip derivatives definition completed")
        return results


def main():
    """Execute rhombic lip derivatives definition."""
    print("🧬 RHOMBIC LIP DERIVATIVES DEFINITION")
    print("=" * 60)
    print("Phase 1 ▸ Batch A ▸ Step A3.1")
    print("External Granular Layer Migration Routes")
    
    definer = RhombicLipDerivativesDefiner()
    results = definer.execute_definition()
    
    print(f"✅ Rhombic lip derivatives definition completed")
    print(f"🧬 Derivatives defined: {len(results['derivatives_defined'])}")
    print(f"🗺️ Migration routes: {len(results['migration_routes_defined'])}")
    print(f"💾 Total data: {results['total_data_mb']:.1f}MB")
    
    print("\n🎯 Rhombic Lip Derivatives:")
    for derivative in results['derivatives_defined']:
        print(f"  • {derivative}")
    
    print("\n🗺️ Migration Routes:")
    for route in results['migration_routes_defined']:
        print(f"  • {route}")
    
    if 'migration_map' in results:
        stats = results['migration_map']['migration_statistics']
        print(f"\n📊 Migration Statistics:")
        print(f"  • Total migrating cells: {stats['total_migrating_cells']:,}")
        print(f"  • Tangential pathways: {stats['tangential_fraction']:.1%}")
        print(f"  • Radial pathways: {stats['radial_fraction']:.1%}")
    
    print(f"\n📁 Data location: {definer.rhombic_lip_dir}")


if __name__ == "__main__":
    main()
