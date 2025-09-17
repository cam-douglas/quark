#!/usr/bin/env python3
"""
Cerebellar Structural Scaffold Constructor

Constructs all structural scaffolds for cerebellar development including
parasagittal microzones, Purkinje cell monolayer, external granular layer,
deep nuclei primordia, and cortical layer organization.

Structural components:
- 50 parasagittal microzones (150-200 μm width) using Zebrin II
- Purkinje cell monolayer template (50 μm soma spacing)
- External granular layer proliferative zone (100-200 μm thickness)
- Deep nuclei primordia: fastigial, interposed, dentate
- Cortical layers: molecular (200 μm), Purkinje (50 μm), granular (400 μm)

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
class MicrozoneDefinition:
    """Definition of parasagittal microzone."""
    zone_id: int
    zone_type: str  # "positive" or "negative"
    mediolateral_position: float
    width_um: float
    zebrin_expression_level: float
    climbing_fiber_source: str
    functional_domain: str


@dataclass
class CorticalLayerDefinition:
    """Definition of cerebellar cortical layer."""
    layer_name: str
    thickness_um: float
    cell_types: List[str]
    dorsoventral_position: float
    developmental_onset: str


@dataclass
class DeepNucleusDefinition:
    """Definition of deep cerebellar nucleus."""
    nucleus_name: str
    mediolateral_position: float
    volume_mm3: float
    neuron_count: int
    cell_types: List[str]
    afferent_sources: List[str]
    efferent_targets: List[str]


class CerebellarStructuralScaffoldConstructor:
    """Constructs all cerebellar structural scaffolds."""
    
    def __init__(self, data_dir: str = "/Users/camdouglas/quark/data/datasets/cerebellum"):
        """Initialize structural scaffold constructor."""
        self.data_dir = Path(data_dir)
        self.scaffold_dir = self.data_dir / "structural_scaffolds"
        self.microzones_dir = self.scaffold_dir / "microzones"
        self.layers_dir = self.scaffold_dir / "cortical_layers"
        self.nuclei_dir = self.scaffold_dir / "deep_nuclei"
        self.metadata_dir = self.scaffold_dir / "metadata"
        
        for directory in [self.scaffold_dir, self.microzones_dir, self.layers_dir, self.nuclei_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initialized structural scaffold constructor")
        logger.info(f"Data directory: {self.scaffold_dir}")
    
    def generate_parasagittal_microzones(self) -> List[MicrozoneDefinition]:
        """Generate 50 parasagittal microzones using Zebrin II expression."""
        logger.info("Generating 50 parasagittal microzones")
        
        microzones = []
        
        # Generate 50 zones across mediolateral extent (8mm total width)
        total_width_mm = 8.0
        zone_positions = np.linspace(0.0, 1.0, 51)  # 51 boundaries for 50 zones
        
        for i in range(50):
            zone_id = i + 1
            
            # Alternate between positive and negative zones
            zone_type = "positive" if i % 2 == 0 else "negative"
            zebrin_level = 1.0 if zone_type == "positive" else 0.1
            
            # Calculate zone width (varies slightly across cerebellum)
            base_width = total_width_mm / 50  # 160 μm average
            width_variation = np.random.uniform(0.8, 1.2)  # ±20% variation
            zone_width = base_width * width_variation * 1000  # Convert to μm
            zone_width = np.clip(zone_width, 150.0, 200.0)  # Constrain to requirement
            
            # Mediolateral position
            ml_position = (zone_positions[i] + zone_positions[i + 1]) / 2
            
            # Climbing fiber source mapping
            if ml_position < 0.25:  # Vermis
                cf_source = "medial_accessory_olive"
                functional_domain = "oculomotor"
            elif ml_position < 0.6:  # Paravermis
                cf_source = "dorsal_accessory_olive"
                functional_domain = "spinocerebellar"
            else:  # Hemispheres
                cf_source = "principal_olive"
                functional_domain = "cerebrocerebellar"
            
            microzone = MicrozoneDefinition(
                zone_id=zone_id,
                zone_type=zone_type,
                mediolateral_position=ml_position,
                width_um=zone_width,
                zebrin_expression_level=zebrin_level,
                climbing_fiber_source=cf_source,
                functional_domain=functional_domain
            )
            
            microzones.append(microzone)
        
        logger.info(f"Generated {len(microzones)} parasagittal microzones")
        return microzones
    
    def create_purkinje_monolayer_template(self) -> Dict[str, Any]:
        """Create Purkinje cell monolayer template with 50 μm spacing."""
        logger.info("Creating Purkinje cell monolayer template")
        
        # Cerebellar dimensions (approximate)
        cerebellar_width_mm = 8.0
        cerebellar_length_mm = 6.0
        soma_spacing_um = 50.0
        
        # Calculate Purkinje cell positions
        x_positions = np.arange(0, cerebellar_width_mm * 1000, soma_spacing_um)
        y_positions = np.arange(0, cerebellar_length_mm * 1000, soma_spacing_um)
        
        purkinje_positions = []
        purkinje_count = 0
        
        for x in x_positions:
            for y in y_positions:
                # Purkinje layer at fixed dorsoventral position
                z_position = 0.65 * 50 * 50  # D-V position in μm (grid coordinate * voxel size)
                
                purkinje_positions.append({
                    "purkinje_id": purkinje_count + 1,
                    "position_um": [float(x), float(y), float(z_position)],
                    "soma_diameter_um": 20.0,
                    "dendritic_field_um": 400.0,
                    "microzone_assignment": int((x / (cerebellar_width_mm * 1000)) * 50) + 1
                })
                purkinje_count += 1
        
        monolayer_template = {
            "creation_date": datetime.now().isoformat(),
            "template_type": "purkinje_monolayer",
            "spatial_parameters": {
                "soma_spacing_um": soma_spacing_um,
                "cerebellar_width_mm": cerebellar_width_mm,
                "cerebellar_length_mm": cerebellar_length_mm,
                "layer_thickness_um": 50.0
            },
            "purkinje_cells": {
                "total_count": purkinje_count,
                "density_per_mm2": purkinje_count / (cerebellar_width_mm * cerebellar_length_mm),
                "positions": purkinje_positions[:1000]  # Limit to first 1000 for file size
            },
            "microzone_integration": {
                "microzones_covered": 50,
                "cells_per_microzone": purkinje_count // 50,
                "spacing_validation": "50um_requirement_met"
            }
        }
        
        # Save Purkinje positions
        positions_file = self.layers_dir / "purkinje_cell_positions.npy"
        positions_array = np.array([p["position_um"] for p in purkinje_positions])
        np.save(positions_file, positions_array)
        
        logger.info(f"Created Purkinje monolayer: {purkinje_count:,} cells, {soma_spacing_um}μm spacing")
        return monolayer_template
    
    def model_external_granular_layer(self) -> Dict[str, Any]:
        """Model external granular layer as proliferative zone."""
        logger.info("Modeling external granular layer")
        
        egl_model = {
            "creation_date": datetime.now().isoformat(),
            "layer_type": "external_granular_layer",
            "spatial_properties": {
                "thickness_range_um": [100.0, 200.0],
                "average_thickness_um": 150.0,
                "surface_area_mm2": 48.0,  # 8mm × 6mm
                "total_volume_mm3": 7.2   # 48 mm² × 0.15 mm
            },
            "proliferative_properties": {
                "cell_density_per_mm3": 2000000,  # 2M cells/mm³
                "total_cells": int(7.2 * 2000000),  # ~14.4M cells
                "division_rate_per_hour": 0.8,
                "cell_cycle_length_hours": 12.0,
                "proliferation_window": "E12.5-P14"
            },
            "molecular_characteristics": {
                "progenitor_markers": ["Math1", "Atoh1", "Pax6"],
                "proliferation_markers": ["Ki67", "PCNA", "Cyclin_D1"],
                "migration_markers": ["Dcx", "Tuba1a", "Map1b"],
                "differentiation_markers": ["NeuroD1", "Zic1", "Zic2"]
            },
            "spatial_organization": {
                "outer_proliferative_zone": {
                    "thickness_um": 100.0,
                    "cell_density_ratio": 1.0,
                    "proliferation_rate": 0.8
                },
                "inner_differentiating_zone": {
                    "thickness_um": 50.0,
                    "cell_density_ratio": 0.6,
                    "proliferation_rate": 0.2
                }
            }
        }
        
        # Create EGL density map
        grid_size = (100, 100, 50)
        egl_density_map = np.zeros(grid_size)
        
        # EGL occupies dorsal surface (top 4 voxels = 200μm)
        egl_thickness_voxels = 4
        surface_z = grid_size[2] - egl_thickness_voxels
        
        # Fill EGL region with cell density
        egl_density_map[:, :, surface_z:] = egl_model["proliferative_properties"]["cell_density_per_mm3"] / 1000  # cells per voxel
        
        # Save EGL density map
        egl_file = self.layers_dir / "external_granular_layer_density.npy"
        np.save(egl_file, egl_density_map)
        
        logger.info(f"Modeled EGL: {egl_model['proliferative_properties']['total_cells']:,} cells")
        return egl_model
    
    def construct_deep_nuclei_primordia(self) -> List[DeepNucleusDefinition]:
        """Construct deep cerebellar nuclei primordia."""
        logger.info("Constructing deep cerebellar nuclei primordia")
        
        deep_nuclei = [
            DeepNucleusDefinition(
                nucleus_name="fastigial_nucleus",
                mediolateral_position=0.5,  # Medial (midline)
                volume_mm3=3.14,
                neuron_count=50000,
                cell_types=["fastigial_neuron", "GABAergic_interneuron"],
                afferent_sources=["vermis_purkinje_cells", "spinal_cord", "vestibular_nuclei"],
                efferent_targets=["vestibular_nuclei", "reticular_formation", "spinal_cord"]
            ),
            DeepNucleusDefinition(
                nucleus_name="interposed_nucleus",
                mediolateral_position=0.65,  # Intermediate
                volume_mm3=1.77,
                neuron_count=35000,
                cell_types=["interposed_neuron", "GABAergic_interneuron"],
                afferent_sources=["paravermis_purkinje_cells", "spinal_cord", "pontine_nuclei"],
                efferent_targets=["red_nucleus", "thalamus", "reticular_formation"]
            ),
            DeepNucleusDefinition(
                nucleus_name="dentate_nucleus",
                mediolateral_position=0.8,   # Lateral
                volume_mm3=7.07,
                neuron_count=100000,
                cell_types=["dentate_neuron", "GABAergic_interneuron"],
                afferent_sources=["hemisphere_purkinje_cells", "pontine_nuclei", "inferior_olive"],
                efferent_targets=["thalamus", "red_nucleus", "inferior_olive"]
            )
        ]
        
        logger.info(f"Constructed {len(deep_nuclei)} deep nuclei primordia")
        return deep_nuclei
    
    def define_cortical_layers(self) -> List[CorticalLayerDefinition]:
        """Define cerebellar cortical layer organization."""
        logger.info("Defining cerebellar cortical layers")
        
        layers = [
            CorticalLayerDefinition(
                layer_name="molecular_layer",
                thickness_um=200.0,
                cell_types=["basket_cell", "stellate_cell", "parallel_fibers", "climbing_fibers"],
                dorsoventral_position=0.8,  # Superficial
                developmental_onset="E13.5"
            ),
            CorticalLayerDefinition(
                layer_name="purkinje_cell_layer",
                thickness_um=50.0,
                cell_types=["Purkinje_cell"],
                dorsoventral_position=0.65,  # Middle
                developmental_onset="E11.5"
            ),
            CorticalLayerDefinition(
                layer_name="internal_granular_layer",
                thickness_um=400.0,
                cell_types=["granule_cell", "Golgi_cell", "unipolar_brush_cell"],
                dorsoventral_position=0.4,   # Deep
                developmental_onset="E12.5"
            )
        ]
        
        logger.info(f"Defined {len(layers)} cortical layers")
        return layers
    
    def create_integrated_scaffold_map(self, microzones: List[MicrozoneDefinition],
                                     monolayer: Dict[str, Any],
                                     egl_model: Dict[str, Any],
                                     deep_nuclei: List[DeepNucleusDefinition],
                                     layers: List[CorticalLayerDefinition]) -> Dict[str, Any]:
        """Create integrated 3D structural scaffold map."""
        logger.info("Creating integrated 3D structural scaffold map")
        
        grid_size = (100, 100, 50)  # X (M-L), Y (A-P), Z (D-V)
        
        # Initialize scaffold maps
        microzone_map = np.zeros(grid_size)
        layer_map = np.zeros(grid_size)
        nuclei_map = np.zeros(grid_size)
        
        # Map microzones
        for zone in microzones:
            ml_center = int(zone.mediolateral_position * grid_size[0])
            zone_width_voxels = int(zone.width_um / 50.0)  # 50μm per voxel
            
            ml_start = max(0, ml_center - zone_width_voxels // 2)
            ml_end = min(grid_size[0], ml_center + zone_width_voxels // 2)
            
            # Fill microzone (full rostrocaudal, cortical layers only)
            microzone_map[ml_start:ml_end, :, 20:45] = zone.zone_id
        
        # Map cortical layers
        for layer in layers:
            z_center = int(layer.dorsoventral_position * grid_size[2])
            layer_thickness_voxels = int(layer.thickness_um / 50.0)
            
            z_start = max(0, z_center - layer_thickness_voxels // 2)
            z_end = min(grid_size[2], z_center + layer_thickness_voxels // 2)
            
            # Layer encoding: 1=molecular, 2=Purkinje, 3=granular
            layer_id = {"molecular_layer": 1, "purkinje_cell_layer": 2, "internal_granular_layer": 3}.get(layer.layer_name, 0)
            layer_map[:, :, z_start:z_end] = layer_id
        
        # Map deep nuclei
        for nucleus in deep_nuclei:
            ml_center = int(nucleus.mediolateral_position * grid_size[0])
            
            # Nucleus size based on volume (approximate as sphere)
            radius_mm = (3 * nucleus.volume_mm3 / (4 * np.pi)) ** (1/3)
            radius_voxels = int(radius_mm * 1000 / 50.0)  # Convert to voxels
            
            # Deep nuclei at ventral position
            z_center = int(0.3 * grid_size[2])  # Ventral to cortical layers
            y_center = int(0.45 * grid_size[1])  # Central A-P position
            
            # Fill spherical nucleus region
            for x in range(max(0, ml_center - radius_voxels), min(grid_size[0], ml_center + radius_voxels)):
                for y in range(max(0, y_center - radius_voxels), min(grid_size[1], y_center + radius_voxels)):
                    for z in range(max(0, z_center - radius_voxels), min(grid_size[2], z_center + radius_voxels)):
                        distance = np.sqrt((x - ml_center)**2 + (y - y_center)**2 + (z - z_center)**2)
                        if distance <= radius_voxels:
                            nucleus_id = {"fastigial_nucleus": 1, "interposed_nucleus": 2, "dentate_nucleus": 3}.get(nucleus.nucleus_name, 0)
                            nuclei_map[x, y, z] = nucleus_id
        
        # Save scaffold maps
        microzone_file = self.microzones_dir / "microzone_map.npy"
        layer_file = self.layers_dir / "cortical_layer_map.npy"
        nuclei_file = self.nuclei_dir / "deep_nuclei_map.npy"
        
        np.save(microzone_file, microzone_map)
        np.save(layer_file, layer_map)
        np.save(nuclei_file, nuclei_map)
        
        scaffold_results = {
            "creation_date": datetime.now().isoformat(),
            "grid_dimensions": grid_size,
            "voxel_size_um": 50.0,
            "scaffold_statistics": {
                "microzones_mapped": len(microzones),
                "cortical_layers_mapped": len(layers),
                "deep_nuclei_mapped": len(deep_nuclei),
                "purkinje_cells_templated": monolayer["purkinje_cells"]["total_count"],
                "total_scaffold_voxels": int(np.sum((microzone_map > 0) | (layer_map > 0) | (nuclei_map > 0)))
            },
            "file_locations": {
                "microzone_map": str(microzone_file),
                "layer_map": str(layer_file),
                "nuclei_map": str(nuclei_file),
                "purkinje_positions": str(self.layers_dir / "purkinje_cell_positions.npy")
            }
        }
        
        logger.info("Created integrated 3D structural scaffold map")
        return scaffold_results
    
    def execute_construction(self) -> Dict[str, Any]:
        """Execute all structural scaffold construction."""
        logger.info("Executing cerebellar structural scaffold construction")
        
        construction_results = {
            "construction_date": datetime.now().isoformat(),
            "batch_phase": "Phase_1_Batch_B_Steps_B3.1_to_B3.5",
            "scaffolds_constructed": [],
            "construction_status": "completed"
        }
        
        # 1. Generate parasagittal microzones
        logger.info("=== Generating Parasagittal Microzones ===")
        microzones = self.generate_parasagittal_microzones()
        construction_results["scaffolds_constructed"].append("parasagittal_microzones")
        
        # 2. Create Purkinje monolayer template
        logger.info("=== Creating Purkinje Monolayer Template ===")
        monolayer = self.create_purkinje_monolayer_template()
        construction_results["scaffolds_constructed"].append("purkinje_monolayer")
        
        # 3. Model external granular layer
        logger.info("=== Modeling External Granular Layer ===")
        egl_model = self.model_external_granular_layer()
        construction_results["scaffolds_constructed"].append("external_granular_layer")
        
        # 4. Construct deep nuclei primordia
        logger.info("=== Constructing Deep Nuclei Primordia ===")
        deep_nuclei = self.construct_deep_nuclei_primordia()
        construction_results["scaffolds_constructed"].append("deep_nuclei_primordia")
        
        # 5. Define cortical layers
        logger.info("=== Defining Cortical Layers ===")
        layers = self.define_cortical_layers()
        construction_results["scaffolds_constructed"].append("cortical_layers")
        
        # 6. Create integrated scaffold map
        logger.info("=== Creating Integrated Scaffold Map ===")
        scaffold_map = self.create_integrated_scaffold_map(microzones, monolayer, egl_model, deep_nuclei, layers)
        construction_results["integrated_scaffold"] = scaffold_map
        
        # Save all definitions
        microzones_file = self.metadata_dir / "microzone_definitions.json"
        microzones_data = [
            {
                "zone_id": zone.zone_id,
                "zone_type": zone.zone_type,
                "mediolateral_position": zone.mediolateral_position,
                "width_um": zone.width_um,
                "zebrin_expression_level": zone.zebrin_expression_level,
                "climbing_fiber_source": zone.climbing_fiber_source,
                "functional_domain": zone.functional_domain
            } for zone in microzones
        ]
        
        with open(microzones_file, 'w') as f:
            json.dump(microzones_data, f, indent=2)
        
        # Save other definitions
        monolayer_file = self.metadata_dir / "purkinje_monolayer_template.json"
        with open(monolayer_file, 'w') as f:
            json.dump(monolayer, f, indent=2)
        
        egl_file = self.metadata_dir / "external_granular_layer_model.json"
        with open(egl_file, 'w') as f:
            json.dump(egl_model, f, indent=2)
        
        nuclei_file = self.metadata_dir / "deep_nuclei_definitions.json"
        nuclei_data = [
            {
                "nucleus_name": nucleus.nucleus_name,
                "mediolateral_position": nucleus.mediolateral_position,
                "volume_mm3": nucleus.volume_mm3,
                "neuron_count": nucleus.neuron_count,
                "cell_types": nucleus.cell_types,
                "afferent_sources": nucleus.afferent_sources,
                "efferent_targets": nucleus.efferent_targets
            } for nucleus in deep_nuclei
        ]
        
        with open(nuclei_file, 'w') as f:
            json.dump(nuclei_data, f, indent=2)
        
        layers_file = self.metadata_dir / "cortical_layer_definitions.json"
        layers_data = [
            {
                "layer_name": layer.layer_name,
                "thickness_um": layer.thickness_um,
                "cell_types": layer.cell_types,
                "dorsoventral_position": layer.dorsoventral_position,
                "developmental_onset": layer.developmental_onset
            } for layer in layers
        ]
        
        with open(layers_file, 'w') as f:
            json.dump(layers_data, f, indent=2)
        
        # Save complete results
        results_file = self.metadata_dir / "structural_scaffold_construction_results.json"
        with open(results_file, 'w') as f:
            json.dump(construction_results, f, indent=2)
        
        logger.info(f"Structural scaffold construction completed. Results saved to {results_file}")
        return construction_results


def main():
    """Execute cerebellar structural scaffold construction."""
    
    print("🧬 CEREBELLAR STRUCTURAL SCAFFOLD CONSTRUCTION")
    print("=" * 60)
    print("Phase 1 ▸ Batch B ▸ Steps B3.1-B3.5")
    print("Microzones + Purkinje Layer + EGL + Deep Nuclei + Cortical Layers")
    print()
    
    # Initialize constructor
    constructor = CerebellarStructuralScaffoldConstructor()
    
    # Execute construction
    results = constructor.execute_construction()
    
    # Print construction summary
    print(f"✅ Structural scaffold construction completed")
    print(f"🧬 Scaffolds constructed: {len(results['scaffolds_constructed'])}")
    print(f"📊 Construction status: {results['construction_status']}")
    print()
    
    # Display scaffold details
    print("📥 Scaffolds Constructed:")
    for scaffold in results['scaffolds_constructed']:
        print(f"  • {scaffold.replace('_', ' ')}")
    
    if 'integrated_scaffold' in results:
        scaffold_stats = results['integrated_scaffold']['scaffold_statistics']
        print(f"\n📊 Scaffold Statistics:")
        print(f"  • Microzones mapped: {scaffold_stats['microzones_mapped']}")
        print(f"  • Cortical layers: {scaffold_stats['cortical_layers_mapped']}")
        print(f"  • Deep nuclei: {scaffold_stats['deep_nuclei_mapped']}")
        print(f"  • Purkinje cells: {scaffold_stats['purkinje_cells_templated']:,}")
        print(f"  • Total scaffold voxels: {scaffold_stats['total_scaffold_voxels']:,}")
    
    print()
    print("📁 Data Locations:")
    print(f"  • Microzones: {constructor.microzones_dir}")
    print(f"  • Layers: {constructor.layers_dir}")
    print(f"  • Deep nuclei: {constructor.nuclei_dir}")
    print(f"  • Metadata: {constructor.metadata_dir}")
    
    print()
    print("🎯 Next Steps:")
    print("- Validate scaffold dimensions against literature")
    print("- Integrate scaffolds with cell populations")
    print("- Proceed to B4: Migration Path Implementation")
    print("- Prepare for Batch C: Validation & Testing")


if __name__ == "__main__":
    main()
