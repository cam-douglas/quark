#!/usr/bin/env python3
"""Allen Brain Atlas Validation System.

Main coordinator for Allen Brain Atlas validation including data pipeline,
coordinate mapping, and comprehensive validation metrics for regional
segmentation accuracy against embryonic reference data.

Integration: Main coordinator for atlas validation system
Rationale: Main validation coordinator with focused responsibilities
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

from .spatial_grid import SpatialGrid
from .cell_fate_specifier import CellFateSpecifier
from .atlas_data_downloader import AtlasDataDownloader
from .atlas_coordinate_mapper import AtlasCoordinateMapper
from .atlas_validation_metrics import AtlasValidationMetrics
from .atlas_validation_types import AtlasReference

logger = logging.getLogger(__name__)

class AtlasValidationSystem:
    """Allen Brain Atlas validation system coordinator.
    
    Main coordinator for validating regional segmentation against Allen
    Brain Atlas embryonic reference data including data pipeline setup,
    coordinate system mapping, and comprehensive validation metrics.
    
    Key Components:
    - Atlas data download and management
    - Coordinate system mapping
    - Validation metrics calculation
    - Regional accuracy assessment
    """
    
    def __init__(self, spatial_grid: SpatialGrid,
                 cell_fate_specifier: CellFateSpecifier,
                 developmental_stage: str = "E10.5"):
        """Initialize atlas validation system.
        
        Args:
            spatial_grid: Spatial grid with morphogen data
            cell_fate_specifier: Cell fate specification system
            developmental_stage: Target developmental stage
        """
        self.grid = spatial_grid
        self.cell_fate_specifier = cell_fate_specifier
        self.developmental_stage = developmental_stage
        
        # Initialize component systems
        self.data_downloader = AtlasDataDownloader()
        self.coordinate_mapper = AtlasCoordinateMapper(spatial_grid.dimensions)
        self.validation_metrics = AtlasValidationMetrics(dice_threshold=0.80)
        
        # Validation state
        self.atlas_reference: Optional[AtlasReference] = None
        self.morphogen_segmentation: Optional[np.ndarray] = None
        self.coordinate_transform = None
        
        logger.info("Initialized AtlasValidationSystem")
        logger.info(f"Target stage: {developmental_stage}")
        logger.info(f"Grid dimensions: {spatial_grid.dimensions.x_size}x{spatial_grid.dimensions.y_size}x{spatial_grid.dimensions.z_size}")
    
    def setup_atlas_data_pipeline(self) -> bool:
        """Setup atlas data pipeline using existing downloaded data.
        
        Returns:
            True if pipeline setup successful
        """
        logger.info("Setting up atlas data pipeline using existing downloaded data")
        
        # Check for existing downloaded data first
        existing_manifest = self.data_downloader.data_dir / "atlas_integration_manifest.json"
        
        if existing_manifest.exists():
            logger.info("Found existing atlas data, using cached integration")
            # Load existing atlas reference
            self.atlas_reference = self._load_existing_atlas_reference()
        else:
            logger.info("No existing data found, downloading fresh data")
            # Download integrated atlas reference data (BrainSpan + Allen)
            self.atlas_reference = self.data_downloader.integrate_brainspan_with_allen(self.developmental_stage)
        
        if self.atlas_reference is None:
            logger.error("Failed to setup atlas reference data")
            return False
        
        # Validate atlas integrity
        integrity_results = self.data_downloader.validate_atlas_integrity(self.atlas_reference)
        
        if not integrity_results["overall_valid"]:
            logger.error("Atlas data integrity validation failed")
            return False
        
        # Create coordinate transformation
        self.coordinate_transform = self.coordinate_mapper.create_coordinate_transform(self.atlas_reference)
        
        logger.info("Atlas data pipeline setup completed successfully")
        logger.info(f"Atlas ID: {self.atlas_reference.atlas_id}")
        logger.info(f"Atlas dimensions: {self.atlas_reference.dimensions}")
        logger.info(f"Atlas resolution: {self.atlas_reference.resolution_um} µm")
        
        return True
    
    def generate_morphogen_segmentation(self) -> np.ndarray:
        """Generate segmentation from current morphogen concentrations.
        
        Returns:
            Segmentation labels array
        """
        logger.info("Generating morphogen-based segmentation")
        
        # Get current morphogen concentrations
        morphogen_concentrations = {}
        for morphogen in ['SHH', 'BMP', 'WNT', 'FGF']:
            if self.grid.has_morphogen(morphogen):
                morphogen_concentrations[morphogen] = self.grid.get_morphogen_concentration(morphogen)
        
        # Generate cell fate specification
        cell_fate_map = self.cell_fate_specifier.specify_cell_fates(morphogen_concentrations)
        
        # Convert cell fates to regional labels
        segmentation_labels = self._convert_cell_fates_to_regions(cell_fate_map)
        
        self.morphogen_segmentation = segmentation_labels
        
        unique_regions = len(np.unique(segmentation_labels)) - 1  # Exclude background
        logger.info(f"Generated segmentation with {unique_regions} regions")
        
        return segmentation_labels
    
    def _convert_cell_fates_to_regions(self, cell_fate_map: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert cell fate map to regional segmentation labels."""
        dims = self.grid.dimensions
        segmentation = np.zeros((dims.x_size, dims.y_size, dims.z_size), dtype=int)
        
        # Define mapping from cell fates to regions
        fate_to_region = {
            'floor_plate': 5,        # Ventral neural tube
            'motor_neuron': 5,       # Ventral neural tube
            'v3_interneuron': 5,     # Ventral neural tube
            'v2_interneuron': 5,     # Ventral neural tube
            'v1_interneuron': 5,     # Ventral neural tube
            'v0_interneuron': 5,     # Ventral neural tube
            'dorsal_interneuron': 6, # Dorsal neural tube
            'neural_crest': 6,       # Dorsal neural tube
            'roof_plate': 6,         # Dorsal neural tube
        }
        
        # Apply regional labels based on cell fates
        for fate_name, fate_mask in cell_fate_map.items():
            if fate_name in fate_to_region:
                region_label = fate_to_region[fate_name]
                segmentation[fate_mask > 0.5] = region_label
        
        # Add A-P regional labels based on position
        # This is simplified - in practice would use WNT/FGF concentrations
        for y in range(dims.y_size):
            ap_position = y / dims.y_size
            
            if ap_position < 0.3:
                # Forebrain
                mask = (segmentation == 0)  # Background regions
                segmentation[mask] = 1
            elif ap_position < 0.5:
                # Midbrain
                mask = (segmentation == 0)
                segmentation[mask] = 2
            elif ap_position < 0.8:
                # Hindbrain
                mask = (segmentation == 0)
                segmentation[mask] = 3
            else:
                # Spinal cord
                mask = (segmentation == 0)
                segmentation[mask] = 4
        
        return segmentation
    
    def validate_against_atlas(self) -> Dict[str, Any]:
        """Validate morphogen segmentation against Allen Atlas.
        
        Returns:
            Comprehensive validation results
        """
        if self.atlas_reference is None:
            logger.error("Atlas reference not loaded")
            return {}
        
        logger.info("Validating segmentation against Allen Atlas")
        
        # Generate morphogen segmentation if not available
        if self.morphogen_segmentation is None:
            self.generate_morphogen_segmentation()
        
        # Transform morphogen segmentation to atlas coordinates
        transformed_segmentation = self.coordinate_mapper.transform_morphogen_to_atlas(
            self.morphogen_segmentation, self.coordinate_transform)
        
        # Ensure same dimensions as atlas
        if transformed_segmentation.shape != self.atlas_reference.region_labels.shape:
            transformed_segmentation = self.coordinate_mapper._resize_to_match_atlas(
                transformed_segmentation, self.atlas_reference.region_labels.shape)
        
        # Calculate validation metrics
        validation_results = self.validation_metrics.calculate_comprehensive_validation(
            predicted=transformed_segmentation,
            reference=self.atlas_reference.region_labels
        )
        
        # Create comprehensive validation report
        validation_report = {
            "atlas_reference": {
                "atlas_id": self.atlas_reference.atlas_id,
                "developmental_stage": self.atlas_reference.developmental_stage,
                "resolution_um": self.atlas_reference.resolution_um,
                "dimensions": self.atlas_reference.dimensions
            },
            "morphogen_segmentation": {
                "grid_resolution_um": self.grid.dimensions.resolution,
                "grid_dimensions": (self.grid.dimensions.x_size, 
                                   self.grid.dimensions.y_size, 
                                   self.grid.dimensions.z_size),
                "unique_regions": len(np.unique(self.morphogen_segmentation)) - 1
            },
            "coordinate_transformation": {
                "scaling_factors": self.coordinate_transform.scaling_factors,
                "translation_offset": self.coordinate_transform.translation_offset,
                "transformation_applied": True
            },
            "validation_metrics": {
                metric_name: {
                    "metric_value": result.metric_value,
                    "target_threshold": result.target_threshold,
                    "validation_passed": result.validation_passed,
                    "overall_score": result.overall_score,
                    "region_scores": result.region_specific_scores
                }
                for metric_name, result in validation_results.items()
            },
            "overall_validation": {
                "all_metrics_passed": all(result.validation_passed for result in validation_results.values()),
                "average_score": np.mean([result.overall_score for result in validation_results.values()]),
                "dice_coefficient": validation_results["dice"].metric_value,
                "dice_threshold_met": validation_results["dice"].validation_passed
            }
        }
        
        # Log summary
        dice_score = validation_results["dice"].metric_value
        dice_passed = validation_results["dice"].validation_passed
        overall_passed = validation_report["overall_validation"]["all_metrics_passed"]
        
        logger.info(f"Atlas validation summary:")
        logger.info(f"  Dice coefficient: {dice_score:.3f} {'✅' if dice_passed else '❌'}")
        logger.info(f"  Overall validation: {'✅ PASS' if overall_passed else '❌ FAIL'}")
        
        return validation_report
    
    def export_validation_analysis(self) -> Dict[str, Any]:
        """Export complete validation analysis.
        
        Returns:
            Dictionary with comprehensive validation analysis
        """
        # Ensure atlas pipeline is setup
        if self.atlas_reference is None:
            pipeline_setup = self.setup_atlas_data_pipeline()
            if not pipeline_setup:
                return {"error": "Atlas pipeline setup failed"}
        
        # Run validation
        validation_report = self.validate_against_atlas()
        
        # Add system information
        complete_analysis = {
            "system_info": {
                "developmental_stage": self.developmental_stage,
                "validation_system_initialized": True,
                "atlas_data_available": self.atlas_reference is not None,
                "morphogen_segmentation_generated": self.morphogen_segmentation is not None
            },
            "validation_results": validation_report,
            "recommendations": self._generate_validation_recommendations(validation_report)
        }
        
        return complete_analysis
    
    def _generate_validation_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        dice_score = validation_report["overall_validation"]["dice_coefficient"]
        dice_threshold = 0.80
        
        if dice_score < dice_threshold:
            recommendations.append(f"Dice coefficient ({dice_score:.3f}) below threshold ({dice_threshold}). "
                                 "Consider parameter tuning for improved accuracy.")
        
        if not validation_report["overall_validation"]["all_metrics_passed"]:
            recommendations.append("Some validation metrics failed. Review regional segmentation quality.")
        
        if validation_report["overall_validation"]["average_score"] < 0.8:
            recommendations.append("Overall validation score low. Consider morphogen parameter optimization.")
        
        if not recommendations:
            recommendations.append("All validation metrics passed successfully. System ready for production.")
        
        return recommendations
    
    def _load_existing_atlas_reference(self) -> Optional[AtlasReference]:
        """Load atlas reference from existing downloaded data."""
        import json
        from pathlib import Path
        
        try:
            # Load integration manifest from a relative path
            manifest_file = self.data_downloader.data_dir / "atlas_integration_manifest.json"
            
            if not manifest_file.exists():
                logger.warning("Atlas integration manifest not found.")
                return None

            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            logger.info(f"Loaded existing atlas manifest: {manifest['total_datasets']} datasets, {manifest['total_size_mb']:.1f} MB")
            
            # This is a critical flaw. If the manifest exists, we should load the actual
            # processed data, not create a new synthetic reference.
            # This requires a more robust caching/loading mechanism than is present.
            # For now, we will fail if we can't properly load a real pre-processed atlas.
            logger.warning("Found manifest, but loading pre-processed atlas is not yet implemented. Failing.")
            return None
            
            logger.info("Successfully loaded existing atlas data for validation")
            
            return atlas_ref
            
        except Exception as e:
            logger.warning(f"Failed to load existing atlas data: {e}")
            return None
