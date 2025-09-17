#!/usr/bin/env python3
"""
Embryonic Simulation Engine with Brainstem Segmentation Integration

Enhanced embryonic simulation engine that automatically performs brainstem
segmentation during startup using the trained ViT-GNN model.

Phase 4 Step 2.O2: Wire inference into voxel-map builder to auto-segment on embryo simulation start-up

Author: Quark AI
Date: 2025-09-16
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from .embryonic_simulation_engine import EmbryonicSimulationEngine
from .neuroepithelial_cells import NeuroepithelialCell
from .neuroepithelial_cell_types import NeuroepithelialCellType
from .committed_progenitor_generator import CommittedProgenitorGenerator
from .apoptosis_manager import ApoptosisManager

# Import brainstem segmentation components
try:
    from ..brainstem_segmentation.inference_engine import (
        BrainstemInferenceEngine,
        InferenceConfig,
        auto_segment_brainstem
    )
    BRAINSTEM_SEGMENTATION_AVAILABLE = True
except ImportError:
    BRAINSTEM_SEGMENTATION_AVAILABLE = False
    logging.warning("Brainstem segmentation module not available")

logger = logging.getLogger(__name__)


class EmbryonicSimulationEngineWithSegmentation(EmbryonicSimulationEngine):
    """
    Enhanced embryonic simulation engine with automatic brainstem segmentation.

    This engine extends the base embryonic simulation to automatically
    perform brainstem segmentation when voxel maps are initialized.
    """

    def __init__(self):
        """Initialize simulation engine with brainstem segmentation capability."""
        super().__init__()

        # Initialize brainstem segmentation components
        self.brainstem_engine = None
        self.segmentation_results = None

        if BRAINSTEM_SEGMENTATION_AVAILABLE:
            self._initialize_brainstem_segmentation()
        else:
            logger.warning("Brainstem segmentation not available - running without segmentation")

    def _initialize_brainstem_segmentation(self):
        """Initialize brainstem segmentation engine."""
        try:
            config = InferenceConfig()
            self.brainstem_engine = BrainstemInferenceEngine(config)
            logger.info("Brainstem segmentation engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize brainstem segmentation: {e}")
            self.brainstem_engine = None

    def run_simulation(self,
                      initial_cell_count: int,
                      simulation_duration: float,
                      time_step: float,
                      voxel_map_data: Optional[np.ndarray] = None,
                      morphogen_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run embryonic development simulation with automatic brainstem segmentation.

        Args:
            initial_cell_count: Number of initial cells
            simulation_duration: Duration in hours
            time_step: Time step in hours
            voxel_map_data: Optional 3D voxel map data for segmentation
            morphogen_data: Optional morphogen concentration data

        Returns:
            Simulation results including segmentation data
        """
        logger.info("🚀 Starting embryonic simulation with brainstem segmentation")

        # Run base simulation
        results = super().run_simulation(
            initial_cell_count=initial_cell_count,
            simulation_duration=simulation_duration,
            time_step=time_step
        )

        # Perform automatic brainstem segmentation if data available
        if voxel_map_data is not None and self.brainstem_engine is not None:
            logger.info("🧠 Performing automatic brainstem segmentation...")

            try:
                # Segment the voxel map
                segmentation = self.brainstem_engine.segment_volume(
                    voxel_map_data,
                    morphogen_data
                )

                if segmentation is not None:
                    # Extract anatomical regions
                    regions = self.brainstem_engine.get_brainstem_regions(segmentation)

                    # Get segmentation statistics
                    stats = self.brainstem_engine.get_segmentation_stats(segmentation)

                    # Store segmentation results
                    self.segmentation_results = {
                        'segmentation_mask': segmentation,
                        'anatomical_regions': regions,
                        'statistics': stats,
                        'timestamp': time.time(),
                        'voxel_map_shape': voxel_map_data.shape
                    }

                    # Add segmentation results to simulation results
                    results.update({
                        'brainstem_segmentation': self.segmentation_results,
                        'segmentation_success': True
                    })

                    logger.info("✅ Brainstem segmentation completed successfully")
                    logger.info(f"   📊 Segmentation coverage: {stats['segmentation_coverage']:.1%}")
                    logger.info(f"   🏗️  Detected regions: {stats['num_regions']}")

                    # Log key anatomical regions
                    for region_name, mask in regions.items():
                        voxel_count = np.sum(mask)
                        if voxel_count > 0:
                            logger.info(f"   📍 {region_name}: {voxel_count:,} voxels")

                else:
                    logger.warning("❌ Brainstem segmentation failed")
                    results['segmentation_success'] = False

            except Exception as e:
                logger.error(f"❌ Brainstem segmentation error: {e}")
                results['segmentation_success'] = False
                results['segmentation_error'] = str(e)

        elif voxel_map_data is None:
            logger.info("ℹ️  No voxel map data provided - skipping brainstem segmentation")
            results['segmentation_success'] = False
            results['segmentation_reason'] = 'no_voxel_map_data'

        elif self.brainstem_engine is None:
            logger.warning("⚠️  Brainstem segmentation engine not available")
            results['segmentation_success'] = False
            results['segmentation_reason'] = 'engine_not_available'

        return results

    def get_segmentation_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest brainstem segmentation results.

        Returns:
            Dictionary containing segmentation mask, regions, and statistics
        """
        return self.segmentation_results

    def get_brainstem_region_mask(self, region_name: str) -> Optional[np.ndarray]:
        """
        Get binary mask for a specific brainstem region.

        Args:
            region_name: Name of the anatomical region

        Returns:
            Binary mask array or None if region not found
        """
        if self.segmentation_results is None:
            return None

        regions = self.segmentation_results.get('anatomical_regions', {})
        return regions.get(region_name)

    def get_segmentation_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get segmentation statistics.

        Returns:
            Dictionary with segmentation statistics or None
        """
        if self.segmentation_results is None:
            return None

        return self.segmentation_results.get('statistics')

    def save_segmentation_results(self, output_dir: str = "/data/models/brainstem_segmentation/results") -> bool:
        """
        Save segmentation results to disk.

        Args:
            output_dir: Directory to save results

        Returns:
            True if successful, False otherwise
        """
        if self.segmentation_results is None:
            logger.warning("No segmentation results to save")
            return False

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save segmentation mask as numpy array
            mask_file = output_path / "latest_segmentation_mask.npy"
            np.save(mask_file, self.segmentation_results['segmentation_mask'])

            # Save regions as separate files
            regions_dir = output_path / "anatomical_regions"
            regions_dir.mkdir(exist_ok=True)

            for region_name, mask in self.segmentation_results['anatomical_regions'].items():
                region_file = regions_dir / f"{region_name}_mask.npy"
                np.save(region_file, mask)

            # Save statistics as JSON
            import json
            stats_file = output_path / "segmentation_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(self.segmentation_results['statistics'], f, indent=2)

            # Save metadata
            metadata = {
                'timestamp': self.segmentation_results['timestamp'],
                'voxel_map_shape': self.segmentation_results['voxel_map_shape'],
                'generation_time': time.time()
            }

            metadata_file = output_path / "segmentation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Segmentation results saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save segmentation results: {e}")
            return False


def create_segmented_embryonic_simulation(voxel_map_data: Optional[np.ndarray] = None,
                                         morphogen_data: Optional[np.ndarray] = None) -> EmbryonicSimulationEngineWithSegmentation:
    """
    Factory function to create an embryonic simulation engine with brainstem segmentation.

    Args:
        voxel_map_data: Optional 3D voxel map data for segmentation
        morphogen_data: Optional morphogen concentration data

    Returns:
        Configured embryonic simulation engine with segmentation capability
    """
    engine = EmbryonicSimulationEngineWithSegmentation()

    # Store voxel map data for segmentation during simulation
    if voxel_map_data is not None:
        engine.voxel_map_data = voxel_map_data
    if morphogen_data is not None:
        engine.morphogen_data = morphogen_data

    return engine
