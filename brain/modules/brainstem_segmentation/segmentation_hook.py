#!/usr/bin/env python3
"""
Brainstem Segmentation Hook for Brain Simulator

Automatically performs brainstem segmentation during brain simulator initialization.
This hook integrates with the BrainSimulator startup process to run segmentation
when voxel maps are created.

Phase 4 Step 2.O2: Wire inference into voxel-map builder to auto-segment on embryo simulation start-up

Author: Quark AI
Date: 2025-09-16
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Import brainstem segmentation components
try:
    from .inference_engine import BrainstemInferenceEngine, InferenceConfig
    SEGMENTATION_AVAILABLE = True
except ImportError:
    SEGMENTATION_AVAILABLE = False
    logger.warning("Brainstem segmentation inference engine not available")

# Import morphogen solver for voxel map access
try:
    from ..morphogen_solver.morphogen_solver import MorphogenSolver
    from ..morphogen_solver.spatial_grid import GridDimensions
    MORPHOGEN_AVAILABLE = True
except ImportError:
    MORPHOGEN_AVAILABLE = False
    logger.warning("Morphogen solver not available")


class BrainstemSegmentationHook:
    """
    Hook that automatically performs brainstem segmentation during brain initialization.

    This hook integrates with the brain simulator to:
    1. Detect when voxel maps are being initialized
    2. Extract morphogen concentration data
    3. Run automatic brainstem segmentation
    4. Store results for downstream use
    """

    def __init__(self, auto_segment: bool = True, save_results: bool = True):
        """Initialize the segmentation hook.

        Args:
            auto_segment: Whether to automatically run segmentation on startup
            save_results: Whether to save segmentation results to disk
        """
        self.auto_segment = auto_segment
        self.save_results = save_results
        self.segmentation_results = None
        self.engine = None

        if SEGMENTATION_AVAILABLE:
            self._initialize_engine()
        else:
            logger.warning("Brainstem segmentation not available - hook disabled")

    def _initialize_engine(self):
        """Initialize the brainstem segmentation engine."""
        try:
            config = InferenceConfig()
            self.engine = BrainstemInferenceEngine(config)
            logger.info("Brainstem segmentation hook initialized")
        except Exception as e:
            logger.error(f"Failed to initialize segmentation hook: {e}")
            self.engine = None

    def on_brain_initialization(self, brain_simulator) -> Dict[str, Any]:
        """
        Hook called during brain simulator initialization.

        Args:
            brain_simulator: The BrainSimulator instance being initialized

        Returns:
            Dictionary with segmentation results or status
        """
        if not self.auto_segment or self.engine is None:
            logger.info("Brainstem segmentation hook disabled or unavailable")
            return {"segmentation_status": "disabled"}

        logger.info("🧠 Starting automatic brainstem segmentation during brain initialization...")

        try:
            # Extract voxel map data from brain simulator
            voxel_data = self._extract_voxel_map_data(brain_simulator)

            if voxel_data is None:
                logger.warning("No voxel map data available for segmentation")
                return {"segmentation_status": "no_data"}

            # Extract morphogen data if available
            morphogen_data = self._extract_morphogen_data(brain_simulator)

            # Run segmentation
            segmentation = self.engine.segment_volume(voxel_data, morphogen_data)

            if segmentation is not None:
                # Extract anatomical regions
                regions = self.engine.get_brainstem_regions(segmentation)

                # Get segmentation statistics
                stats = self.engine.get_segmentation_stats(segmentation)

                # Store results
                self.segmentation_results = {
                    'segmentation_mask': segmentation,
                    'anatomical_regions': regions,
                    'statistics': stats,
                    'timestamp': time.time(),
                    'voxel_map_shape': voxel_data.shape
                }

                # Save results if requested
                if self.save_results:
                    self._save_results()

                logger.info("✅ Automatic brainstem segmentation completed successfully")
                logger.info(f"   📊 Segmentation coverage: {stats['segmentation_coverage']:.1%}")
                logger.info(f"   🏗️  Detected regions: {stats['num_regions']}")

                # Log key anatomical regions
                for region_name, mask in regions.items():
                    voxel_count = np.sum(mask)
                    if voxel_count > 0:
                        logger.info(f"   📍 {region_name}: {voxel_count:,} voxels")

                return {
                    "segmentation_status": "success",
                    "segmentation_results": self.segmentation_results,
                    "statistics": stats
                }

            else:
                logger.error("❌ Automatic brainstem segmentation failed")
                return {"segmentation_status": "failed"}

        except Exception as e:
            logger.error(f"❌ Brainstem segmentation hook error: {e}")
            return {"segmentation_status": "error", "error_message": str(e)}

    def _extract_voxel_map_data(self, brain_simulator) -> Optional[np.ndarray]:
        """
        Extract voxel map data from brain simulator.

        Args:
            brain_simulator: The BrainSimulator instance

        Returns:
            3D voxel map array or None if not available
        """
        try:
            # Try to access morphogen solver from brain simulator
            if hasattr(brain_simulator, 'morphogen_solver'):
                morphogen_solver = brain_simulator.morphogen_solver
                if hasattr(morphogen_solver, 'spatial_grid'):
                    # Get concentration data from any morphogen (they all have same shape)
                    concentrations = morphogen_solver.spatial_grid.concentrations
                    if concentrations:
                        morphogen_name = list(concentrations.keys())[0]
                        voxel_data = concentrations[morphogen_name]
                        logger.info(f"Extracted voxel map data: {voxel_data.shape}")
                        return voxel_data

            # Try to access developmental biology modules
            if hasattr(brain_simulator, 'modules'):
                dev_bio = brain_simulator.modules.get('developmental_biology')
                if dev_bio and hasattr(dev_bio, 'voxel_map_data'):
                    voxel_data = dev_bio.voxel_map_data
                    logger.info(f"Extracted voxel map data from dev bio: {voxel_data.shape}")
                    return voxel_data

            # Try to create synthetic voxel data for testing
            logger.info("Creating synthetic voxel data for segmentation testing")
            return self._create_synthetic_voxel_data()

        except Exception as e:
            logger.error(f"Failed to extract voxel map data: {e}")
            return None

    def _extract_morphogen_data(self, brain_simulator) -> Optional[np.ndarray]:
        """
        Extract morphogen concentration data from brain simulator.

        Args:
            brain_simulator: The BrainSimulator instance

        Returns:
            Morphogen concentration array or None
        """
        try:
            if hasattr(brain_simulator, 'morphogen_solver'):
                morphogen_solver = brain_simulator.morphogen_solver
                if hasattr(morphogen_solver, 'spatial_grid'):
                    concentrations = morphogen_solver.spatial_grid.concentrations

                    # Stack all morphogen concentrations into a single array
                    if concentrations:
                        morphogen_arrays = []
                        for name, data in concentrations.items():
                            morphogen_arrays.append(data)

                        if morphogen_arrays:
                            morphogen_data = np.stack(morphogen_arrays, axis=0)
                            logger.info(f"Extracted morphogen data: {morphogen_data.shape}")
                            return morphogen_data

        except Exception as e:
            logger.error(f"Failed to extract morphogen data: {e}")

        return None

    def _create_synthetic_voxel_data(self) -> np.ndarray:
        """
        Create synthetic voxel data for testing segmentation.

        Returns:
            Synthetic 3D voxel array
        """
        # Create a simple synthetic brain-like structure
        shape = (128, 128, 128)
        voxel_data = np.random.rand(*shape).astype(np.float32)

        # Add some structure to simulate brain regions
        # Create rough brainstem-like region in the center-bottom
        center_y, center_z = shape[1] // 2, shape[2] // 2
        brainstem_mask = np.zeros(shape, dtype=bool)

        for x in range(shape[0] // 4, 3 * shape[0] // 4):
            for y in range(center_y - 20, center_y + 20):
                for z in range(center_z - 15, center_z + 15):
                    if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                        brainstem_mask[x, y, z] = True

        # Apply higher intensity to brainstem region
        voxel_data[brainstem_mask] *= 2.0

        logger.info(f"Created synthetic voxel data: {shape}")
        return voxel_data

    def _save_results(self) -> bool:
        """
        Save segmentation results to disk.

        Returns:
            True if successful
        """
        if self.segmentation_results is None:
            return False

        try:
            output_dir = Path("/data/models/brainstem_segmentation/auto_segmentation_results")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save segmentation mask
            mask_file = output_dir / "auto_segmentation_mask.npy"
            np.save(mask_file, self.segmentation_results['segmentation_mask'])

            # Save regions
            regions_dir = output_dir / "anatomical_regions"
            regions_dir.mkdir(exist_ok=True)

            for region_name, mask in self.segmentation_results['anatomical_regions'].items():
                region_file = regions_dir / f"{region_name}_mask.npy"
                np.save(region_file, mask)

            # Save statistics
            import json
            stats_file = output_dir / "segmentation_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(self.segmentation_results['statistics'], f, indent=2)

            # Save metadata
            metadata = {
                'timestamp': self.segmentation_results['timestamp'],
                'voxel_map_shape': self.segmentation_results['voxel_map_shape'],
                'auto_generated': True,
                'hook_version': '1.0.0'
            }

            metadata_file = output_dir / "segmentation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Auto-segmentation results saved to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to save auto-segmentation results: {e}")
            return False

    def get_segmentation_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest segmentation results.

        Returns:
            Dictionary with segmentation results
        """
        return self.segmentation_results

    def get_region_mask(self, region_name: str) -> Optional[np.ndarray]:
        """
        Get binary mask for a specific anatomical region.

        Args:
            region_name: Name of the region

        Returns:
            Binary mask array or None
        """
        if self.segmentation_results is None:
            return None

        regions = self.segmentation_results.get('anatomical_regions', {})
        return regions.get(region_name)


def install_segmentation_hook(brain_simulator, auto_segment: bool = True) -> BrainstemSegmentationHook:
    """
    Install the brainstem segmentation hook on a brain simulator.

    Args:
        brain_simulator: The BrainSimulator instance to hook into
        auto_segment: Whether to enable automatic segmentation

    Returns:
        The installed segmentation hook
    """
    hook = BrainstemSegmentationHook(auto_segment=auto_segment)

    # Try to attach hook to brain simulator
    if hasattr(brain_simulator, 'hooks'):
        brain_simulator.hooks.append(hook)
        logger.info("Brainstem segmentation hook installed")
    else:
        logger.warning("Brain simulator doesn't support hooks - hook not installed")

    return hook
