"""
Utilities shared by segmentation hook modules.

Keeps `segmentation_hook.py` concise by factoring helpers out.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import nibabel as nib


def save_segmentation_results(output_dir: Path, results: Dict[str, Any]) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_file = output_dir / "auto_segmentation_mask.npy"
    np.save(mask_file, results["segmentation_mask"])  # type: ignore[index]

    regions_dir = output_dir / "anatomical_regions"
    regions_dir.mkdir(exist_ok=True)
    for region_name, mask in results.get("anatomical_regions", {}).items():
        region_file = regions_dir / f"{region_name}_mask.npy"
        np.save(region_file, mask)

    stats_file = output_dir / "segmentation_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(results.get("statistics", {}), f, indent=2)

    metadata = {
        "timestamp": results.get("timestamp"),
        "voxel_map_shape": results.get("voxel_map_shape"),
        "auto_generated": True,
        "hook_version": "1.0.0",
    }
    metadata_file = output_dir / "segmentation_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return True


def extract_voxel_map_data(brain_simulator) -> Optional[np.ndarray]:
    try:
        if hasattr(brain_simulator, "morphogen_solver"):
            morphogen_solver = brain_simulator.morphogen_solver
            if hasattr(morphogen_solver, "spatial_grid"):
                concentrations = morphogen_solver.spatial_grid.concentrations
                if concentrations:
                    morphogen_name = list(concentrations.keys())[0]
                    voxel_data = concentrations[morphogen_name]
                    return voxel_data

        if hasattr(brain_simulator, "modules"):
            dev_bio = brain_simulator.modules.get("developmental_biology")
            if dev_bio and hasattr(dev_bio, "voxel_map_data"):
                return dev_bio.voxel_map_data

        return create_synthetic_voxel_data()
    except Exception:
        return None


def extract_morphogen_data(brain_simulator) -> Optional[np.ndarray]:
    try:
        if hasattr(brain_simulator, "morphogen_solver"):
            morphogen_solver = brain_simulator.morphogen_solver
            if hasattr(morphogen_solver, "spatial_grid"):
                concentrations = morphogen_solver.spatial_grid.concentrations
                if concentrations:
                    arrays = [arr for _, arr in concentrations.items()]
                    if arrays:
                        return np.stack(arrays, axis=0)
    except Exception:
        return None
    return None


def create_synthetic_voxel_data() -> np.ndarray:
    shape = (128, 128, 128)
    voxel_data = np.random.rand(*shape).astype(np.float32)

    center_y, center_z = shape[1] // 2, shape[2] // 2
    brainstem_mask = np.zeros(shape, dtype=bool)
    for x in range(shape[0] // 4, 3 * shape[0] // 4):
        for y in range(center_y - 20, center_y + 20):
            for z in range(center_z - 15, center_z + 15):
                if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                    brainstem_mask[x, y, z] = True

    voxel_data[brainstem_mask] *= 2.0
    return voxel_data


def load_spotcheck_labels(base_dir: Path) -> Optional[np.ndarray]:
    """Load manual spot-check labels if available.

    Attempts typical paths under data/datasets/brainstem_segmentation.
    """
    candidates = [
        base_dir / "data" / "datasets" / "brainstem_segmentation" / "test_splits" / "test_manual.nii.gz",
        base_dir / "data" / "datasets" / "brainstem_segmentation" / "qa" / "test_manual_adjudicated.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            img = nib.load(str(path))
            return img.get_fdata().astype(np.int32)
    return None


