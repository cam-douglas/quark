import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
import json

from brain.modules.morphogen_solver.atlas_validation_system import AtlasValidationSystem
from brain.modules.morphogen_solver.spatial_grid import SpatialGrid, GridDimensions
from brain.modules.morphogen_solver.cell_fate_specifier import CellFateSpecifier
from brain.modules.morphogen_solver.atlas_validation_types import AtlasReference, CoordinateSystem

class TestAtlasValidationSystem(unittest.TestCase):
    """Integration tests for the AtlasValidationSystem."""

    def setUp(self):
        """Set up a mock environment for testing."""
        self.grid_dims = GridDimensions(x_size=64, y_size=96, z_size=64, resolution=10.0)
        self.grid = SpatialGrid(self.grid_dims)
        self.grid.add_morphogen('SHH')
        
        # Mock CellFateSpecifier
        self.cell_fate_specifier = MagicMock(spec=CellFateSpecifier)
        # Mock the output of specify_cell_fates to return a plausible map
        mock_fate_map = {'floor_plate': np.random.rand(64, 96, 64) > 0.9}
        self.cell_fate_specifier.specify_cell_fates.return_value = mock_fate_map

        self.validation_system = AtlasValidationSystem(self.grid, self.cell_fate_specifier)

        # Create a dummy manifest file for testing the loading logic
        self.data_dir = self.validation_system.data_downloader.data_dir
        self.manifest_path = self.data_dir / "atlas_integration_manifest.json"
        with open(self.manifest_path, 'w') as f:
            json.dump({"total_datasets": 1, "total_size_mb": 10}, f)

    def tearDown(self):
        """Clean up dummy files."""
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    @patch('brain.modules.morphogen_solver.atlas_data_downloader.AtlasDataDownloader.integrate_brainspan_with_allen')
    def test_pipeline_fails_gracefully_without_real_data(self, mock_integrate):
        """Test that the system fails explicitly if real atlas data cannot be loaded."""
        # Configure the mock to simulate a download/integration failure
        mock_integrate.return_value = None

        # Attempt to set up the pipeline
        success = self.validation_system.setup_atlas_data_pipeline()

        # Assert that the setup failed as expected
        self.assertFalse(success)
        self.assertIsNone(self.validation_system.atlas_reference)

    @patch('brain.modules.morphogen_solver.atlas_data_downloader.AtlasDataDownloader.validate_atlas_integrity')
    @patch('brain.modules.morphogen_solver.atlas_data_downloader.AtlasDataDownloader.integrate_brainspan_with_allen')
    def test_full_validation_run_with_mock_data(self, mock_integrate, mock_validate_integrity):
        """Test the full end-to-end validation process with mock data."""
        # Ensure the manifest does not exist for this test to force the download path
        if self.manifest_path.exists():
            self.manifest_path.unlink()

        # 1. Setup Mock AtlasReference
        mock_atlas_dims = (64, 96, 64)
        mock_atlas_labels = np.zeros(mock_atlas_dims, dtype=int)
        mock_atlas_labels[:, :48, :] = 1 # Mock forebrain
        mock_atlas_labels[:, 48:, :] = 3 # Mock hindbrain
        
        mock_atlas = AtlasReference(
            atlas_id='mock_atlas',
            developmental_stage='E10.5',
            coordinate_system=CoordinateSystem.ALLEN_CCF,
            resolution_um=10.0,
            dimensions=mock_atlas_dims,
            region_labels=mock_atlas_labels,
            region_names={1: "forebrain", 3: "hindbrain"},
            reference_url='mock://url'
        )
        mock_integrate.return_value = mock_atlas
        mock_validate_integrity.return_value = {"overall_valid": True}

        # 2. Run the pipeline
        pipeline_success = self.validation_system.setup_atlas_data_pipeline()
        self.assertTrue(pipeline_success)
        
        # 3. Run validation
        report = self.validation_system.validate_against_atlas()

        # 4. Assertions on the report
        self.assertIn("validation_metrics", report)
        self.assertIn("dice", report["validation_metrics"])
        self.assertIn("hausdorff", report["validation_metrics"])
        self.assertIn("jaccard", report["validation_metrics"])
        
        # Check that a dice score was calculated (even if it's low)
        dice_score = report["validation_metrics"]["dice"]["metric_value"]
        self.assertIsInstance(dice_score, float)
        self.assertGreaterEqual(dice_score, 0.0)
        self.assertLessEqual(dice_score, 1.0)
        
        # Check overall validation status
        self.assertIn("overall_validation", report)
        self.assertIn("average_score", report["overall_validation"])

if __name__ == '__main__':
    unittest.main()