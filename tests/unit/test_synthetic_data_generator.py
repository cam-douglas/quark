import unittest
import numpy as np

from brain.modules.morphogen_solver.spatial_grid import GridDimensions
from brain.modules.morphogen_solver.ml_diffusion_types import SyntheticDataConfig
from brain.modules.morphogen_solver.synthetic_data_generator import SyntheticEmbryoDataGenerator

class TestSyntheticEmbryoDataGenerator(unittest.TestCase):
    """Unit tests for the SyntheticEmbryoDataGenerator."""

    def setUp(self):
        """Set up test environment."""
        self.grid_dims = GridDimensions(x_size=32, y_size=32, z_size=32, resolution=1.0)
        self.data_config = SyntheticDataConfig(
            num_samples=2,
            noise_level=0.05,
            biological_constraints=True
        )
        self.generator = SyntheticEmbryoDataGenerator(self.grid_dims, self.data_config)

    def test_generate_synthetic_dataset(self):
        """Test the generation of a full synthetic dataset."""
        dataset = self.generator.generate_synthetic_dataset()

        # Check dataset structure
        self.assertIn("morphogen_concentrations", dataset)
        self.assertIn("parameters", dataset)
        self.assertIn("metadata", dataset)

        # Check concentrations shape
        concentrations = dataset["morphogen_concentrations"]
        expected_shape = (self.data_config.num_samples, 4, 
                          self.grid_dims.x_size, self.grid_dims.y_size, self.grid_dims.z_size)
        self.assertEqual(concentrations.shape, expected_shape)

        # Check that the fields are not empty or uniform
        # This confirms the simulations ran for all morphogens
        for i in range(4): # SHH, BMP, WNT, FGF
            morphogen_field = concentrations[0, i, :, :, :]
            self.assertFalse(np.all(morphogen_field == 0), f"Morphogen {i} field is all zeros.")
            # Check that there is some variation
            self.assertGreater(np.std(morphogen_field), 0.0, f"Morphogen {i} field has no variation.")

    def test_train_val_split(self):
        """Test the training and validation set splitting."""
        # Generate a slightly larger dataset for splitting
        self.data_config.num_samples = 10
        self.generator = SyntheticEmbryoDataGenerator(self.grid_dims, self.data_config)
        dataset = self.generator.generate_synthetic_dataset()
        
        train_set, val_set = self.generator.create_train_val_split(dataset, val_fraction=0.2)
        
        # Check sizes
        self.assertEqual(train_set["morphogen_concentrations"].shape[0], 8)
        self.assertEqual(val_set["morphogen_concentrations"].shape[0], 2)
        self.assertEqual(len(train_set["parameters"]), 8)
        self.assertEqual(len(val_set["parameters"]), 2)

if __name__ == '__main__':
    unittest.main()