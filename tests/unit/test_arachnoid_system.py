import unittest
import numpy as np

from brain.modules.morphogen_solver.spatial_grid import SpatialGrid, GridDimensions
from brain.modules.morphogen_solver.ventricular_topology import VentricularTopology
from brain.modules.morphogen_solver.dura_mater_system import DuraMaterSystem
from brain.modules.morphogen_solver.arachnoid_system import ArachnoidSystem

class TestArachnoidSystem(unittest.TestCase):

    def setUp(self):
        # Setup mock objects for testing
        dims = GridDimensions(100, 100, 100, 1.0) # 100x100x100 grid, 1.0 um resolution
        self.grid = SpatialGrid(dims)
        self.topology = VentricularTopology(self.grid)
        self.dura_system = DuraMaterSystem(self.grid, self.topology)
        self.arachnoid_system = ArachnoidSystem(self.grid, self.topology, self.dura_system)

    def test_initialization(self):
        """Test arachnoid system initialization."""
        self.assertIsNotNone(self.arachnoid_system.arachnoid_layer)
        self.assertTrue(len(self.arachnoid_system.vascular_integration_points) > 0)
        self.assertEqual(self.arachnoid_system.arachnoid_layer.mechanical_properties.thickness_um, 20.0)

    def test_generate_trabecular_structure(self):
        """Test generation of the trabecular structure."""
        # Ensure dura surface is generated
        self.dura_system.generate_dura_surface_mesh()
        
        # Generate trabecular structure
        trabecular_mesh = self.arachnoid_system.generate_trabecular_structure()
        
        self.assertIsNotNone(trabecular_mesh)
        self.assertEqual(trabecular_mesh.shape, (100, 100, 100))
        self.assertTrue(np.sum(trabecular_mesh) > 0)

    def test_create_subarachnoid_space(self):
        """Test creation of the subarachnoid space."""
        # Ensure dura surface is generated
        self.dura_system.generate_dura_surface_mesh()

        # Create subarachnoid space
        subarachnoid_space = self.arachnoid_system.create_subarachnoid_space()

        self.assertIsNotNone(subarachnoid_space)
        self.assertEqual(subarachnoid_space.shape, (100, 100, 100))
        self.assertTrue(np.any(subarachnoid_space))

    def test_export_arachnoid_analysis(self):
        """Test comprehensive arachnoid analysis export."""
        # Generate necessary components first
        self.dura_system.generate_dura_surface_mesh()
        self.arachnoid_system.generate_trabecular_structure()
        self.arachnoid_system.create_subarachnoid_space()
        
        analysis = self.arachnoid_system.export_arachnoid_analysis()
        
        self.assertIn("developmental_stage", analysis)
        self.assertIn("geometry", analysis)
        self.assertIn("trabecular_connectivity", analysis)
        self.assertIn("vascular_integration", analysis)
        self.assertTrue(analysis["geometry"]["subarachnoid_volume_mm3"] > 0)
        self.assertTrue(analysis["geometry"]["trabecular_volume_mm3"] > 0)

if __name__ == '__main__':
    unittest.main()