"""
Unit tests for the dura mater system.
"""

import unittest
import numpy as np

from brain.modules.morphogen_solver.spatial_grid import SpatialGrid, GridDimensions
from brain.modules.morphogen_solver.ventricular_topology import VentricularTopology
from brain.modules.morphogen_solver.dura_mater_system import DuraMaterSystem

class TestDuraMaterSystem(unittest.TestCase):
    """
    Tests the components of the dura mater system.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        dims = GridDimensions(100, 100, 100, 1.0)
        self.grid = SpatialGrid(dims)
        self.topology = VentricularTopology(self.grid)
        self.dura_system = DuraMaterSystem(self.grid, self.topology)

    def test_dura_mater_initialization(self):
        """
        Test that the dura mater system is initialized correctly.
        """
        self.assertIsNotNone(self.dura_system.dura_layer)

    def test_generate_dura_surface_mesh(self):
        """
        Test that a dura surface mesh can be generated.
        """
        mesh = self.dura_system.generate_dura_surface_mesh()
        self.assertIsNotNone(mesh)
        self.assertEqual(mesh.shape, (100, 100, 100))

    def test_compute_stress_distribution(self):
        """
        Test that a stress distribution can be computed.
        """
        stress_field = self.dura_system.compute_stress_distribution()
        self.assertIsNotNone(stress_field)
        self.assertEqual(stress_field.shape, (100, 100, 100))

if __name__ == '__main__':
    unittest.main()