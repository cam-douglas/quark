"""
Unit tests for the ventricular morphogenesis system.
"""

import unittest
import numpy as np

from brain.modules.morphogen_solver.spatial_grid import SpatialGrid, GridDimensions
from brain.modules.morphogen_solver.ventricular_topology import VentricularTopology
from brain.modules.morphogen_solver.ventricular_morphogenesis import VentricularMorphogenesis
from brain.modules.morphogen_solver.csf_flow_dynamics import CSFFlowDynamics

class TestVentricularMorphogenesis(unittest.TestCase):
    """
    Tests the components of the ventricular morphogenesis system.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        dims = GridDimensions(100, 100, 100, 1.0)
        self.grid = SpatialGrid(dims)
        self.topology = VentricularTopology(self.grid)

    def test_topology_initialization(self):
        """
        Test that the ventricular topology is initialized correctly.
        """
        self.assertIsNotNone(self.topology.get_lumen_mask())
        self.assertEqual(len(self.topology.ventricular_regions), 1)

    def test_morphogenesis_initialization(self):
        """
        Test that the ventricular morphogenesis can be initialized.
        """
        morphogenesis = VentricularMorphogenesis(self.grid, self.topology)
        self.assertIsNotNone(morphogenesis)

    def test_csf_flow_initialization(self):
        """
        Test that the CSF flow dynamics can be initialized.
        """
        csf_flow = CSFFlowDynamics(self.grid, self.topology)
        self.assertIsNotNone(csf_flow)
        lumen_mask = self.topology.get_lumen_mask()
        csf_flow.initialize_flow_domain(lumen_mask)
        self.assertIsNotNone(csf_flow.lumen_mask)

    def test_diffusion_step(self):
        """
        Test that a diffusion step can be run.
        """
        csf_flow = CSFFlowDynamics(self.grid, self.topology)
        lumen_mask = self.topology.get_lumen_mask()
        csf_flow.initialize_flow_domain(lumen_mask)
        csf_flow.run_diffusion_step(0.1)
        self.assertIsNotNone(csf_flow.concentration_field)

if __name__ == '__main__':
    unittest.main()