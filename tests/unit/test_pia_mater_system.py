import unittest
import numpy as np

from brain.modules.morphogen_solver.spatial_grid import SpatialGrid, GridDimensions
from brain.modules.morphogen_solver.ventricular_topology import VentricularTopology
from brain.modules.morphogen_solver.dura_mater_system import DuraMaterSystem
from brain.modules.morphogen_solver.arachnoid_system import ArachnoidSystem
from brain.modules.morphogen_solver.pia_mater_system import PiaMaterSystem

class TestPiaMaterSystem(unittest.TestCase):

    def setUp(self):
        # Setup mock objects for testing
        dims = GridDimensions(100, 100, 100, 1.0)
        self.grid = SpatialGrid(dims)
        self.topology = VentricularTopology(self.grid)
        self.dura_system = DuraMaterSystem(self.grid, self.topology)
        self.arachnoid_system = ArachnoidSystem(self.grid, self.topology, self.dura_system)
        self.pia_system = PiaMaterSystem(self.grid, self.topology, self.arachnoid_system)

    def test_initialization(self):
        """Test pia mater system initialization."""
        self.assertIsNotNone(self.pia_system.pia_layer)
        self.assertEqual(self.pia_system.pia_layer.mechanical_properties.thickness_um, 5.0)

    def test_generate_neural_interface(self):
        """Test generation of the neural interface."""
        neural_interface = self.pia_system.generate_neural_interface()
        self.assertIsNotNone(neural_interface)
        self.assertEqual(neural_interface.shape, (100, 100, 100))
        self.assertTrue(np.any(neural_interface))

    def test_establish_blood_vessel_pathways(self):
        """Test establishment of blood vessel pathways."""
        vascular_pathways = self.pia_system.establish_blood_vessel_pathways()
        self.assertIsNotNone(vascular_pathways)
        self.assertTrue(len(vascular_pathways) > 0)
        self.assertIn('pia_vessel_1', vascular_pathways)

    def test_validate_metabolic_exchange(self):
        """Test validation of metabolic exchange."""
        metabolic_metrics = self.pia_system.validate_metabolic_exchange()
        self.assertIsNotNone(metabolic_metrics)
        self.assertGreater(metabolic_metrics['interface_area_mm2'], 0)
        self.assertGreater(metabolic_metrics['exchange_capacity'], 0)

    def test_export_pia_analysis(self):
        """Test comprehensive pia mater analysis export."""
        analysis = self.pia_system.export_pia_analysis()
        self.assertIn('developmental_stage', analysis)
        self.assertIn('geometry', analysis)
        self.assertIn('metabolic_exchange', analysis)
        self.assertGreater(analysis['geometry']['volume_mm3'], 0)

if __name__ == '__main__':
    unittest.main()