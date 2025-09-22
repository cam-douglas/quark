import unittest
import torch
import numpy as np

from brain.modules.morphogen_solver.gnn_vit_hybrid import GNNViTHybridModel
from brain.modules.morphogen_solver.spatial_grid import SpatialGrid, GridDimensions
from brain.modules.morphogen_solver.gnn_spatial_graph import SpatialGraphConstructor

class TestGNNViTHybridModel(unittest.TestCase):
    """Integration tests for the GNNViTHybridModel."""

    def setUp(self):
        """Set up a mock environment and model for testing."""
        self.input_resolution = 32  # Smaller for faster testing
        self.patch_size = 4
        self.num_classes = 5
        self.input_channels = 4
        self.batch_size = 2

        self.model = GNNViTHybridModel(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            vit_embed_dim=192,  # Must be divisible by num_heads (default 12)
            gnn_hidden_dim=64,
            fusion_dim=64,
            input_resolution=self.input_resolution
        )

        # Mock SpatialGrid for Graph Constructor
        dims = GridDimensions(x_size=self.input_resolution, y_size=self.input_resolution, z_size=self.input_resolution, resolution=1.0)
        grid = SpatialGrid(dims)
        self.graph_constructor = SpatialGraphConstructor(grid, connectivity_radius=2.0)
        
        # Assign the robust graph constructor to the model
        self.model.graph_constructor = self.graph_constructor


    def test_forward_pass_with_external_graph(self):
        """Test the forward pass using a pre-computed graph."""
        # 1. Create realistic input data
        input_tensor = torch.randn(
            self.batch_size, self.input_channels, self.input_resolution, self.input_resolution, self.input_resolution
        )

        # 2. Pre-compute graph data for each item in the batch
        graph_data_list = []
        for i in range(self.batch_size):
            sample_np = input_tensor[i].numpy()
            morphogen_concentrations = {
                'SHH': sample_np[0],
                'BMP': sample_np[1],
                'WNT': sample_np[2],
                'FGF': sample_np[3],
            }
            graph = self.graph_constructor.construct_spatial_graph(
                morphogen_concentrations,
                downsample_factor=self.patch_size 
            )
            graph_data_list.append(graph)
            
        # Collate batch of graphs
        graph_data_batched = {
            "node_features": [g["node_features"] for g in graph_data_list],
            "edge_index": [g["edge_index"] for g in graph_data_list],
            "edge_features": [g["edge_features"] for g in graph_data_list],
        }

        # 3. Run forward pass
        outputs = self.model(input_tensor, graph_data=graph_data_batched)

        # 4. Assertions to validate output shapes
        self.assertIn("segmentation_logits", outputs)
        self.assertIsInstance(outputs["segmentation_logits"], list)
        self.assertEqual(len(outputs["segmentation_logits"]), self.batch_size)

        num_nodes_per_sample = (self.input_resolution // self.patch_size)**3
        
        for i in range(self.batch_size):
            # Check segmentation logits shape
            logits = outputs["segmentation_logits"][i]
            self.assertEqual(logits.shape[0], graph_data_list[i]['num_nodes'])
            self.assertEqual(logits.shape[1], self.num_classes)

            # Check feature shapes
            self.assertEqual(outputs['gnn_features'][i].shape[0], graph_data_list[i]['num_nodes'])
            self.assertEqual(outputs['fused_features'][i].shape[0], graph_data_list[i]['num_nodes'])
        
        vit_num_patches = (self.input_resolution // self.model.vit_encoder.patch_embed.patch_size)**3
        self.assertEqual(outputs['vit_patch_features'].shape, (self.batch_size, vit_num_patches, self.model.vit_encoder.embed_dim))


    def test_forward_pass_without_graph(self):
        """Test the forward pass using the internal simplified graph creation."""
        input_tensor = torch.randn(
            self.batch_size, self.input_channels, self.input_resolution, self.input_resolution, self.input_resolution
        )
        # Detach the robust graph constructor to force use of the internal one
        self.model.graph_constructor = None
        
        outputs = self.model(input_tensor)

        self.assertIn("segmentation_logits", outputs)
        self.assertEqual(len(outputs["segmentation_logits"]), self.batch_size)
        
        # The internal creator has its own downsample factor
        internal_downsample_factor = 4
        num_nodes = (self.input_resolution // internal_downsample_factor)**3

        for i in range(self.batch_size):
            logits = outputs["segmentation_logits"][i]
            self.assertEqual(logits.shape, (num_nodes, self.num_classes))


if __name__ == '__main__':
    unittest.main()