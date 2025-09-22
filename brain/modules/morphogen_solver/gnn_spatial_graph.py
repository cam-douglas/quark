#!/usr/bin/env python3
"""Graph Neural Network Spatial Graph System.

Implements spatial connectivity graphs for morphogen concentration data
with node features and edge relationships for regional boundary prediction
in the GNN-ViT hybrid segmentation system.

Integration: GNN component for hybrid segmentation system
Rationale: Focused spatial graph construction for GNN processing
"""

from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import logging

from .spatial_grid import SpatialGrid

logger = logging.getLogger(__name__)

class SpatialGraphConstructor:
    """Constructor for spatial connectivity graphs from morphogen data.
    
    Creates graph representations of morphogen concentration fields
    with spatial connectivity and morphogen-based node features
    for GNN processing.
    """
    
    def __init__(self, spatial_grid: SpatialGrid, 
                 connectivity_radius: float = 10.0):
        """Initialize spatial graph constructor.
        
        Args:
            spatial_grid: 3D spatial grid with morphogen data
            connectivity_radius: Radius for spatial connectivity (µm)
        """
        self.grid = spatial_grid
        self.connectivity_radius = connectivity_radius
        
        logger.info("Initialized SpatialGraphConstructor")
        logger.info(f"Connectivity radius: {connectivity_radius} µm")
    
    def construct_spatial_graph(self, morphogen_concentrations: Dict[str, np.ndarray],
                               downsample_factor: int = 4) -> Dict[str, torch.Tensor]:
        """Construct spatial graph from morphogen concentration data.
        
        Args:
            morphogen_concentrations: Dictionary of morphogen concentration fields
            downsample_factor: Factor to downsample grid for graph construction
            
        Returns:
            Dictionary with graph components (nodes, edges, features)
        """
        logger.info("Constructing spatial connectivity graph")
        
        # Downsample for computational efficiency
        downsampled_data = self._downsample_morphogen_data(morphogen_concentrations, downsample_factor)
        
        # Create node positions and features
        node_positions, node_features = self._create_nodes(downsampled_data)
        
        # Create edges based on spatial connectivity
        edge_indices, edge_features = self._create_edges(node_positions, node_features)
        
        # Convert to PyTorch tensors
        graph_data = {
            "node_positions": torch.from_numpy(node_positions).float(),
            "node_features": torch.from_numpy(node_features).float(),
            "edge_index": torch.from_numpy(edge_indices).long(),
            "edge_features": torch.from_numpy(edge_features).float(),
            "num_nodes": node_positions.shape[0],
            "num_edges": edge_indices.shape[1]
        }
        
        logger.info(f"Graph constructed: {graph_data['num_nodes']} nodes, {graph_data['num_edges']} edges")
        
        return graph_data
    
    def _downsample_morphogen_data(self, morphogen_data: Dict[str, np.ndarray],
                                  factor: int) -> Dict[str, np.ndarray]:
        """Downsample morphogen data for graph construction."""
        downsampled = {}
        
        for morphogen, data in morphogen_data.items():
            # Simple downsampling by taking every nth voxel
            downsampled[morphogen] = data[::factor, ::factor, ::factor]
        
        return downsampled
    
    def _create_nodes(self, morphogen_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Create graph nodes from morphogen data.
        
        Returns:
            Tuple of (node_positions, node_features)
        """
        # Get dimensions of downsampled data
        first_field = next(iter(morphogen_data.values()))
        x_size, y_size, z_size = first_field.shape
        
        # Create node positions (3D coordinates)
        positions = []
        features = []
        
        for x in range(x_size):
            for y in range(y_size):
                for z in range(z_size):
                    # Node position
                    positions.append([x, y, z])
                    
                    # Node features (morphogen concentrations + spatial info)
                    node_feature = []
                    
                    # Add morphogen concentrations
                    for morphogen in ['SHH', 'BMP', 'WNT', 'FGF']:
                        if morphogen in morphogen_data:
                            concentration = morphogen_data[morphogen][x, y, z]
                        else:
                            concentration = 0.0
                        node_feature.append(concentration)
                    
                    # Add spatial features
                    node_feature.extend([
                        x / x_size,  # Normalized x position
                        y / y_size,  # Normalized y position
                        z / z_size,  # Normalized z position
                    ])
                    
                    # Add morphogen gradients (local derivatives)
                    gradients = self._calculate_local_gradients(morphogen_data, x, y, z)
                    node_feature.extend(gradients)
                    
                    features.append(node_feature)
        
        node_positions = np.array(positions)
        node_features = np.array(features)
        
        logger.debug(f"Created {len(positions)} nodes with {node_features.shape[1]} features each")
        
        return node_positions, node_features
    
    def _calculate_local_gradients(self, morphogen_data: Dict[str, np.ndarray],
                                  x: int, y: int, z: int) -> List[float]:
        """Calculate local morphogen gradients at position."""
        gradients = []
        
        for morphogen in ['SHH', 'BMP', 'WNT', 'FGF']:
            if morphogen in morphogen_data:
                field = morphogen_data[morphogen]
                x_size, y_size, z_size = field.shape
                
                # Calculate gradients using finite differences
                grad_x = 0.0
                if x > 0 and x < x_size - 1:
                    grad_x = (field[x+1, y, z] - field[x-1, y, z]) / 2.0
                
                grad_y = 0.0
                if y > 0 and y < y_size - 1:
                    grad_y = (field[x, y+1, z] - field[x, y-1, z]) / 2.0
                
                grad_z = 0.0
                if z > 0 and z < z_size - 1:
                    grad_z = (field[x, y, z+1] - field[x, y, z-1]) / 2.0
                
                # Add gradient magnitude
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                gradients.append(grad_magnitude)
            else:
                gradients.append(0.0)
        
        return gradients
    
    def _create_edges(self, node_positions: np.ndarray, 
                     node_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create graph edges based on spatial connectivity.
        
        Returns:
            Tuple of (edge_indices, edge_features)
        """
        # Calculate pairwise distances
        distances = cdist(node_positions, node_positions)
        
        # Create edges for nodes within connectivity radius
        connectivity_threshold = self.connectivity_radius / self.grid.dimensions.resolution
        
        edge_list = []
        edge_features_list = []
        
        for i in range(len(node_positions)):
            for j in range(i + 1, len(node_positions)):
                distance = distances[i, j]
                
                if distance <= connectivity_threshold:
                    # Add bidirectional edges
                    edge_list.extend([(i, j), (j, i)])
                    
                    # Edge features: distance + morphogen concentration differences
                    morphogen_diff = np.abs(node_features[i, :4] - node_features[j, :4])  # First 4 are morphogens
                    edge_feature = [distance] + morphogen_diff.tolist()
                    
                    edge_features_list.extend([edge_feature, edge_feature])  # Same for both directions
        
        # Convert to arrays
        if edge_list:
            edge_indices = np.array(edge_list).T  # Shape: (2, num_edges)
            edge_features = np.array(edge_features_list)
        else:
            edge_indices = np.zeros((2, 0), dtype=int)
            edge_features = np.zeros((0, 5))  # distance + 4 morphogen differences
        
        logger.debug(f"Created {edge_indices.shape[1]} edges")
        
        return edge_indices, edge_features

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for spatial morphogen analysis.
    
    Implements GNN layers for processing spatial connectivity graphs
    with morphogen concentration node features for regional boundary
    prediction and segmentation tasks.
    """
    
    def __init__(self, input_features: int = 11, hidden_dim: int = 128,
                 output_dim: int = 64, num_layers: int = 3,
                 dropout: float = 0.1):
        """Initialize Graph Neural Network.
        
        Args:
            input_features: Number of input node features
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # GNN layers (simplified GraphConv implementation)
        self.gnn_layers = nn.ModuleList()
        
        # Input layer
        self.gnn_layers.append(nn.Linear(input_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.gnn_layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Activation and normalization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        
        logger.info("Initialized GraphNeuralNetwork")
        logger.info(f"Architecture: {input_features} → {hidden_dim} → {output_dim}")
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GNN.
        
        Args:
            node_features: Node feature tensor (num_nodes, input_features)
            edge_index: Edge connectivity tensor (2, num_edges)
            edge_features: Edge feature tensor (num_edges, edge_features)
            
        Returns:
            Node embeddings (num_nodes, output_dim)
        """
        x = node_features
        
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Graph convolution (simplified message passing)
            x = self._graph_conv(x, edge_index, gnn_layer)
            
            # Normalization and activation
            x = norm(x)
            if i < len(self.gnn_layers) - 1:  # No activation on output layer
                x = self.activation(x)
                x = self.dropout(x)
        
        return x
    
    def _graph_conv(self, x: torch.Tensor, edge_index: torch.Tensor,
                   linear_layer: nn.Linear) -> torch.Tensor:
        """Vectorized and efficient graph convolution operation."""
        # Apply linear transformation to all nodes
        x_transformed = linear_layer(x)

        # Message passing (aggregate neighbor features)
        if edge_index.shape[1] > 0:
            source_nodes, target_nodes = edge_index
            
            # Gather features from source nodes
            source_features = x_transformed[source_nodes]
            
            # Initialize aggregated features tensor
            num_nodes = x_transformed.shape[0]
            aggregated = torch.zeros_like(x_transformed, device=x.device)
            
            # Use scatter_add_ for efficient aggregation
            aggregated.scatter_add_(0, target_nodes.unsqueeze(1).expand_as(source_features), source_features)
            
            # Normalize by node degree
            node_degrees = torch.zeros(num_nodes, device=x.device).unsqueeze(1)
            ones = torch.ones_like(source_nodes, dtype=torch.float)
            node_degrees.scatter_add_(0, target_nodes.unsqueeze(1), ones.unsqueeze(1))
            
            # Avoid division by zero for isolated nodes
            node_degrees[node_degrees == 0] = 1
            
            return aggregated / node_degrees
        else:
            # For graphs with no edges, return the transformed features
            return x_transformed
    
    def predict_regional_boundaries(self, node_embeddings: torch.Tensor,
                                   node_positions: torch.Tensor) -> torch.Tensor:
        """Predict regional boundaries from node embeddings.
        
        Args:
            node_embeddings: Node embeddings from GNN
            node_positions: Spatial positions of nodes
            
        Returns:
            Regional boundary predictions
        """
        # Simple boundary prediction head
        boundary_head = nn.Sequential(
            nn.Linear(self.output_dim + 3, 64),  # embeddings + position
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 regions: forebrain, midbrain, hindbrain, spinal
        ).to(node_embeddings.device)
        
        # Combine embeddings with positions
        combined_features = torch.cat([node_embeddings, node_positions], dim=1)
        
        # Predict regional labels
        boundary_predictions = boundary_head(combined_features)
        boundary_probs = F.softmax(boundary_predictions, dim=1)
        
        return boundary_probs
