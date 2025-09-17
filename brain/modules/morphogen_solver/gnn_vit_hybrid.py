#!/usr/bin/env python3
"""GNN-ViT Hybrid Model for 3D Segmentation.

Implements hybrid GNN-ViT model combining graph neural networks with
vision transformers for 3D segmentation with limited labels including
semi-supervised learning and transfer learning capabilities.

Integration: Hybrid model for advanced ML segmentation system
Rationale: Combined GNN-ViT approach for enhanced 3D segmentation
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .vit3d_encoder import ViT3DEncoder
from .gnn_spatial_graph import GraphNeuralNetwork, SpatialGraphConstructor

logger = logging.getLogger(__name__)

class CrossModalFusion(nn.Module):
    """Cross-modal fusion layer for GNN and ViT features."""
    
    def __init__(self, gnn_dim: int = 64, vit_dim: int = 192, 
                 fusion_dim: int = 128):
        """Initialize cross-modal fusion.
        
        Args:
            gnn_dim: GNN feature dimension
            vit_dim: ViT feature dimension
            fusion_dim: Fused feature dimension
        """
        super().__init__()
        
        self.gnn_projection = nn.Linear(gnn_dim, fusion_dim)
        self.vit_projection = nn.Linear(vit_dim, fusion_dim)
        
        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, gnn_features: torch.Tensor, 
                vit_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through cross-modal fusion.
        
        Args:
            gnn_features: GNN node features (B, num_nodes, gnn_dim)
            vit_features: ViT patch features (B, num_patches, vit_dim)
            
        Returns:
            Fused features (B, num_nodes, fusion_dim)
        """
        # Project to common dimension
        gnn_proj = self.gnn_projection(gnn_features)
        vit_proj = self.vit_projection(vit_features)
        
        # Cross-attention: GNN queries ViT
        gnn_attended, _ = self.cross_attention(gnn_proj, vit_proj, vit_proj)
        
        # Concatenate and fuse
        concatenated = torch.cat([gnn_proj, gnn_attended], dim=-1)
        fused = self.fusion_layers(concatenated)
        
        # Residual connection and normalization
        output = self.layer_norm(fused + gnn_proj)
        
        return output

class GNNViTHybridModel(nn.Module):
    """Hybrid GNN-ViT model for 3D morphogen segmentation.
    
    Combines Graph Neural Networks with Vision Transformers for enhanced
    3D segmentation of morphogen concentration data with limited labels
    and semi-supervised learning capabilities.
    
    Key Components:
    - ViT3D encoder for patch-based processing
    - GNN for spatial connectivity modeling
    - Cross-modal fusion layer
    - Segmentation prediction head
    """
    
    def __init__(self, input_channels: int = 4, num_classes: int = 4,
                 vit_embed_dim: int = 768, gnn_hidden_dim: int = 128,
                 fusion_dim: int = 128, input_resolution: int = 64):
        """Initialize GNN-ViT hybrid model.
        
        Args:
            input_channels: Number of input channels (morphogens)
            num_classes: Number of segmentation classes
            vit_embed_dim: ViT embedding dimension
            gnn_hidden_dim: GNN hidden dimension
            fusion_dim: Cross-modal fusion dimension
            input_resolution: Input spatial resolution
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_resolution = input_resolution
        
        # ViT3D encoder
        self.vit_encoder = ViT3DEncoder(
            input_channels=input_channels,
            embed_dim=vit_embed_dim,
            input_resolution=input_resolution
        )
        
        # Graph Neural Network
        self.gnn = GraphNeuralNetwork(
            input_features=11,  # 4 morphogens + 3 spatial + 4 gradients
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_hidden_dim
        )
        
        # Cross-modal fusion
        vit_feature_dim = vit_embed_dim // 4  # From ViT feature head
        self.fusion = CrossModalFusion(
            gnn_dim=gnn_hidden_dim,
            vit_dim=vit_feature_dim,
            fusion_dim=fusion_dim
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Graph constructor for converting 3D data to graphs
        self.graph_constructor = None  # Will be set externally
        
        logger.info("Initialized GNNViTHybridModel")
        logger.info(f"Classes: {num_classes}, Resolution: {input_resolution}³")
    
    def forward(self, x: torch.Tensor, graph_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through hybrid model.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            graph_data: Precomputed graph data (optional)
            
        Returns:
            Dictionary with model outputs
        """
        batch_size = x.shape[0]
        
        # ViT3D processing
        vit_patch_features, vit_global_features = self.vit_encoder(x)
        
        # Process each sample in batch for GNN (graphs have different sizes)
        gnn_outputs = []
        fusion_outputs = []
        
        for b in range(batch_size):
            # Get or create graph for this sample
            if graph_data is not None and b < len(graph_data.get("node_features", [])):
                # Use provided graph data
                node_features = graph_data["node_features"][b]
                edge_index = graph_data["edge_index"][b]
                edge_features = graph_data.get("edge_features", [None])[b]
            else:
                # Create graph from 3D data (simplified)
                node_features, edge_index = self._create_sample_graph(x[b])
                edge_features = None
            
            # GNN processing
            gnn_features = self.gnn(node_features, edge_index, edge_features)
            gnn_outputs.append(gnn_features)
            
            # Cross-modal fusion
            vit_features_sample = vit_patch_features[b:b+1]  # Keep batch dimension
            gnn_features_sample = gnn_features.unsqueeze(0)  # Add batch dimension
            
            fused_features = self.fusion(gnn_features_sample, vit_features_sample)
            fusion_outputs.append(fused_features.squeeze(0))
        
        # Segmentation predictions
        segmentation_outputs = []
        for fused_features in fusion_outputs:
            seg_logits = self.segmentation_head(fused_features)
            segmentation_outputs.append(seg_logits)
        
        # Combine outputs
        outputs = {
            "vit_patch_features": vit_patch_features,
            "vit_global_features": vit_global_features,
            "gnn_features": gnn_outputs,
            "fused_features": fusion_outputs,
            "segmentation_logits": segmentation_outputs
        }
        
        return outputs
    
    def _create_sample_graph(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create graph from single 3D sample (simplified)."""
        # This is a simplified version - in practice would use SpatialGraphConstructor
        
        # Convert to numpy for graph construction
        sample_np = sample.cpu().numpy()
        
        # Create simple grid graph (6-connectivity)
        C, D, H, W = sample_np.shape
        
        # Downsample for graph construction
        downsample_factor = 4
        d_small = D // downsample_factor
        h_small = H // downsample_factor
        w_small = W // downsample_factor
        
        # Create node features (simplified)
        node_features = []
        node_positions = []
        
        for z in range(d_small):
            for y in range(h_small):
                for x in range(w_small):
                    # Original coordinates
                    orig_z = z * downsample_factor
                    orig_y = y * downsample_factor
                    orig_x = x * downsample_factor
                    
                    # Node features: morphogen concentrations + position
                    features = [
                        sample_np[0, orig_z, orig_y, orig_x],  # SHH
                        sample_np[1, orig_z, orig_y, orig_x],  # BMP
                        sample_np[2, orig_z, orig_y, orig_x],  # WNT
                        sample_np[3, orig_z, orig_y, orig_x],  # FGF
                        x / w_small, y / h_small, z / d_small,  # Normalized position
                        0.0, 0.0, 0.0, 0.0  # Placeholder gradients
                    ]
                    
                    node_features.append(features)
                    node_positions.append([x, y, z])
        
        # Create edges (6-connectivity)
        edges = []
        num_nodes = len(node_features)
        
        for i, pos1 in enumerate(node_positions):
            for j, pos2 in enumerate(node_positions):
                if i < j:
                    # Manhattan distance
                    dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
                    if dist == 1:  # Adjacent nodes
                        edges.extend([[i, j], [j, i]])  # Bidirectional
        
        # Convert to tensors
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32, device=sample.device)
        
        if edges:
            edge_index_tensor = torch.tensor(edges, dtype=torch.long, device=sample.device).T
        else:
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long, device=sample.device)
        
        return node_features_tensor, edge_index_tensor
    
    def compute_segmentation_loss(self, outputs: Dict[str, torch.Tensor],
                                 targets: torch.Tensor, 
                                 use_consistency_loss: bool = True) -> torch.Tensor:
        """Compute segmentation loss with optional consistency regularization.
        
        Args:
            outputs: Model outputs
            targets: Ground truth segmentation targets
            use_consistency_loss: Whether to use consistency regularization
            
        Returns:
            Total loss
        """
        segmentation_logits = outputs["segmentation_logits"]
        
        # Primary segmentation loss
        seg_loss = 0.0
        for i, logits in enumerate(segmentation_logits):
            if i < len(targets):
                target = targets[i]
                loss = F.cross_entropy(logits, target)
                seg_loss += loss
        
        seg_loss = seg_loss / len(segmentation_logits) if segmentation_logits else 0.0
        
        total_loss = seg_loss
        
        # Consistency loss between ViT and GNN predictions (semi-supervised)
        if use_consistency_loss and len(segmentation_logits) > 1:
            consistency_loss = 0.0
            for i in range(len(segmentation_logits) - 1):
                for j in range(i + 1, len(segmentation_logits)):
                    # KL divergence between predictions
                    pred_i = F.softmax(segmentation_logits[i], dim=1)
                    pred_j = F.softmax(segmentation_logits[j], dim=1)
                    
                    kl_loss = F.kl_div(pred_i.log(), pred_j, reduction='batchmean')
                    consistency_loss += kl_loss
            
            consistency_loss = consistency_loss / max(1, len(segmentation_logits) * (len(segmentation_logits) - 1) // 2)
            total_loss += 0.1 * consistency_loss  # Weight consistency loss
        
        return total_loss
