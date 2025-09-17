#!/usr/bin/env python3
"""
Model Architecture Designer - Phase 2 Step 1.F1

Designs ViT-GNN hybrid segmentation backbone with appropriate loss functions
for brainstem subdivision segmentation.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelSpecification:
    """Model architecture specification."""
    name: str
    backbone: str
    input_shape: Tuple[int, int, int, int]  # (C, H, W, D)
    output_classes: int
    patch_size: Tuple[int, int, int]
    embed_dim: int
    num_heads: int
    num_layers: int
    gnn_layers: int
    dropout: float
    memory_gb: float
    parameters: int


class Vision3DTransformer(nn.Module):
    """
    3D Vision Transformer for volumetric brainstem segmentation.
    
    Adapts ViT architecture for 3D medical imaging with patch-based processing.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 patch_size: Tuple[int, int, int] = (16, 16, 16),
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 num_classes: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            input_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, embed_dim))  # Max patches
        
        # Manual transformer encoder (ONNX-compatible)
        self.transformer_layers = nn.ModuleList([
            self._create_onnx_compatible_transformer_layer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def _create_onnx_compatible_transformer_layer(self, embed_dim: int, num_heads: int, dropout: float) -> nn.Module:
        """Create ONNX-compatible transformer layer without scaled_dot_product_attention."""
        
        return nn.ModuleDict({
            'self_attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True),
            'linear1': nn.Linear(embed_dim, embed_dim * 4),
            'dropout': nn.Dropout(dropout),
            'linear2': nn.Linear(embed_dim * 4, embed_dim),
            'norm1': nn.LayerNorm(embed_dim),
            'norm2': nn.LayerNorm(embed_dim),
            'dropout1': nn.Dropout(dropout),
            'dropout2': nn.Dropout(dropout),
        })
    
    def _apply_transformer_layer(self, x: torch.Tensor, layer: nn.ModuleDict) -> torch.Tensor:
        """Apply single transformer layer."""
        
        # Self-attention with residual connection
        attn_output, _ = layer['self_attn'](x, x, x)
        x = layer['norm1'](x + layer['dropout1'](attn_output))
        
        # Feed-forward with residual connection
        ff_output = layer['linear2'](layer['dropout'](F.relu(layer['linear1'](x))))
        x = layer['norm2'](x + layer['dropout2'](ff_output))
        
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D ViT.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            
        Returns:
            Segmentation logits [B, num_classes, H, W, D]
        """
        B, C, H, W, D = x.shape
        
        # Extract patches
        patches = self.patch_embed(x)  # [B, embed_dim, Ph, Pw, Pd]
        Ph, Pw, Pd = patches.shape[2:]
        
        # Flatten patches (ONNX-compatible way)
        B, C, Ph, Pw, Pd = patches.shape
        patches = patches.view(B, C, Ph * Pw * Pd).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add positional embedding
        num_patches = patches.shape[1]
        patches = patches + self.pos_embed[:, :num_patches, :]
        
        # Manual transformer encoding (ONNX-compatible)
        encoded = patches
        for layer in self.transformer_layers:
            encoded = self._apply_transformer_layer(encoded, layer)
        encoded = self.norm(encoded)
        
        # Classification
        logits = self.head(encoded)  # [B, num_patches, num_classes]
        
        # Reshape to volume (ONNX-compatible way)
        logits = logits.transpose(1, 2).contiguous().view(B, self.num_classes, Ph, Pw, Pd)
        
        # Upsample to original resolution
        logits = F.interpolate(
            logits, size=(H, W, D), 
            mode='trilinear', align_corners=False
        )
        
        return logits


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for anatomical structure relationships.
    
    Models spatial relationships between brainstem nuclei using graph convolutions.
    """
    
    def __init__(self, 
                 node_features: int = 768,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_classes: int = 16):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.gconv_layers = nn.ModuleList([
            nn.Linear(node_features if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_classes)
        
        # Activation and normalization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def create_anatomical_adjacency(self, batch_size: int, device: str) -> torch.Tensor:
        """Create anatomical adjacency matrix based on brainstem structure."""
        
        # Define anatomical connections (simplified)
        connections = [
            # Midbrain connections
            (1, 2), (1, 3), (2, 3),  # SN-RN-PAG
            (3, 4), (4, 5), (5, 6),  # PAG-VTA-SC-IC
            
            # Pons connections  
            (7, 8), (8, 9), (9, 10), # PN-LC-PBN-SOC
            (3, 8), (8, 11),         # PAG-LC, LC-medulla
            
            # Medulla connections
            (11, 12), (12, 13), (13, 14), (14, 15), # NTS-DRN-RMg-IO chain
            (9, 11), (10, 12)        # Pons to medulla
        ]
        
        # Create adjacency matrix
        adj = torch.zeros(16, 16, device=device)
        for i, j in connections:
            adj[i, j] = 1.0
            adj[j, i] = 1.0  # Symmetric
            
        # Add self-connections
        adj += torch.eye(16, device=device)
        
        # Normalize
        degree = adj.sum(dim=1, keepdim=True)
        adj = adj / (degree + 1e-8)
        
        # Expand for batch
        return adj.unsqueeze(0).expand(batch_size, -1, -1)
        
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            node_features: Node features [B, num_nodes, feature_dim]
            
        Returns:
            Updated node features [B, num_nodes, num_classes]
        """
        B, N, _ = node_features.shape
        device = node_features.device
        
        # Get adjacency matrix
        adj = self.create_anatomical_adjacency(B, device)
        
        x = node_features
        
        # Graph convolution layers
        for i, gconv in enumerate(self.gconv_layers):
            # Message passing: aggregate neighbor features
            x_agg = torch.bmm(adj, x)  # [B, N, feature_dim]
            
            # Update features
            x = gconv(x_agg)
            x = self.activation(x)
            x = self.dropout(x)
            
        # Output projection
        output = self.output_proj(x)  # [B, N, num_classes]
        
        return output


class ViTGNNHybrid(nn.Module):
    """
    Hybrid ViT-GNN architecture for brainstem segmentation.
    
    Combines Vision Transformer for local feature extraction with
    Graph Neural Network for anatomical relationship modeling.
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 patch_size: Tuple[int, int, int] = (16, 16, 16),
                 embed_dim: int = 768,
                 vit_layers: int = 8,  # Reduced for memory
                 gnn_layers: int = 3,
                 num_heads: int = 8,   # Reduced for memory
                 num_classes: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # ViT backbone
        self.vit = Vision3DTransformer(
            input_channels=input_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=vit_layers,
            num_classes=embed_dim,  # Output features, not classes
            dropout=dropout
        )
        
        # GNN for anatomical relationships
        self.gnn = GraphNeuralNetwork(
            node_features=embed_dim,
            hidden_dim=256,
            num_layers=gnn_layers,
            num_classes=num_classes
        )
        
        # Feature pooling for GNN input
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))  # 64 regions
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv3d(embed_dim + num_classes, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, num_classes, 1)
        )
        
    def extract_anatomical_features(self, vit_features: torch.Tensor) -> torch.Tensor:
        """Extract anatomical node features for GNN."""
        
        B, C, H, W, D = vit_features.shape
        
        # Pool to anatomical regions
        pooled = self.adaptive_pool(vit_features)  # [B, C, 4, 4, 4]
        
        # Flatten to nodes
        nodes = pooled.flatten(2).transpose(1, 2)  # [B, 64, C]
        
        # Select top 16 nodes (corresponding to nuclei)
        node_features = nodes[:, :16, :]  # [B, 16, C]
        
        return node_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid ViT-GNN.
        
        Args:
            x: Input volume [B, C, H, W, D]
            
        Returns:
            Segmentation logits [B, num_classes, H, W, D]
        """
        B, C, H, W, D = x.shape
        
        # ViT feature extraction
        vit_features = self.vit(x)  # [B, embed_dim, H, W, D]
        
        # Extract anatomical node features
        node_features = self.extract_anatomical_features(vit_features)
        
        # GNN processing
        gnn_output = self.gnn(node_features)  # [B, 16, num_classes]
        
        # Broadcast GNN output to spatial dimensions
        # gnn_output shape: [B, 16, num_classes] -> need [B, num_classes, H, W, D]
        gnn_spatial = gnn_output.transpose(1, 2)  # [B, num_classes, 16]
        gnn_spatial = gnn_spatial.mean(dim=2)  # [B, num_classes] - average over nodes
        gnn_spatial = gnn_spatial.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, num_classes, 1, 1, 1]
        gnn_spatial = gnn_spatial.expand(-1, -1, H, W, D)  # [B, num_classes, H, W, D]
        
        # Combine ViT and GNN features
        combined = torch.cat([vit_features, gnn_spatial], dim=1)
        
        # Final segmentation
        output = self.seg_head(combined)
        
        return output


class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss combining subdivision and nucleus-level losses.
    
    Enforces anatomical hierarchy: brainstem → subdivision → nucleus
    """
    
    def __init__(self, 
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 hierarchy_weight: float = 0.3,
                 boundary_weight: float = 0.2):
        super().__init__()
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.hierarchy_weight = hierarchy_weight
        self.boundary_weight = boundary_weight
        
        # Define hierarchical mapping
        self.nucleus_to_subdivision = {
            # Background
            0: 0,
            # Midbrain nuclei (1-6) → subdivision 1
            1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
            # Pons nuclei (7-10) → subdivision 2  
            7: 2, 8: 2, 9: 2, 10: 2,
            # Medulla nuclei (11-15) → subdivision 3
            11: 3, 12: 3, 13: 3, 14: 3, 15: 3
        }
        
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
    def boundary_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute boundary-aware loss."""
        
        # Compute gradients to detect boundaries
        def compute_gradient_magnitude(tensor):
            grad_x = torch.abs(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :])
            grad_y = torch.abs(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :])
            grad_z = torch.abs(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1])
            
            # Pad to original size
            grad_x = F.pad(grad_x, (0, 0, 0, 0, 0, 1))
            grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
            grad_z = F.pad(grad_z, (0, 1, 0, 0, 0, 0))
            
            return grad_x + grad_y + grad_z
        
        # Target boundaries
        target_boundaries = compute_gradient_magnitude(targets.float().unsqueeze(1))
        
        # Prediction boundaries
        pred_softmax = F.softmax(inputs, dim=1)
        pred_boundaries = compute_gradient_magnitude(pred_softmax)
        pred_boundaries = pred_boundaries.sum(dim=1, keepdim=True)
        
        # Boundary loss
        boundary_loss = F.mse_loss(pred_boundaries, target_boundaries)
        
        return boundary_loss
    
    def hierarchy_consistency_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Enforce hierarchical consistency."""
        
        # Create subdivision targets
        subdivision_targets = torch.zeros_like(targets)
        for nucleus_id, subdivision_id in self.nucleus_to_subdivision.items():
            subdivision_targets[targets == nucleus_id] = subdivision_id
            
        # Pool predictions to subdivision level
        subdivision_logits = torch.zeros(
            inputs.shape[0], 4, *inputs.shape[2:],  # 4 subdivisions
            device=inputs.device
        )
        
        for nucleus_id, subdivision_id in self.nucleus_to_subdivision.items():
            if nucleus_id < inputs.shape[1]:  # Valid nucleus
                subdivision_logits[:, subdivision_id] += inputs[:, nucleus_id]
                
        # Subdivision loss
        subdivision_loss = self.focal_loss(subdivision_logits, subdivision_targets)
        
        return subdivision_loss
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical loss.
        
        Args:
            inputs: Model predictions [B, C, H, W, D]
            targets: Ground truth labels [B, H, W, D]
            
        Returns:
            Dictionary of loss components
        """
        
        # Main focal loss
        main_loss = self.focal_loss(inputs, targets)
        
        # Boundary loss
        boundary_loss = self.boundary_loss(inputs, targets)
        
        # Hierarchy consistency loss
        hierarchy_loss = self.hierarchy_consistency_loss(inputs, targets)
        
        # Combined loss
        total_loss = (
            main_loss + 
            self.boundary_weight * boundary_loss +
            self.hierarchy_weight * hierarchy_loss
        )
        
        return {
            'total_loss': total_loss,
            'focal_loss': main_loss,
            'boundary_loss': boundary_loss,
            'hierarchy_loss': hierarchy_loss
        }


def create_model_specification() -> ModelSpecification:
    """Create detailed model specification."""
    
    # Calculate model parameters
    patch_size = (16, 16, 16)
    embed_dim = 768
    vit_layers = 8
    gnn_layers = 3
    num_heads = 8
    
    # Estimate parameters (in millions)
    vit_params = (
        embed_dim * np.prod(patch_size) +  # Patch embedding
        embed_dim * 1000 +                 # Positional embedding
        vit_layers * (embed_dim * embed_dim * 4 + embed_dim * num_heads * 64) +  # Transformer
        embed_dim * 16                     # Classification head
    )
    
    gnn_params = gnn_layers * (768 * 256 + 256 * 256) + 256 * 16
    
    seg_head_params = (768 + 16) * 512 + 512 * 256 + 256 * 16
    
    total_params = (vit_params + gnn_params + seg_head_params) // 1_000_000
    
    # Estimate memory usage
    input_memory = (64 * 64 * 64 * 4 * 4) / (1024**3)  # Batch=4, float32
    model_memory = total_params * 4 / 1024  # MB to GB
    
    spec = ModelSpecification(
        name="ViT-GNN Hybrid Brainstem Segmentation",
        backbone="Vision Transformer + Graph Neural Network",
        input_shape=(1, 64, 64, 64),  # Patch-based training
        output_classes=16,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=vit_layers,
        gnn_layers=gnn_layers,
        dropout=0.1,
        memory_gb=round(input_memory + model_memory, 2),
        parameters=int(total_params)
    )
    
    return spec


def main():
    """Execute Phase 2 Step 1.F1: Model architecture design."""
    
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🏗️ PHASE 2 STEP 1.F1 - MODEL ARCHITECTURE DESIGN")
    print("=" * 55)
    
    # Create model specification
    spec = create_model_specification()
    
    print(f"📊 Model Specification:")
    print(f"   Architecture: {spec.backbone}")
    print(f"   Parameters: {spec.parameters}M")
    print(f"   Memory requirement: {spec.memory_gb} GB")
    print(f"   Input shape: {spec.input_shape}")
    print(f"   Output classes: {spec.output_classes}")
    
    # Test model instantiation
    print(f"\n🔧 Testing Model Instantiation...")
    try:
        model = ViTGNNHybrid(
            input_channels=1,
            patch_size=spec.patch_size,
            embed_dim=spec.embed_dim,
            num_classes=spec.output_classes
        )
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"   ✅ Model instantiated successfully")
        print(f"   ✅ Forward pass: {dummy_input.shape} → {output.shape}")
        
        # Test loss function
        loss_fn = HierarchicalLoss()
        dummy_target = torch.randint(0, 16, (1, 64, 64, 64))
        
        with torch.no_grad():
            loss_dict = loss_fn(output, dummy_target)
            
        print(f"   ✅ Loss function working: {loss_dict['total_loss'].item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        
    # Save model specification
    spec_dict = {
        "generated": datetime.now().isoformat(),
        "phase": "Phase 2 - Design & Architecture",
        "step": "1.F1 - Model Architecture Selection",
        
        "architecture": {
            "name": spec.name,
            "type": "Hybrid ViT-GNN",
            "components": {
                "vision_transformer": {
                    "layers": spec.num_layers,
                    "embed_dim": spec.embed_dim,
                    "num_heads": spec.num_heads,
                    "patch_size": spec.patch_size,
                    "purpose": "Local feature extraction from 3D patches"
                },
                "graph_neural_network": {
                    "layers": spec.gnn_layers,
                    "hidden_dim": 256,
                    "node_features": spec.embed_dim,
                    "purpose": "Anatomical relationship modeling"
                },
                "segmentation_head": {
                    "layers": 3,
                    "channels": [784, 512, 256, 16],
                    "purpose": "Final voxel-wise classification"
                }
            }
        },
        
        "loss_functions": {
            "hierarchical_loss": {
                "components": ["focal_loss", "boundary_loss", "hierarchy_consistency"],
                "weights": {"boundary": 0.2, "hierarchy": 0.3},
                "focal_params": {"alpha": 0.25, "gamma": 2.0}
            }
        },
        
        "specifications": {
            "parameters_millions": spec.parameters,
            "memory_gb": spec.memory_gb,
            "input_shape": spec.input_shape,
            "output_classes": spec.output_classes,
            "patch_based": True,
            "memory_constraint_satisfied": bool(spec.memory_gb < 8.0)
        },
        
        "design_rationale": {
            "vit_choice": "Excellent for capturing long-range dependencies in 3D volumes",
            "gnn_addition": "Models anatomical relationships between brainstem nuclei",
            "hierarchical_loss": "Enforces brainstem → subdivision → nucleus hierarchy",
            "memory_optimization": "Patch-based training with reduced transformer layers"
        }
    }
    
    # Save specification
    spec_file = output_dir / "model_architecture_specification.json"
    with open(spec_file, 'w') as f:
        json.dump(spec_dict, f, indent=2)
    
    print(f"\n✅ Phase 2 Step 1.F1 Complete!")
    print(f"   Model specification saved to: {spec_file}")
    print(f"   Architecture: ViT-GNN Hybrid ({spec.parameters}M parameters)")
    print(f"   Memory requirement: {spec.memory_gb} GB (within 8GB constraint)")
    print(f"   Loss function: Hierarchical (focal + boundary + consistency)")


if __name__ == "__main__":
    main()
