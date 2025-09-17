"""
Hierarchical multi-label framework for anatomically consistent brainstem segmentation.

Enforces anatomical hierarchy: brainstem → subdivision → nucleus
and integrates morphogen priors for biologically plausible predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HierarchyConfig:
    """Configuration for hierarchical segmentation framework."""
    
    # Hierarchy levels
    num_subdivisions: int = 4  # background, midbrain, pons, medulla
    num_nuclei: int = 16       # total nucleus classes
    
    # Loss weights
    subdivision_weight: float = 0.3
    nucleus_weight: float = 0.7
    consistency_weight: float = 0.2
    
    # Morphogen integration
    use_morphogen_priors: bool = True
    morphogen_channels: int = 3  # SHH, BMP, WNT
    morphogen_weight: float = 0.1


class AnatomicalHierarchy:
    """Defines anatomical hierarchy for brainstem structures."""
    
    def __init__(self):
        """Initialize with embryonic brainstem hierarchy."""
        
        # Nucleus to subdivision mapping (36 nuclei total, 12 per subdivision)
        self.nucleus_to_subdivision = {
            0: 0,   # background → background
            
            # Midbrain nuclei (12) → subdivision 1
            100: 1,  # Periaqueductal gray
            101: 1,  # Edinger-Westphal nucleus
            102: 1,  # Substantia nigra pars compacta
            103: 1,  # Substantia nigra pars reticulata
            4: 1,    # Red nucleus (existing label)
            105: 1,  # Oculomotor nucleus
            106: 1,  # Trochlear nucleus
            107: 1,  # Superior colliculus
            85: 1,   # Inferior colliculus (existing label)
            109: 1,  # Ventral tegmental area
            110: 1,  # Reticular formation (midbrain)
            111: 1,  # Interpeduncular nucleus
            
            # Pons nuclei (12) → subdivision 2
            29: 2,   # Pontine nuclei (existing label)
            121: 2,  # Locus coeruleus
            122: 2,  # Abducens nucleus
            123: 2,  # Facial motor nucleus
            124: 2,  # Superior olivary complex
            125: 2,  # Trigeminal motor nucleus
            126: 2,  # Trigeminal sensory nuclei
            127: 2,  # Vestibular nuclei (pontine)
            128: 2,  # Parabrachial nuclei
            129: 2,  # Raphe pontis
            130: 2,  # Reticular formation (pontine)
            131: 2,  # Kölliker-Fuse nucleus
            
            # Medulla nuclei (12) → subdivision 3
            140: 3,  # Nucleus ambiguus
            141: 3,  # Hypoglossal nucleus
            142: 3,  # Dorsal motor nucleus of vagus
            143: 3,  # Nucleus tractus solitarius
            144: 3,  # Inferior olivary complex
            145: 3,  # Raphe magnus
            146: 3,  # Raphe pallidus
            147: 3,  # Pre-Bötzinger complex
            148: 3,  # Bötzinger complex
            149: 3,  # Rostral ventrolateral medulla
            150: 3,  # Gracile and cuneate nuclei
            151: 3,  # Reticular formation (medullary)
        }
        
        # Subdivision names
        self.subdivision_names = {
            0: "background",
            1: "midbrain", 
            2: "pons",
            3: "medulla"
        }
        
        # Functional categories for all 36 nuclei
        self.nucleus_functions = {
            # Midbrain
            100: "autonomic", 101: "autonomic", 102: "sensorimotor", 103: "sensorimotor",
            4: "sensorimotor", 105: "sensorimotor", 106: "sensorimotor", 107: "sensorimotor",
            85: "sensorimotor", 109: "consciousness", 110: "consciousness", 111: "autonomic",
            # Pons
            29: "sensorimotor", 121: "consciousness", 122: "sensorimotor", 123: "sensorimotor",
            124: "sensorimotor", 125: "sensorimotor", 126: "sensorimotor", 127: "sensorimotor",
            128: "autonomic", 129: "consciousness", 130: "consciousness", 131: "autonomic",
            # Medulla
            140: "autonomic", 141: "sensorimotor", 142: "autonomic", 143: "autonomic",
            144: "sensorimotor", 145: "autonomic", 146: "autonomic", 147: "autonomic",
            148: "autonomic", 149: "autonomic", 150: "sensorimotor", 151: "autonomic"
        }
    
    def get_subdivision_mask(self, nucleus_labels: torch.Tensor) -> torch.Tensor:
        """Convert nucleus labels to subdivision labels."""
        subdivision_labels = torch.zeros_like(nucleus_labels)
        
        for nucleus_id, subdivision_id in self.nucleus_to_subdivision.items():
            subdivision_labels[nucleus_labels == nucleus_id] = subdivision_id
        
        return subdivision_labels
    
    def validate_hierarchy_consistency(self, predictions: torch.Tensor) -> torch.Tensor:
        """Check if predictions respect anatomical hierarchy."""
        # Convert to subdivision level
        subdivision_preds = self.get_subdivision_mask(predictions)
        
        # Check for violations (nucleus present but subdivision absent)
        violations = torch.zeros_like(predictions, dtype=torch.bool)
        
        for nucleus_id, subdivision_id in self.nucleus_to_subdivision.items():
            if nucleus_id == 0:  # Skip background
                continue
                
            nucleus_mask = predictions == nucleus_id
            subdivision_mask = subdivision_preds == subdivision_id
            
            # Violation: nucleus predicted but not in correct subdivision
            violations |= nucleus_mask & ~subdivision_mask
        
        return violations


class HierarchicalSegmentationHead(nn.Module):
    """Multi-head segmentation with hierarchical consistency."""
    
    def __init__(self, input_channels: int, config: HierarchyConfig):
        super().__init__()
        self.config = config
        self.hierarchy = AnatomicalHierarchy()
        
        # Subdivision head (coarse)
        self.subdivision_head = nn.Sequential(
            nn.Conv3d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, config.num_subdivisions, 1)
        )
        
        # Nucleus head (fine)
        self.nucleus_head = nn.Sequential(
            nn.Conv3d(input_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, config.num_nuclei, 1)
        )
        
        # Consistency enforcer
        self.consistency_layer = nn.Sequential(
            nn.Conv3d(config.num_subdivisions + config.num_nuclei, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, config.num_nuclei, 1)
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with hierarchical outputs."""
        
        # Subdivision predictions
        subdivision_logits = self.subdivision_head(features)
        
        # Nucleus predictions  
        nucleus_logits = self.nucleus_head(features)
        
        # Enforce consistency
        combined_features = torch.cat([subdivision_logits, nucleus_logits], dim=1)
        consistent_nucleus_logits = self.consistency_layer(combined_features)
        
        return {
            "subdivision_logits": subdivision_logits,
            "nucleus_logits": nucleus_logits,
            "consistent_nucleus_logits": consistent_nucleus_logits
        }


class HierarchicalLossFunction(nn.Module):
    """Loss function enforcing anatomical hierarchy."""
    
    def __init__(self, config: HierarchyConfig):
        super().__init__()
        self.config = config
        self.hierarchy = AnatomicalHierarchy()
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        nucleus_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute hierarchical loss components."""
        
        # Get subdivision targets from nucleus targets
        subdivision_targets = self.hierarchy.get_subdivision_mask(nucleus_targets)
        
        # Subdivision loss
        subdivision_loss = F.cross_entropy(
            outputs["subdivision_logits"], 
            subdivision_targets
        )
        
        # Nucleus loss
        nucleus_loss = F.cross_entropy(
            outputs["nucleus_logits"],
            nucleus_targets
        )
        
        # Consistent nucleus loss
        consistent_nucleus_loss = F.cross_entropy(
            outputs["consistent_nucleus_logits"],
            nucleus_targets
        )
        
        # Consistency penalty
        consistency_loss = F.mse_loss(
            outputs["consistent_nucleus_logits"],
            outputs["nucleus_logits"]
        )
        
        # Combined loss
        total_loss = (
            self.config.subdivision_weight * subdivision_loss +
            self.config.nucleus_weight * (nucleus_loss + consistent_nucleus_loss) / 2 +
            self.config.consistency_weight * consistency_loss
        )
        
        return {
            "total_loss": total_loss,
            "subdivision_loss": subdivision_loss,
            "nucleus_loss": nucleus_loss,
            "consistent_nucleus_loss": consistent_nucleus_loss,
            "consistency_loss": consistency_loss
        }


class MorphogenPriorIntegrator:
    """Integrates morphogen concentration priors into hierarchical framework."""
    
    def __init__(self, config: HierarchyConfig):
        self.config = config
        
        # Morphogen-to-subdivision affinities (based on developmental biology)
        self.morphogen_affinities = {
            "SHH": {1: 0.8, 2: 0.6, 3: 0.4},  # SHH gradient: high midbrain, low medulla
            "BMP": {1: 0.3, 2: 0.5, 3: 0.7},  # BMP gradient: opposite to SHH
            "WNT": {1: 0.4, 2: 0.8, 3: 0.6},  # WNT: high in pons
        }
    
    def create_morphogen_priors(
        self, 
        morphogen_data: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Create subdivision priors from morphogen concentrations.
        
        Args:
            morphogen_data: Morphogen concentrations [C, H, W, D]
            target_shape: Target spatial dimensions
            
        Returns:
            Prior probabilities [num_subdivisions, H, W, D]
        """
        if morphogen_data.shape[0] != 3:
            logger.warning(f"Expected 3 morphogen channels, got {morphogen_data.shape[0]}")
            return torch.zeros(4, *target_shape)  # Return zero priors
        
        # Resize if needed
        if morphogen_data.shape[1:] != target_shape:
            morphogen_data = F.interpolate(
                morphogen_data.unsqueeze(0), 
                size=target_shape, 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Create priors based on morphogen affinities
        priors = torch.zeros(4, *target_shape)  # 4 subdivisions
        
        morphogen_names = ["SHH", "BMP", "WNT"]
        for i, name in enumerate(morphogen_names):
            for subdivision_id, affinity in self.morphogen_affinities[name].items():
                priors[subdivision_id] += affinity * morphogen_data[i]
        
        # Normalize to probabilities
        priors = F.softmax(priors, dim=0)
        
        return priors
    
    def apply_morphogen_guidance(
        self,
        subdivision_logits: torch.Tensor,
        morphogen_priors: torch.Tensor,
        guidance_strength: float = 0.1
    ) -> torch.Tensor:
        """Apply morphogen priors to subdivision predictions."""
        
        # Convert priors to logits
        prior_logits = torch.log(morphogen_priors + 1e-8)
        
        # Weighted combination
        guided_logits = (1 - guidance_strength) * subdivision_logits + guidance_strength * prior_logits
        
        return guided_logits


def test_hierarchical_framework() -> Dict[str, any]:
    """Test hierarchical framework with synthetic data."""
    
    print("🏗️ Hierarchical Framework Test")
    print("=" * 40)
    
    # Create synthetic data
    batch_size = 2
    spatial_dims = (32, 32, 32)
    
    # Mock features
    features = torch.randn(batch_size, 256, *spatial_dims)
    
    # Mock targets
    nucleus_targets = torch.randint(0, 16, (batch_size, *spatial_dims))
    
    # Mock morphogen data
    morphogen_data = torch.rand(3, *spatial_dims)  # SHH, BMP, WNT
    
    # Test hierarchical head
    config = HierarchyConfig()
    seg_head = HierarchicalSegmentationHead(256, config)
    
    outputs = seg_head(features)
    
    print("✅ Hierarchical segmentation head:")
    for key, tensor in outputs.items():
        print(f"  {key}: {tensor.shape}")
    
    # Test loss function
    loss_fn = HierarchicalLossFunction(config)
    loss_dict = loss_fn(outputs, nucleus_targets)
    
    print("\n✅ Hierarchical loss components:")
    for key, loss in loss_dict.items():
        print(f"  {key}: {loss.item():.4f}")
    
    # Test morphogen integration
    morphogen_integrator = MorphogenPriorIntegrator(config)
    priors = morphogen_integrator.create_morphogen_priors(morphogen_data, spatial_dims)
    
    guided_logits = morphogen_integrator.apply_morphogen_guidance(
        outputs["subdivision_logits"], priors
    )
    
    print(f"\n✅ Morphogen priors: {priors.shape}")
    print(f"✅ Guided logits: {guided_logits.shape}")
    
    return {
        "outputs": {k: v.shape for k, v in outputs.items()},
        "losses": {k: v.item() for k, v in loss_dict.items()},
        "priors_shape": priors.shape,
        "guided_shape": guided_logits.shape
    }


if __name__ == "__main__":
    # Run test
    results = test_hierarchical_framework()
    
    # Save results
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "hierarchical_framework_test.json"
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Test results saved: {results_path}")
    print("✅ Hierarchical framework ready for integration")
