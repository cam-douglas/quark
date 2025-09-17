# Brainstem Segmentation Training Pipeline

**Generated**: 2025-09-14  
**Phase**: Phase 2 - Design & Architecture  
**Step**: 3.F3 - Training Pipeline Implementation

## Overview
Complete preprocessing → augmentation → training pipeline for brainstem subdivision segmentation. Integrates ViT-GNN hybrid model with data-centric augmentations and hierarchical loss functions.

## Files

### Core Pipeline
- [`pipeline.py`](pipeline.py) - Main training pipeline implementation
  - **Last Modified**: 2025-09-14
  - **Status**: ✔ active
  - **Function**: Complete training workflow with ViT-GNN model

- [`__init__.py`](__init__.py) - Package initialization  
  - **Last Modified**: 2025-09-14
  - **Status**: ✔ active
  - **Function**: Exports main classes for easy import

## Architecture

### Pipeline Components
1. **Preprocessing**
   - Z-score volume normalization
   - Patch extraction (64³ with 50% overlap)
   - Subdivision mask generation from nucleus labels

2. **Augmentation** 
   - Elastic deformation (anatomically-constrained)
   - Multi-modal noise (Gaussian, Rician, bias field, motion)
   - CutMix nuclei (class-imbalance aware)
   - Morphogen-aware spatial priors

3. **Training**
   - ViT-GNN hybrid model (63M parameters)
   - Hierarchical loss (focal + boundary + consistency)
   - AdamW optimizer with cosine scheduling
   - Mixed precision training support

### Key Features
- **Memory Efficient**: Patch-based training fits within 8GB GPU constraint
- **Class Imbalance**: Specialized CutMix for minority nuclei
- **Anatomical Constraints**: Morphogen gradient priors
- **Hierarchical Learning**: Enforces brainstem → subdivision → nucleus hierarchy

## Usage

### Basic Training
```python
from brain.modules.brainstem_seg import BrainstemTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    batch_size=4,
    learning_rate=1e-4,
    num_epochs=100
)

# Initialize trainer
trainer = BrainstemTrainer(
    config=config,
    data_dir="data/datasets/brainstem_segmentation",
    output_dir="data/models/brainstem_segmentation"
)

# Start training
trainer.train()
```

### Custom Configuration
```python
config = TrainingConfig(
    # Model parameters
    embed_dim=768,
    vit_layers=8,
    gnn_layers=3,
    
    # Training parameters
    batch_size=2,  # Reduce for memory constraints
    learning_rate=5e-5,
    num_epochs=150,
    
    # Augmentation parameters
    elastic_prob=0.9,
    noise_prob=0.7,
    cutmix_prob=0.5
)
```

## Data Requirements

### Expected Directory Structure
```
data/datasets/brainstem_segmentation/
├── nextbrain/
│   ├── T2w.nii.gz                    # 200μm T2-weighted volume
│   ├── manual_segmentation.nii.gz    # 333 ROI labels
│   └── labels.txt                    # Label lookup table
├── metadata/
│   └── brainstem_labels_schema.json  # Hierarchical label schema
└── scripts/
    └── training_augmentations.py     # Augmentation utilities
```

### Supported Data Formats
- **Volumes**: NIfTI format (.nii.gz)
- **Labels**: Integer segmentation masks
- **Schema**: JSON hierarchical mapping

## Training Metrics

### Tracked Metrics
- **Loss**: Hierarchical loss components
- **Dice Coefficient**: Per-class and mean
- **Pixel Accuracy**: Overall classification accuracy
- **Learning Rate**: Adaptive scheduling

### Validation Strategy
- **Interval**: Every 5 epochs
- **Early Stopping**: Dice plateau (20 epoch patience)
- **Best Model**: Saved based on validation Dice

## Output Artifacts

### Model Checkpoints
- `checkpoint_epoch_XXX.pth` - Regular training checkpoints
- `best_model.pth` - Best validation performance model
- `pipeline_configuration.json` - Complete training configuration

### Logs
- `training.log` - Detailed training progress
- Console output with progress bars and metrics

## Memory Optimization

### GPU Memory Management
- **Patch-based training**: 64³ patches vs full volume
- **Mixed precision**: FP16 training when available  
- **Gradient checkpointing**: Memory-efficient backpropagation
- **Batch size adaptation**: Automatic adjustment for hardware

### Estimated Requirements
- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB for faster training
- **CPU fallback**: Supported but slower

## Integration Points

### Dependencies
- **Model**: `../brainstem_segmentation/model_architecture_designer.py`
- **Augmentation**: `../brainstem_segmentation/data_augmentation_designer.py`  
- **Data**: NextBrain atlas and human brain datasets
- **Schema**: JSON label hierarchy from Phase 1

### External Libraries
- PyTorch (model training)
- NiBabel (neuroimaging I/O)
- NumPy/SciPy (data processing)
- tqdm (progress tracking)

## Validation Status

### Testing Results (2025-09-14)
✅ **All pipeline components working**
- Dataset: 50 samples loaded successfully
- Model: 63M parameters initialized
- Loss function: Hierarchical loss computed
- Trainer: Full workflow validated

### Performance Targets
- **Dice Coefficient**: ≥0.87 on validation nuclei
- **Subdivision Accuracy**: ≥0.90 on major divisions
- **Training Time**: <2 hours per epoch on GPU
- **Memory Usage**: <8GB GPU memory

## Next Steps

### Phase 3 - Validation & Testing
1. **Manual Annotation**: 30 slice gold standard
2. **Model Training**: Achieve Dice ≥0.85 target
3. **Cross-validation**: Second annotator grading
4. **CI Integration**: Automated testing pipeline

---

*This pipeline implements Phase 2 Step 3.F3 deliverable as specified in the brainstem segmentation roadmap.*
