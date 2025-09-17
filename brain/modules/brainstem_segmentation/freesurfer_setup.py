#!/usr/bin/env python3
"""
FreeSurfer Setup for Human Brainstem Segmentation

Downloads and configures FreeSurfer for brainstem analysis.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements for FreeSurfer installation."""
    
    requirements = {}
    
    # Check macOS version
    try:
        result = subprocess.run(['sw_vers', '-productVersion'], 
                              capture_output=True, text=True)
        macos_version = result.stdout.strip()
        requirements['macos_version'] = macos_version
        logger.info(f"macOS version: {macos_version}")
    except Exception as e:
        logger.error(f"Could not determine macOS version: {e}")
        requirements['macos_version'] = "unknown"
    
    # Check available disk space
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        disk_info = result.stdout.split('\n')[1].split()
        available_space = disk_info[3]
        requirements['disk_space'] = available_space
        logger.info(f"Available disk space: {available_space}")
    except Exception as e:
        logger.error(f"Could not check disk space: {e}")
        requirements['disk_space'] = "unknown"
    
    # Check memory
    try:
        result = subprocess.run(['sysctl', 'hw.memsize'], 
                              capture_output=True, text=True)
        mem_bytes = int(result.stdout.split(': ')[1])
        mem_gb = mem_bytes / (1024**3)
        requirements['memory_gb'] = f"{mem_gb:.1f} GB"
        logger.info(f"System memory: {mem_gb:.1f} GB")
    except Exception as e:
        logger.error(f"Could not check memory: {e}")
        requirements['memory_gb'] = "unknown"
    
    return requirements


def generate_freesurfer_install_guide() -> str:
    """Generate installation guide for FreeSurfer."""
    
    guide = """
# FreeSurfer Installation Guide for Human Brainstem Segmentation

## Option 1: Download FreeSurfer 7.4+ (Recommended)

### 1. Download
Visit: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
- Choose: "freesurfer-darwin-macOS-7.4.1.tar.gz" for macOS
- Size: ~3.5 GB download, ~15 GB installed
- Registration required (free)

### 2. Installation Commands
```bash
# Extract to /Applications
cd /Applications
sudo tar -xzf ~/Downloads/freesurfer-darwin-macOS-7.4.1.tar.gz

# Set environment variables (add to ~/.zshrc or ~/.bash_profile)
export FREESURFER_HOME=/Applications/freesurfer
export SUBJECTS_DIR=$FREESURFER_HOME/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

### 3. License Setup
```bash
# Copy license file to FreeSurfer directory
cp ~/Downloads/license.txt $FREESURFER_HOME/license.txt
```

## Option 2: Homebrew Installation (Alternative)

```bash
# Install via Homebrew (if available)
brew install --cask freesurfer
```

## Option 3: Docker Container (Isolated)

```bash
# Pull FreeSurfer Docker image
docker pull freesurfer/freesurfer:7.4.1

# Run FreeSurfer in container
docker run -ti --rm -v /path/to/data:/data freesurfer/freesurfer:7.4.1
```

## Verification

After installation, verify FreeSurfer works:

```bash
# Check version
freesurfer --version

# Test brainstem segmentation
mri_segment_brainstem --help

# List available tools
ls $FREESURFER_HOME/bin/mri_*brainstem*
```

## Brainstem-Specific Tools

FreeSurfer 7.4+ includes these brainstem tools:
- `mri_segment_brainstem`: Segment brainstem structures
- `mri_brainstem_ss`: Brainstem surface extraction  
- `brainstem-substructures`: Detailed substructure segmentation

## Required Files for Brainstem Analysis

1. **Atlas Files**: Located in `$FREESURFER_HOME/average/`
   - `brainstemSsLabels.mgz`: Brainstem labels
   - `brainstem_atlas.mgz`: Probabilistic atlas

2. **Model Files**: Located in `$FREESURFER_HOME/models/`
   - Bayesian segmentation models
   - Prior probability maps

## Test Dataset

Use FreeSurfer's test data to verify installation:
```bash
# Download test subject
cd $SUBJECTS_DIR
curl -O https://surfer.nmr.mgh.harvard.edu/pub/data/bert.tar.gz
tar -xzf bert.tar.gz

# Test brainstem segmentation
mri_segment_brainstem bert/mri/norm.mgz bert/mri/brainstem.mgz
```

## Troubleshooting

### Common Issues:
1. **License Error**: Ensure license.txt is in $FREESURFER_HOME
2. **Path Issues**: Source SetUpFreeSurfer.sh in your shell profile
3. **Memory Issues**: FreeSurfer needs 8+ GB RAM for large datasets
4. **Disk Space**: Ensure 20+ GB free space for installation and processing

### Support Resources:
- FreeSurfer Wiki: https://surfer.nmr.mgh.harvard.edu/fswiki/
- Mailing List: freesurfer@nmr.mgh.harvard.edu
- GitHub Issues: https://github.com/freesurfer/freesurfer/issues
"""
    
    return guide


def create_freesurfer_workflow() -> str:
    """Create workflow for human brainstem segmentation with FreeSurfer."""
    
    workflow = """
# Human Brainstem Segmentation Workflow with FreeSurfer

## Input Requirements

1. **T1-weighted MRI**: 1mm isotropic resolution (preferred)
2. **Format**: NIFTI (.nii or .nii.gz) or MGZ
3. **Quality**: Good contrast, minimal motion artifacts
4. **Preprocessing**: Skull-stripped (optional, FreeSurfer can do this)

## Step-by-Step Workflow

### 1. Subject Setup
```bash
# Set up subject directory
export SUBJECTS_DIR=/path/to/your/subjects
mkdir -p $SUBJECTS_DIR/subject_001/mri/orig

# Convert NIFTI to MGZ (if needed)
mri_convert input_t1.nii.gz $SUBJECTS_DIR/subject_001/mri/orig/001.mgz
```

### 2. Basic Preprocessing
```bash
# Run full FreeSurfer pipeline (6-24 hours)
recon-all -subject subject_001 -all

# OR run minimal preprocessing for brainstem only (faster)
recon-all -subject subject_001 -autorecon1
```

### 3. Brainstem Segmentation
```bash
# Segment brainstem structures
mri_segment_brainstem \\
    $SUBJECTS_DIR/subject_001/mri/norm.mgz \\
    $SUBJECTS_DIR/subject_001/mri/brainstem_structures.mgz

# Alternative: Use brainstem substructures tool
brainstem-substructures \\
    --subject subject_001 \\
    --output-dir $SUBJECTS_DIR/subject_001/brainstem/
```

### 4. Extract Specific Nuclei
```bash
# Extract individual structures
mri_extract_label \\
    $SUBJECTS_DIR/subject_001/mri/brainstem_structures.mgz \\
    174 \\  # Brainstem label
    $SUBJECTS_DIR/subject_001/mri/brainstem_only.mgz

# Get volume statistics
mri_segstats \\
    --seg $SUBJECTS_DIR/subject_001/mri/brainstem_structures.mgz \\
    --ctab $FREESURFER_HOME/FreeSurferColorLUT.txt \\
    --sum $SUBJECTS_DIR/subject_001/stats/brainstem_stats.txt
```

### 5. Visualization
```bash
# View results in FreeView
freeview \\
    -v $SUBJECTS_DIR/subject_001/mri/norm.mgz \\
    -v $SUBJECTS_DIR/subject_001/mri/brainstem_structures.mgz:colormap=lut

# Generate screenshots
freeview \\
    -v $SUBJECTS_DIR/subject_001/mri/norm.mgz \\
    -v $SUBJECTS_DIR/subject_001/mri/brainstem_structures.mgz:colormap=lut \\
    -ss $SUBJECTS_DIR/subject_001/screenshots/brainstem.png
```

## Output Files

After processing, you'll have:

1. **Segmentation**: `mri/brainstem_structures.mgz`
2. **Statistics**: `stats/brainstem_stats.txt`
3. **Surfaces**: `surf/` directory (if full recon-all run)
4. **Transforms**: Registration matrices in `mri/transforms/`

## Integration with Quark Pipeline

### 1. Convert to Standard Format
```python
import nibabel as nib
import numpy as np

# Load FreeSurfer segmentation
fs_seg = nib.load('brainstem_structures.mgz')
seg_data = fs_seg.get_fdata()

# Save as NIFTI for Quark pipeline
nifti_img = nib.Nifti1Image(seg_data, fs_seg.affine)
nib.save(nifti_img, 'brainstem_segmentation.nii.gz')
```

### 2. Extract Nucleus Coordinates
```python
# Get center coordinates for each nucleus
from scipy import ndimage

unique_labels = np.unique(seg_data)[1:]  # Exclude background
nucleus_coords = {}

for label in unique_labels:
    mask = seg_data == label
    coords = ndimage.center_of_mass(mask)
    nucleus_coords[int(label)] = coords
```

## Expected Processing Times

- **Minimal preprocessing**: 1-2 hours
- **Full recon-all**: 6-24 hours
- **Brainstem segmentation**: 10-30 minutes
- **Visualization**: 1-5 minutes

## Quality Control

1. **Visual Inspection**: Check segmentation accuracy
2. **Volume Validation**: Compare to literature values
3. **Symmetry Check**: Verify bilateral structures
4. **Atlas Alignment**: Confirm registration quality
"""
    
    return workflow


def main():
    """Generate FreeSurfer setup guide and workflow."""
    
    output_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧠 FREESURFER SETUP FOR HUMAN BRAINSTEM SEGMENTATION")
    print("=" * 60)
    
    # Check system requirements
    requirements = check_system_requirements()
    print(f"\n💻 SYSTEM REQUIREMENTS:")
    for key, value in requirements.items():
        print(f"  {key}: {value}")
    
    # Generate installation guide
    install_guide = generate_freesurfer_install_guide()
    guide_file = output_dir / "freesurfer_installation_guide.md"
    with open(guide_file, 'w') as f:
        f.write(install_guide)
    
    # Generate workflow
    workflow = create_freesurfer_workflow()
    workflow_file = output_dir / "freesurfer_brainstem_workflow.md"
    with open(workflow_file, 'w') as f:
        f.write(workflow)
    
    print(f"\n📋 GENERATED FILES:")
    print(f"  Installation Guide: {guide_file}")
    print(f"  Workflow Guide: {workflow_file}")
    
    print(f"\n🚀 IMMEDIATE ACTIONS:")
    print("1. Visit: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall")
    print("2. Register for free license")
    print("3. Download FreeSurfer 7.4.1 for macOS")
    print("4. Follow installation guide above")
    print("5. Test with provided workflow")
    
    print(f"\n⚡ QUICK START:")
    print("After installation, test with:")
    print("  freesurfer --version")
    print("  mri_segment_brainstem --help")
    
    logger.info("FreeSurfer setup guide generated successfully")


if __name__ == "__main__":
    main()
