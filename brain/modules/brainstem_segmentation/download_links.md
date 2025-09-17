# Human Brainstem Dataset Download Links

*Updated: 2025-09-11*

## 🎯 Priority Downloads (Ready to Use)

### 1. NextBrain Histological Atlas ⭐ TOP PRIORITY
**What**: Most comprehensive human brain atlas (333 ROIs including detailed brainstem)
**Resolution**: 100μm histology with probabilistic segmentation tool
**Status**: ✅ Publicly available

**Direct Download Links**:
- **Main Paper**: https://www.biorxiv.org/content/10.1101/2024.02.05.579016v1
- **PDF**: https://www.biorxiv.org/content/biorxiv/early/2024/02/06/2024.02.05.579016.full.pdf
- **Supplementary Data**: https://www.biorxiv.org/content/10.1101/2024.02.05.579016v1.supplementary-material
- **Interactive Viewer**: Look for web-visualizer link in supplementary materials
- **GitHub Repository**: Search https://github.com/acasamitjana for "NextBrain"

**How to Download**:
1. Visit the bioRxiv link above
2. Click "Download PDF" for the paper
3. Click "Supplementary Material" for atlas data
4. Look for data repository links in the paper

### 2. Human Brainstem Chemoarchitecture Atlas (7T MRI + Histology)
**What**: 7T MRI paired with detailed immunohistochemistry of neurotransmitter systems
**Resolution**: 200μm MRI, 2μm histology
**Status**: ✅ Published in Journal of Neuroscience

**Direct Download Links**:
- **Main Paper**: https://www.jneurosci.org/content/43/2/221
- **PDF**: https://www.jneurosci.org/content/jneuro/43/2/221.full.pdf
- **Supplementary Data**: https://www.jneurosci.org/content/43/2/221/tab-figures-data
- **High-Res Images**: https://www.jneurosci.org/content/43/2/221/tab-figures-data

**How to Download**:
1. Visit the Journal of Neuroscience link
2. Click "Figures & Data" tab
3. Download supplementary figures and data files
4. Look for atlas files in supplementary materials

### 3. Brainstem Arousal Nuclei Atlas (Latest 2024)
**What**: Probabilistic atlas of brainstem arousal nuclei with Bayesian segmentation tool
**Resolution**: 750μm diffusion MRI
**Status**: ✅ Available as medRxiv preprint

**Direct Download Links**:
- **Main Paper**: https://www.medrxiv.org/content/10.1101/2024.09.26.24314117v1
- **PDF**: https://www.medrxiv.org/content/medrxiv/early/2024/09/27/2024.09.26.24314117.full.pdf
- **Data Repository**: Check paper for data availability statement

**How to Download**:
1. Visit the medRxiv link above
2. Download the PDF
3. Check supplementary materials section
4. Look for atlas download links or contact authors

### 4. Structural Connectivity Atlas (Bianciardi Lab)
**What**: 7T/3T diffusion MRI connectivity of 15 brainstem nuclei
**Resolution**: 1.7mm at 7T
**Status**: ✅ Published in Human Brain Mapping

**Direct Download Links**:
- **Paper 1**: https://onlinelibrary.wiley.com/doi/10.1002/hbm.25836
- **Paper 2**: https://onlinelibrary.wiley.com/doi/10.1002/hbm.25962
- **PMC Version**: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9188976/
- **Data**: Check papers for data sharing statements

### 5. True-Color Sectioned Brainstem Atlas
**What**: 212 structures from true-color sectioned images
**Resolution**: 60μm pixel size, 200μm intervals
**Status**: ✅ Open access in Journal of Korean Medical Science

**Direct Download Links**:
- **Main Paper**: https://jkms.org/DOIx.php?id=10.3346/jkms.2023.38.e76
- **PDF**: https://doi.org/10.3346/jkms.2023.38.e76
- **Images**: Check supplementary materials for atlas files

## 🔧 Additional Tools and Resources

### FreeSurfer Brainstem Tools (Already Available)
Since you have FreeSurfer installed, you already have access to:
```bash
# Test these commands work:
mri_segment_brainstem --help
brainstem-substructures --help
freeview --help
```

### Human Connectome Project (HCP) Data
**What**: High-quality diffusion and structural MRI from 1000+ subjects
**Access**: https://www.humanconnectome.org/study/hcp-young-adult/data-releases
**Registration**: Required (free for academic use)

### Allen Human Brain Atlas
**What**: Human brain gene expression and connectivity
**Access**: https://human.brain-map.org/
**API**: https://help.brain-map.org/display/api/

### EBRAINS Human Brain Atlas
**What**: European human brain atlas resources
**Access**: https://ebrains.eu/services/atlases/
**Registration**: Required (free)

## 📥 Download Commands

### Using curl/wget:
```bash
# Create download directory
mkdir -p /Users/camdouglas/quark/data/datasets/brainstem_segmentation/human_atlases

# Download NextBrain paper (example)
cd /Users/camdouglas/quark/data/datasets/brainstem_segmentation/human_atlases
curl -O "https://www.biorxiv.org/content/biorxiv/early/2024/02/06/2024.02.05.579016.full.pdf"

# Download Journal of Neuroscience paper
curl -O "https://www.jneurosci.org/content/jneuro/43/2/221.full.pdf"

# Download medRxiv arousal nuclei paper
curl -O "https://www.medrxiv.org/content/medrxiv/early/2024/09/27/2024.09.26.24314117.full.pdf"
```

### Using Python (for programmatic access):
```python
import requests
from pathlib import Path

# Download function
def download_paper(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")

# Download papers
base_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation/human_atlases")
base_dir.mkdir(parents=True, exist_ok=True)

urls = {
    "nextbrain.pdf": "https://www.biorxiv.org/content/biorxiv/early/2024/02/06/2024.02.05.579016.full.pdf",
    "chemoarchitecture.pdf": "https://www.jneurosci.org/content/jneuro/43/2/221.full.pdf",
    "arousal_nuclei.pdf": "https://www.medrxiv.org/content/medrxiv/early/2024/09/27/2024.09.26.24314117.full.pdf"
}

for filename, url in urls.items():
    download_paper(url, base_dir / filename)
```

## 🎯 Immediate Action Plan

### Step 1: Download Core Papers (5 minutes)
```bash
cd /Users/camdouglas/quark/data/datasets/brainstem_segmentation/human_atlases
curl -O "https://www.biorxiv.org/content/biorxiv/early/2024/02/06/2024.02.05.579016.full.pdf"
curl -O "https://www.jneurosci.org/content/jneuro/43/2/221.full.pdf"  
curl -O "https://www.medrxiv.org/content/medrxiv/early/2024/09/27/2024.09.26.24314117.full.pdf"
```

### Step 2: Access Supplementary Data (10 minutes)
1. Visit each paper's supplementary materials section
2. Look for atlas files (.nii.gz, .mgz, .mat files)
3. Download any available segmentation tools or code

### Step 3: Test FreeSurfer (5 minutes)
```bash
# Verify FreeSurfer brainstem tools
freesurfer --version
mri_segment_brainstem --help
ls $FREESURFER_HOME/average/*brainstem*
```

### Step 4: Set Up Human Data Pipeline (15 minutes)
```bash
# Create human-specific directories
mkdir -p /Users/camdouglas/quark/data/datasets/brainstem_segmentation/human_data/{freesurfer,nextbrain,histology,connectivity}

# Test with FreeSurfer sample data
cd $SUBJECTS_DIR
curl -O https://surfer.nmr.mgh.harvard.edu/pub/data/bert.tar.gz
tar -xzf bert.tar.gz
```

## 📞 Contact Information (if direct downloads fail)

### NextBrain Atlas Team
- **Lead Author**: Juan Eugenio Iglesias
- **Institution**: University College London
- **Email**: Available in paper contact section

### Brainstem Chemoarchitecture Team  
- **Lead Author**: Clifford Saper
- **Institution**: Harvard Medical School
- **Email**: Available in paper

### Arousal Nuclei Atlas Team
- **Lead Author**: Brian Edlow  
- **Institution**: Massachusetts General Hospital
- **Email**: Available in medRxiv paper

## ✅ Ready to Proceed

With FreeSurfer installed and these download links, you can immediately:
1. **Test brainstem segmentation** on sample data
2. **Download comprehensive atlases** for validation
3. **Begin human nucleus catalog** compilation  
4. **Proceed to Step 3** of the roadmap (JSON label schema)

The human brainstem segmentation project now has access to state-of-the-art datasets and proven tools!
