# Google Colab Training Files

## 📁 Files to Upload:
1. `colab_training_package.py` - Complete training code
2. `imaging_data_normalized.npy` - Normalized T2w imaging data
3. `labels.npy` - Ground truth segmentation labels
4. `metadata.json` - Dataset information

## 🚀 Instructions:
1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook or use the provided `colab_training_notebook.ipynb`
3. Enable GPU: Runtime → Change runtime type → GPU
4. Upload all files using the file browser or upload widget
5. Run: `exec(open('colab_training_package.py').read())`
6. Wait for training to complete (1.5-2 hours)
7. Download `best_model_colab.pth` when done

## 🎯 Target:
- Achieve Dice coefficient ≥ 0.87
- Expected to reach target within 350 epochs

## 📥 Integration:
After training, download the model and place it at:
`/Users/camdouglas/quark/data/models/brainstem/best_model_colab.pth`
