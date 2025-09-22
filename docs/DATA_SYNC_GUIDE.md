# 🔄 Quark Data Sync Guide

## 📋 **Overview**

You can **keep your single local `data/` directory** for development while automatically organizing uploads into the appropriate GCS buckets. This gives you the best of both worlds:

- ✅ **Simple local development** - Single `data/` folder
- ✅ **Organized cloud storage** - Automatic bucket classification  
- ✅ **Bidirectional sync** - Upload to cloud or download to local
- ✅ **Intelligent routing** - Files go to the right bucket automatically

## 🗂️ **Bucket Organization Strategy**

| **Local Path** | **GCS Bucket** | **Purpose** |
|----------------|----------------|-------------|
| `data/experimental_papers/` | `gs://quark-data-raw` | Raw research papers, external data |
| `data/external/` | `gs://quark-data-raw` | API responses, scraped data |
| `data/datasets/` | `gs://quark-data-processed` | ML-ready training data |
| `data/models/` | `gs://quark-models` | Trained models, checkpoints |
| `data/reports/` | `gs://quark-experiments` | Experimental results, logs |

## 🚀 **Quick Start**

### **1. Preview Classification**
See how your files would be organized:
```bash
./sync_data.sh --preview
```

### **2. Sync to Cloud** 
Upload your local `data/` to organized GCS buckets:
```bash
./sync_data.sh
```

### **3. Download from Cloud**
Pull all bucket contents back to local `data/`:
```bash
./sync_data.sh --download
```

### **4. Test Run**
See what would happen without actually syncing:
```bash
./sync_data.sh --dry-run
```

## 📁 **File Classification Rules**

The sync script automatically routes files based on these patterns:

### **Raw Data** → `gs://quark-data-raw`
- `experimental_papers/`, `external/`, `web_archives/`, `archive/`
- `*.pdf`, `*.xml`, `*.html`
- Files with `_raw.*` or `_original.*` in the name

### **Processed Data** → `gs://quark-data-processed`  
- `datasets/`, `knowledge/`, `structures/`, `tools/`
- `*.nii.gz`, `*.npy`
- Files with `_processed.*`, `_clean.*`, `_features.*` in the name

### **Models** → `gs://quark-models`
- `models/` directory
- `*.pth`, `*.onnx`, `*.ckpt`, `*.pkl`, `*.h5`
- Files with `_model.*` or `_checkpoint.*` in the name

### **Experiments** → `gs://quark-experiments`
- `reports/`, `logs/`, `results/`, `metrics/`
- `*.log` files
- Files with `_report.*`, `_results.*`, `_metrics.*` in the name

## 🔄 **Recommended Workflow**

### **Daily Development**
1. Work normally in your local `data/` directory
2. Add new datasets, models, or results as usual
3. Periodically run: `./sync_data.sh` to backup to cloud

### **Team Collaboration**
1. Download latest data: `./sync_data.sh --download`
2. Work on your changes locally
3. Sync your updates: `./sync_data.sh`

### **New Machine Setup**
1. Clone the Quark repo
2. Run: `./sync_data.sh --download` 
3. All your data is now available locally

## ⚙️ **Advanced Usage**

### **Custom Classification**
Edit `tools_utilities/gcs_data_sync.py` to modify the `bucket_mapping` patterns:

```python
"raw": {
    "bucket": "gs://quark-data-raw",
    "patterns": [
        "your_custom_pattern/",
        "*.your_extension",
        "*_your_suffix.*"
    ]
}
```

### **Selective Sync**
You can also use `gsutil` directly for specific operations:

```bash
# Sync specific directory
gsutil -m cp -r data/models/ gs://quark-models/

# Download specific bucket
gsutil -m cp -r gs://quark-data-raw/* data/

# Sync with deletion (mirror)
gsutil -m rsync -r -d data/datasets/ gs://quark-data-processed/datasets/
```

## 💡 **Benefits of This Approach**

### **✅ Development Simplicity**
- Keep your familiar single `data/` directory
- No need to remember which bucket to use
- All data available locally for fast access

### **✅ Cloud Organization** 
- Automatic classification into appropriate buckets
- Optimized storage classes and access patterns
- Clear separation of concerns for scaling

### **✅ Cost Optimization**
- Raw data can use cheaper Archive storage
- Processed data uses Standard storage for fast access
- Models can use Nearline for older versions

### **✅ Team Workflow**
- Everyone works with the same local structure
- Automatic organization prevents bucket confusion
- Easy onboarding for new team members

## 🔧 **Troubleshooting**

### **File Not Classified Correctly**
- Check the patterns in `tools_utilities/gcs_data_sync.py`
- Add custom patterns for your specific file types
- Default classification is "processed" if no match

### **Sync Failures**
- Ensure you're authenticated: `gcloud auth login`
- Check bucket permissions: `gsutil iam get gs://bucket-name`
- Verify bucket exists: `gsutil ls gs://bucket-name`

### **Large File Transfers**
- Use `gsutil -m` for parallel transfers (already included)
- Consider `gsutil -o GSUtil:parallel_composite_upload_threshold=150M` for large files

## 📊 **Current Status**

Your buckets are now organized with:
- **Raw Data**: 2.66 GiB (research papers, external sources)
- **Processed Data**: 5.64 GiB (ML datasets, features)  
- **Models**: 496 MiB (trained models, checkpoints)
- **Experiments**: 16.5 KiB (reports, results)

**Total**: ~8.8 GiB across 4 specialized buckets

This structure scales perfectly as Quark grows! 🚀

## 🎯 **Enhanced Features**

### **Activation Words**
The sync script responds to natural language triggers:
- **"sync to gcs"** / **"sync to google cloud"** / **"sync to google cloud service"**
- **"download from gcs"** / **"download from google cloud"** / **"download from google cloud service"**

### **🧹 Smart Cleanup with User Confirmation**
After successful sync, the script will:
1. **Show sync summary**: Total data synced to GCS
2. **List heavy directories**: Directories over size thresholds ready for cleanup
3. **Ask for confirmation**: Manual approval before removing any local data
4. **Verify before cleanup**: Double-check all files are safely in GCS
5. **Report cleanup results**: Show how much storage was freed

### **🔄 Automatic Git Integration**
The pre-push hook automatically:
- Opens new Terminal window showing real-time sync progress
- Doesn't wait for sync to complete (push continues immediately)
- Only syncs files that are new or changed (incremental sync)
- Only syncs if there's data to upload

### **⚡ Maximum Speed Optimizations**
The sync system is configured for fastest possible uploads:
- **24 parallel threads** for concurrent file uploads
- **12 parallel processes** for maximum throughput
- **Incremental sync** - only uploads new/changed files
- **Parallel composite uploads** for files >150MB
- **Sliced downloads** with 8 components for large files
- **Skip slow operations** - hash checks and MIME detection

### **📊 Updated Commands**
```bash
# Main sync with user confirmation for cleanup
./gcs_sync_google_cloud_service.sh

# Sync without cleanup option
./gcs_sync_google_cloud_service.sh --no-cleanup

# Check sync progress (when running in background)
tail -f /tmp/quark_gcs_sync.log
```
