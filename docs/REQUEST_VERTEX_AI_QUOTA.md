# 🚀 Vertex AI Quota Request Guide

## Quick Quota Request (5 minutes)

### **Step 1: Go to GCP Quotas Page**
```bash
# Open this URL in your browser:
https://console.cloud.google.com/iam-admin/quotas?project=quark-469612
```

### **Step 2: Find the Right Quota**
1. In the filter box, type: `aiplatform.googleapis.com/custom_model_training_cpus`
2. Select the quota that appears
3. Click "Edit Quotas" button

### **Step 3: Request Increase**
- **Current Limit**: 0
- **New Limit**: 8 (recommended minimum)
- **Justification**: "Training deep learning model for brain segmentation research project. Need 4-8 CPUs for hybrid GNN-ViT model training."

### **Step 4: Submit Request**
- Click "Submit Request"
- **Typical approval time**: 15 minutes to 2 hours for CPU quotas
- **Cost**: ~$2-5 for entire training run

## Alternative: Use Different Region

If quota request takes too long, try a different region:

```bash
# Check quota in different regions
gcloud compute regions list --filter="name:(us-central1 OR us-east1 OR europe-west1)"
```

## Once Quota is Approved

Run our existing training setup:
```bash
source /Users/camdouglas/quark/gcp_env/bin/activate
python complete_gcp_training_setup.py --credentials_file /Users/camdouglas/quark/data/credentials/quark-469604-training-sa.json
```

## Why This is Better Than Local Training

1. **Scalability**: Can use GPUs when needed
2. **Reproducibility**: Consistent environment
3. **Monitoring**: TensorBoard integration
4. **Cost-effective**: Pay only for training time
5. **Professional**: Matches industry practices

## Current Status

✅ **Docker container**: Built and ready  
✅ **Training script**: Working  
✅ **GCP setup**: Complete  
❌ **Quota**: Needs 8 CPU quota for custom model training  
⏳ **ETA**: 15 minutes - 2 hours after quota request  
