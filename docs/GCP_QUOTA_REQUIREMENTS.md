# GCP Quota Requirements for Quark Training

## 🎯 Specific Quota Needed

Based on the error message and our training configuration, here are the exact quotas you need:

### **Primary Quota Requirement**
- **Quota Name**: `aiplatform.googleapis.com/custom_model_training_cpus`
- **Current Limit**: 0 (default for new projects)
- **Required Minimum**: 4 CPUs
- **Recommended**: 8 CPUs (for flexibility)

### **Machine Type Configuration**
- **Machine Type**: `n1-standard-4`
- **CPUs per machine**: 4 vCPUs
- **Memory**: 15 GB RAM
- **Region**: `us-central1` (best availability)

## 📋 Complete Quota Requirements

### **Vertex AI Quotas**
1. **Custom Model Training CPUs**
   - Metric: `aiplatform.googleapis.com/custom_model_training_cpus`
   - Required: 4 CPUs minimum
   - Location: `us-central1` or `global`

2. **Custom Model Training GPU** (if using GPUs later)
   - Metric: `aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus`
   - Required: 0 (we're using CPU-only for now)

3. **Custom Jobs**
   - Metric: `aiplatform.googleapis.com/custom_jobs`
   - Required: 1 concurrent job
   - Usually available by default

### **Storage Quotas** (Usually Sufficient)
- **Cloud Storage**: Already available
- **Container Registry**: Already available

## 🚀 How to Request Quota Increase

### **Method 1: GCP Console (Recommended)**

1. **Go to**: [GCP Console → IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas)

2. **Filter Settings**:
   - Service: `Vertex AI API`
   - Metric: `Custom model training CPUs`
   - Location: `us-central1` (or `global`)

3. **Request Details**:
   - **Current Limit**: 0
   - **New Limit**: 8 CPUs
   - **Justification**: "Training machine learning models for brain segmentation research project using Vertex AI Custom Jobs"

4. **Expected Approval Time**: 1-2 business days

### **Method 2: gcloud Command**

```bash
# Enable the Service Usage API first
gcloud services enable serviceusage.googleapis.com

# Request quota increase (if available in your region)
gcloud alpha quotas update \
  --service=aiplatform.googleapis.com \
  --quota-id=custom_model_training_cpus \
  --value=8 \
  --project=quark-469604
```

## 🌍 Alternative Regions (if us-central1 has no quota)

Try these regions in order of preference:
1. **us-west1** (Oregon)
2. **us-east1** (South Carolina) 
3. **europe-west1** (Belgium)
4. **asia-southeast1** (Singapore)

Update the location in `complete_gcp_training_setup.py`:
```python
LOCATION = 'us-west1'  # Change this line
```

## 📊 Current Resource Usage

### **What We're Requesting**:
- **1 training job** running simultaneously
- **4 vCPUs** for training (n1-standard-4)
- **15 GB RAM** for the training process
- **No GPUs** (to avoid GPU quota issues)

### **Training Duration**:
- **Estimated**: 2-4 hours for 50 epochs
- **Cost**: ~$2-5 for the entire training run

## ⚡ Quick Start After Quota Approval

Once your quota is approved:

```bash
# Activate environment
source /Users/camdouglas/quark/gcp_env/bin/activate

# Submit training job
python complete_gcp_training_setup.py --credentials_file /Users/camdouglas/quark/data/credentials/quark-469604-training-sa.json
```

## 🎯 Success Criteria

You'll know the quota is sufficient when:
- ✅ Training job submits without quota errors
- ✅ Job status shows "RUNNING" in GCP Console
- ✅ TensorBoard logs start appearing
- ✅ Model artifacts saved to GCS bucket

## 💡 Pro Tips

1. **Start Small**: Request 8 CPUs initially, can increase later
2. **Multiple Regions**: Apply for quota in 2-3 regions for flexibility  
3. **Business Justification**: Mention "research" and "machine learning" in requests
4. **Monitor Usage**: Check quota usage in GCP Console after training

The quota request is the **final step** to complete task 2.3.3: Hybrid model training! 🎊
