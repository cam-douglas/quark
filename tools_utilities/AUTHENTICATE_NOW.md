# 🔐 Quick Authentication Guide for Google Cloud

## Step 1: Authenticate (Choose One Method)

### Method A: Using Our Helper Script (Easiest)
```bash
cd /Users/camdouglas/quark
python tools_utilities/google_cloud_api_manager.py --auth YOUR-PROJECT-ID
```

### Method B: Using gcloud Directly
```bash
# 1. Login to Google Cloud
gcloud auth login --no-launch-browser

# 2. Set your project (replace with your actual project ID)
gcloud config set project YOUR-PROJECT-ID

# 3. Set up Application Default Credentials (for Python libraries)
gcloud auth application-default login --no-launch-browser
```

## Step 2: Enable APIs Programmatically

Once authenticated, you can enable APIs without using the console:

```bash
# Enable essential APIs for Quark
python tools_utilities/google_cloud_api_manager.py --enable-essential

# Or enable specific APIs
python tools_utilities/google_cloud_api_manager.py --enable vision
python tools_utilities/google_cloud_api_manager.py --enable aiplatform
```

## Step 3: Verify Authentication

```bash
# Check status
python tools_utilities/google_cloud_api_manager.py

# List enabled APIs
python tools_utilities/google_cloud_api_manager.py --list
```

## 📝 What You Need

1. **Your Google Cloud Project ID** - Find it at https://console.cloud.google.com
2. **A Google Account** with access to the project
3. **Billing Enabled** on the project (for most APIs)

## 🎯 Benefits of Authentication

Once authenticated, you can:
- ✅ Enable/disable APIs programmatically
- ✅ Access Cloud Storage for large files
- ✅ Use Vertex AI for model training
- ✅ Deploy to Cloud Run
- ✅ Query BigQuery datasets
- ✅ Use GPU/TPU resources
- ✅ Access Healthcare and Life Sciences APIs

## ⚡ Quick Test After Authentication

```python
from tools_utilities.google_cloud_api_manager import GoogleCloudAPIManager

manager = GoogleCloudAPIManager()
status = manager.get_authentication_status()
print(f"Authenticated: {status['authenticated']}")
print(f"Project: {status['project_id']}")

# Enable an API
success, message = manager.enable_api("vision")
print(message)
```
