# GCP Training Blocked: GPU Quota Exceeded

## 1. Summary

The automated VM training process has failed because the Google Cloud project (`quark-469604`) has a **zero GPU quota**. This is a hard limit set on the project, meaning no virtual machines with GPUs can be launched.

## 2. The Error

The `gcloud` command failed with the following error:

```
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - Quota 'GPUS_ALL_REGIONS' exceeded.  Limit: 0.0 globally.
```

This indicates that the project is not authorized to use any GPU resources in any region.

## 3. Required Action

To resolve this, you must **request a GPU quota increase** for your GCP project. This is a manual process that must be initiated from the Google Cloud Console.

### Steps to Request a Quota Increase:

1.  **Navigate to Quotas:**
    *   Go to the Google Cloud Console.
    *   In the navigation menu, go to **IAM & Admin** > **Quotas**.

2.  **Filter for GPU Quotas:**
    *   In the "Filter" box, select **Metric**.
    *   Search for and select **"GPUs (all regions)"** (or `compute.googleapis.com/gpus_all_regions`).
    *   You can also filter for specific GPU types if you have a preference (e.g., `NVIDIA T4 GPUs`).

3.  **Request an Increase:**
    *   Select the checkbox next to the specific quota you want to increase.
    *   Click **"EDIT QUOTAS"** at the top of the page.
    *   A form will appear on the right. Fill in your contact details.
    *   In the "New limit" field, enter your desired number of GPUs. A request for **1 or 2** should be sufficient for this task.
    *   In the "Request description," explain that you need the quota for a machine learning research project.
    *   Submit the request.

## 4. Next Steps

Google will review the request, which may take several business days. **The training process is blocked until this quota increase is approved.**

Once the quota is granted, the `setup_and_run_vm_training.sh` script can be run again.