#!/bin/bash

# ==============================================================================
# GCP VM Training Automation Script
# ==============================================================================
# This script automates the creation of a Google Compute Engine VM,
# sets up the environment, clones the repository, and runs the valid
# GNN-ViT Hybrid training script.
#
# Prerequisites:
#   - Google Cloud SDK (`gcloud`) installed and configured.
#   - An active GCP project with billing enabled.
#   - The gcloud user must have permissions to create VMs and manage them.
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
VM_NAME="quark-valid-training-vm"
ZONE="us-west1-a"
MACHINE_TYPE="e2-standard-2"
IMAGE_FAMILY="debian-11"
IMAGE_PROJECT="debian-cloud"
# ACCELERATOR_TYPE="nvidia-tesla-p100" # No GPU for this run
# ACCELERATOR_COUNT="1" # No GPU for this run
PROJECT_ID=$(gcloud config get-value project)
REPO_URL="https://github.com/cam-douglas/quark" # IMPORTANT: CHANGE THIS

# --- Startup Script ---
# This script will be executed by the VM on its first boot.
STARTUP_SCRIPT=$(cat <<EOF
#!/bin/bash
echo "🚀 Starting VM setup..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y git python3-pip

# Clone the repository
git clone ${REPO_URL} /home/jupyter/quark
cd /home/jupyter/quark

# Install Python dependencies
pip install -r requirements.txt

# Run the training script
echo "🧠 Starting the valid training script on CPU..."
python brain/modules/brainstem_segmentation/valid_gcp_training.py \
    --epochs 5 \
    --batch_size 2 \
    --device cpu \
    --model_dir 'gs://${PROJECT_ID}-models/brainstem_segmentation' \
    --log_dir 'gs://${PROJECT_ID}-logs/brainstem_training'

echo "✅ Training script finished."

# Optional: Shut down the VM after training to save costs
# sudo poweroff
EOF
)

# --- Main Execution ---
echo "🤖 Starting GCP VM creation process..."

# 1. Create the VM
echo "   1. Creating VM: ${VM_NAME}..."
gcloud compute instances create "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --maintenance-policy=TERMINATE \
    --metadata="startup-script=${STARTUP_SCRIPT}" \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo "   ✅ VM '${VM_NAME}' created successfully."

# 2. Monitor the startup script (optional but recommended)
echo "   2. Tailing startup script logs. Press Ctrl+C to exit."
echo "      You can also view logs in the GCP Console (VM Instances -> Serial port 1 (console))."
gcloud compute instances tail-serial-port-output "${VM_NAME}" --zone="${ZONE}"

echo "🎉 Process complete. The VM will run the training and may shut down automatically if configured."