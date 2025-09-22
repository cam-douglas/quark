#!/bin/bash
# Setup and run training on Google Compute Engine VM

set -e

echo "🚀 Setting up Quark training on Compute Engine VM..."

# Copy our training files to the VM
echo "📁 Copying training files..."
gcloud compute scp --recurse /Users/camdouglas/quark/brain/ quark-training-vm:~/brain/ --zone=us-central1-a
gcloud compute scp /Users/camdouglas/quark/requirements.txt quark-training-vm:~/requirements.txt --zone=us-central1-a
gcloud compute scp /Users/camdouglas/quark/data/credentials/quark-469604-training-sa.json quark-training-vm:~/service-account.json --zone=us-central1-a

# Setup script to run on the VM
cat > vm_setup.sh << 'EOF'
#!/bin/bash
set -e

echo "🔧 Setting up Python environment..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

echo "📦 Creating virtual environment..."
python3 -m venv training_env
source training_env/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision tensorboard scipy scikit-learn numpy pandas matplotlib seaborn h5py
pip install google-cloud-storage google-auth

echo "🔐 Setting up authentication..."
export GOOGLE_APPLICATION_CREDENTIALS=~/service-account.json
gcloud auth activate-service-account --key-file=~/service-account.json

echo "🧠 Starting training..."
cd ~/brain/modules/brainstem_segmentation/
python run_gcp_training.py \
  --epochs 50 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --num_samples 100 \
  --grid_size 32 \
  --model_dir gs://quark-models/brainstem_segmentation \
  --log_dir gs://quark-experiments/brainstem_training

echo "✅ Training completed!"
EOF

# Copy and run the setup script on the VM
echo "🚀 Running setup on VM..."
gcloud compute scp vm_setup.sh quark-training-vm:~/vm_setup.sh --zone=us-central1-a
gcloud compute ssh quark-training-vm --zone=us-central1-a --command="chmod +x ~/vm_setup.sh && ~/vm_setup.sh"

echo "✅ VM training setup complete!"
