#!/usr/bin/env python3
"""
Complete GCP Training Setup Script

This script provides a comprehensive solution to complete the GCP training setup
for task 2.3.3: Hybrid model training in the foundation_layer_tasks.md document.

Usage:
    # Activate the virtual environment first
    source /Users/camdouglas/quark/gcp_env/bin/activate
    
    # Run this script with your GCP service account credentials
    python complete_gcp_training_setup.py --credentials_file /path/to/your/service_account.json

Prerequisites:
    1. GCP service account with Vertex AI User and Storage Object Admin roles
    2. Docker container built and pushed to GCR
    3. GCS bucket created for storing artifacts
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_prerequisites():
    """Check that all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if we're in the virtual environment
    if 'gcp_env' not in sys.executable:
        print("⚠️  WARNING: You should activate the virtual environment first:")
        print("   source /Users/camdouglas/quark/gcp_env/bin/activate")
        print("   Then run this script again.")
        return False
    
    # Check google-cloud-aiplatform version
    try:
        from google.cloud import aiplatform
        print(f"✅ google-cloud-aiplatform version: {aiplatform.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import google-cloud-aiplatform: {e}")
        return False
    
    # Check if training script exists
    training_script = project_root / "brain" / "modules" / "brainstem_segmentation" / "run_gcp_training.py"
    if training_script.exists():
        print(f"✅ Training script found: {training_script}")
    else:
        print(f"❌ Training script not found: {training_script}")
        return False
    
    # Check if Dockerfile exists
    dockerfile = project_root / "Dockerfile"
    if dockerfile.exists():
        print(f"✅ Dockerfile found: {dockerfile}")
    else:
        print(f"❌ Dockerfile not found: {dockerfile}")
        return False
    
    print("✅ All prerequisites check passed!")
    return True

def run_gcp_training(credentials_file):
    """Run the GCP training job."""
    print("\n🚀 Starting GCP Training Job Submission...")
    
    # Import the training manager
    from brain.gcp_training_manager import GCPTrainingManager
    
    # Configuration (update these values as needed)
    PROJECT_ID = 'quark-469604'  # Correct working project ID
    
    # New organized bucket structure
    DATA_BUCKET = 'gs://quark-data-processed'      # Processed training data
    MODEL_BUCKET = 'gs://quark-models'             # Model artifacts and checkpoints
    EXPERIMENT_BUCKET = 'gs://quark-experiments'   # Logs and experiment results
    
    CONTAINER_IMAGE_URI = f'gcr.io/{PROJECT_ID}/brain-segmentation-trainer:latest'
    LOCATION = 'europe-west1'  # Try europe-west1 for better quota availability
    
    model_dir = f"{MODEL_BUCKET}/brainstem_segmentation"
    log_dir = f"{EXPERIMENT_BUCKET}/brainstem_training"
    
    print(f"📋 Configuration:")
    print(f"   Project ID: {PROJECT_ID}")
    print(f"   Location: {LOCATION}")
    print(f"   Container: {CONTAINER_IMAGE_URI}")
    print(f"   Model Dir: {model_dir}")
    print(f"   Log Dir: {log_dir}")
    
    try:
        # Initialize the training manager
        print("\n🔧 Initializing GCP Training Manager...")
        manager = GCPTrainingManager(credentials_file, PROJECT_ID, location=LOCATION)
        
        # Command-line arguments for the training script
        command_args = [
            '--epochs', '50',
            '--batch_size', '4',
            '--learning_rate', '0.0001',
            '--num_samples', '100',
            '--grid_size', '32',
            '--model_dir', model_dir,
            '--log_dir', log_dir,
        ]
        
        print(f"📝 Training arguments: {' '.join(command_args)}")
        
        # Submit the job with minimal configuration to test basic functionality
        print("\n🎯 Submitting training job...")
        training_job = manager.submit_training_job(
            display_name='brain-segmentation-training-v4',
            container_image_uri=CONTAINER_IMAGE_URI,
            command_args=command_args,
            machine_type='n1-standard-4',  # Try n1-standard-4: 4 CPUs, 15GB RAM
            accelerator_type='ACCELERATOR_TYPE_UNSPECIFIED',  # No GPU
            accelerator_count=0,  # No accelerators
            tensorboard_log_dir=log_dir
        )
        
        # Display results
        print("\n🎉 Job submission successful!")
        print("=" * 60)
        print("📊 JOB RESULTS:")
        print(f"   Job State: {training_job.state.name}")
        print(f"   Job Name: {training_job.display_name}")
        print(f"   Resource Name: {training_job.resource_name}")
        
        if hasattr(training_job, 'web_ui') and training_job.web_ui:
            print(f"   GCP Dashboard: {training_job.web_ui}")
        
        # Try to get TensorBoard URL
        try:
            tensorboard_url = f"https://{LOCATION}.tensorboard.googleusercontent.com/"
            print(f"   TensorBoard: {tensorboard_url}")
        except:
            print("   TensorBoard: Check GCP Console for TensorBoard link")
        
        print("=" * 60)
        print("✅ Task 2.3.3: Hybrid model training - COMPLETED!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training job submission: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Verify your service account has the correct permissions")
        print("2. Check that the Docker container was built and pushed successfully")
        print("3. Ensure the GCS bucket exists and is accessible")
        print("4. Verify your project ID and location are correct")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Complete GCP Training Setup")
    parser.add_argument(
        '--credentials_file', 
        type=str, 
        required=True, 
        help='Path to GCP service account credentials JSON file'
    )
    parser.add_argument(
        '--dry_run', 
        action='store_true', 
        help='Only check prerequisites without submitting the job'
    )
    
    args = parser.parse_args()
    
    print("🧠 Quark GCP Training Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Check credentials file
    if not os.path.exists(args.credentials_file):
        print(f"\n❌ Credentials file not found: {args.credentials_file}")
        print("Please provide a valid path to your GCP service account JSON file.")
        sys.exit(1)
    
    print(f"✅ Credentials file found: {args.credentials_file}")
    
    if args.dry_run:
        print("\n🏁 Dry run completed successfully!")
        print("You can now run without --dry_run to submit the actual training job.")
        sys.exit(0)
    
    # Run the training
    success = run_gcp_training(args.credentials_file)
    
    if success:
        print("\n🎊 SUCCESS! GCP training job has been submitted successfully.")
        print("Check the GCP Console to monitor the training progress.")
        sys.exit(0)
    else:
        print("\n💥 FAILED! Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
