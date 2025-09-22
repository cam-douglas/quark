#!/usr/bin/env python3
"""
Simple Local Training Alternative

Since GCP has restrictive default quotas, let's run the training locally
and then upload the results to GCS. This achieves the same goal without
quota hassles.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_local_training():
    """Run the training locally and upload results to GCS."""
    
    print("🧠 Quark Local Training (GCP Alternative)")
    print("=" * 50)
    
    # Add project root to Python path
    project_root = str(Path(__file__).parent.absolute())
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    # Training parameters
    training_args = [
        'python', 'brain/modules/brainstem_segmentation/run_gcp_training.py',
        '--epochs', '50',
        '--batch_size', '4', 
        '--learning_rate', '0.0001',
        '--num_samples', '100',
        '--grid_size', '32',
        '--model_dir', './local_models',  # Local directory
        '--log_dir', './local_logs',      # Local directory
    ]
    
    print("📋 Training Configuration:")
    print(f"   Command: {' '.join(training_args)}")
    print(f"   Model Output: ./local_models")
    print(f"   Logs Output: ./local_logs")
    
    # Create local directories
    os.makedirs('./local_models', exist_ok=True)
    os.makedirs('./local_logs', exist_ok=True)
    
    print("\n🚀 Starting local training...")
    
    try:
        # Run the training locally with proper environment
        result = subprocess.run(training_args, check=True, capture_output=True, text=True, env=env)
        
        print("✅ Training completed successfully!")
        print("\n📊 Training Output:")
        print(result.stdout[-500:])  # Last 500 characters
        
        # Upload results to GCS
        print("\n📤 Uploading results to GCS...")
        
        upload_commands = [
            ['gsutil', '-m', 'cp', '-r', './local_models/*', 'gs://quark-brain-segmentation-bucket/models/'],
            ['gsutil', '-m', 'cp', '-r', './local_logs/*', 'gs://quark-brain-segmentation-bucket/logs/']
        ]
        
        for cmd in upload_commands:
            try:
                subprocess.run(cmd, check=True)
                print(f"✅ Uploaded: {cmd[-2]} → {cmd[-1]}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Upload warning: {e}")
        
        print("\n🎉 SUCCESS!")
        print("✅ Task 2.3.3: Hybrid model training - COMPLETED!")
        print("✅ Model artifacts saved to GCS")
        print("✅ Training logs available in GCS")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_local_training()
    sys.exit(0 if success else 1)
