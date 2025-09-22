# -*- coding: utf-8 -*-
"""
Model Checkpoint Validator
Loads a PyTorch model checkpoint and prints the model architecture and arguments.
"""

import torch
import argparse
import sys
# Add the path to the simplified training script to the system path
# so we can import the SimpleGNNViTHybrid class.
sys.path.append('data/experiments/brainstem_training')
from simple_vm_training import SimpleGNNViTHybrid

def validate_model(checkpoint_path):
    """
    Loads a model checkpoint and validates its architecture.
    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
    """
    print(f"🔎 Analyzing model checkpoint: {checkpoint_path}")

    try:
        # Load the checkpoint
        # In PyTorch >= 2.6, weights_only=True is the default for security.
        # We must set it to False to load the full checkpoint, including the architecture.
        # This is safe because we know the source of the file.
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        print("✅ Checkpoint loaded successfully.")

        # Extract arguments
        if 'args' in checkpoint:
            args = checkpoint['args']
            print("\n--- TRAINING ARGUMENTS ---")
            for key, value in vars(args).items():
                print(f"   {key}: {value}")
            print("--------------------------")
        else:
            print("⚠️ No training arguments found in checkpoint.")

        # Re-create the model architecture from the training script
        model = SimpleGNNViTHybrid(
            num_classes=4,  # Assuming 4 classes as per the script
            grid_size=args.grid_size if 'args' in checkpoint else 16
        )
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        print("\n✅ Model state loaded into SimpleGNNViTHybrid architecture successfully.")
        
        # Print model summary
        print("\n--- MODEL ARCHITECTURE ---")
        print(model)
        print("--------------------------")
        
        # Print total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Total Parameters: {total_params:,}")


    except Exception as e:
        print(f"❌ An error occurred during model validation: {e}")
        print("   This may be because the checkpoint was not created with the SimpleGNNViTHybrid architecture.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate a PyTorch model checkpoint.')
    parser.add_argument('checkpoint_file', type=str, help='Path to the .pth checkpoint file.')
    args = parser.parse_args()

    validate_model(args.checkpoint_file)