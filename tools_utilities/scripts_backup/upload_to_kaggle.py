#!/usr/bin/env python3
"""
Kaggle Upload Script for Quark Brain Simulation Framework
Automates the process of uploading notebooks and datasets to Kaggle

Purpose: Streamline the upload process for Kaggle integration
Inputs: Notebook files, datasets, competition files
Outputs: Upload status, Kaggle URLs, download instructions
Seeds: Kaggle API configuration, file paths
Dependencies: kaggle, os, json
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

class KaggleUploader:
    """Handles uploading Quark brain simulation files to Kaggle"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.notebooks_dir = self.project_root / "notebooks" / "kaggle_integration"
        self.competitions_dir = self.project_root / "competitions"
        self.upload_log = []
        
    def check_kaggle_installation(self):
        """Check if Kaggle CLI is properly installed and configured"""
        try:
            result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Kaggle CLI is installed")
                return True
            else:
                print("❌ Kaggle CLI not found")
                return False
        except FileNotFoundError:
            print("❌ Kaggle CLI not found. Please install with: pip install kaggle")
            return False
    
    def check_kaggle_auth(self):
        """Check if Kaggle API is authenticated"""
        try:
            result = subprocess.run(['kaggle', 'datasets', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Kaggle API authentication successful")
                return True
            else:
                print("❌ Kaggle API authentication failed")
                print("Please ensure your kaggle.json is in ~/.kaggle/")
                return False
        except Exception as e:
            print(f"❌ Error checking Kaggle auth: {e}")
            return False
    
    def list_uploadable_files(self):
        """List all files ready for Kaggle upload"""
        files = []
        
        # Notebooks
        notebook_files = [
            "dna_consciousness_training.ipynb",
            "unified_kaggle_training.ipynb"
        ]
        
        for notebook in notebook_files:
            notebook_path = self.notebooks_dir / notebook
            if notebook_path.exists():
                files.append({
                    'type': 'notebook',
                    'path': str(notebook_path),
                    'name': notebook,
                    'description': f'Quark Brain Simulation - {notebook.replace(".ipynb", "").replace("_", " ").title()}'
                })
        
        # Competition files
        competition_path = self.competitions_dir / "brain-simulation-benchmark-2025"
        if competition_path.exists():
            for file_type in ['data', 'evaluation', 'submissions', 'notebooks']:
                type_path = competition_path / file_type
                if type_path.exists():
                    for file in type_path.glob("*"):
                        if file.is_file():
                            files.append({
                                'type': f'competition_{file_type}',
                                'path': str(file),
                                'name': file.name,
                                'description': f'Brain Simulation Competition - {file_type.title()} - {file.name}'
                            })
        
        return files
    
    def create_upload_instructions(self):
        """Create detailed upload instructions"""
        instructions = """
# 🚀 Kaggle Upload Instructions for Quark Brain Simulation

## Prerequisites
1. Kaggle account: https://www.kaggle.com/
2. Kaggle API key: https://www.kaggle.com/settings/account
3. Kaggle CLI: `pip install kaggle`

## Step 1: Upload DNA Training Notebook
1. Go to https://www.kaggle.com/code
2. Click "Create" → "New Notebook"
3. Upload: `notebooks/kaggle_integration/dna_consciousness_training.ipynb`
4. Enable GPU: Settings → Accelerator → GPU
5. Run all cells
6. Download trained model and results

## Step 2: Upload Unified Training Notebook
1. Create new notebook on Kaggle
2. Upload: `notebooks/kaggle_integration/unified_kaggle_training.ipynb`
3. Enable GPU
4. Run different simulation types
5. Download results

## Step 3: Launch Brain Simulation Competition
1. Go to https://www.kaggle.com/competitions
2. Click "Create Competition"
3. Upload datasets from `competitions/brain-simulation-benchmark-2025/data/`
4. Set up evaluation metrics
5. Launch competition

## Step 4: Download Results
1. Download trained models (.pth files)
2. Download training results (.json files)
3. Download visualizations (.png files)
4. Integrate back into local consciousness agent

## Expected Results
- GPU-accelerated training (10-50x faster)
- Higher accuracy models
- Community benchmarking
- Research collaboration opportunities
"""
        
        with open(self.project_root / "KAGGLE_UPLOAD_INSTRUCTIONS.md", "w") as f:
            f.write(instructions)
        
        print("✅ Upload instructions saved to KAGGLE_UPLOAD_INSTRUCTIONS.md")
        return instructions
    
    def create_quick_upload_script(self):
        """Create a quick script for automated uploads"""
        script_content = """#!/bin/bash
# Quick Kaggle Upload Script for Quark Brain Simulation

echo "🚀 Starting Kaggle upload process..."

# Check Kaggle installation
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check authentication
if ! kaggle datasets list &> /dev/null; then
    echo "❌ Kaggle authentication failed. Please check your kaggle.json"
    exit 1
fi

echo "✅ Kaggle setup verified"

# Create datasets for competition files
echo "📊 Creating competition datasets..."

# Upload competition data
kaggle datasets create -p competitions/brain-simulation-benchmark-2025/data --title "brain-simulation-competition-data"

echo "✅ Competition data uploaded"

# Instructions for notebook upload
echo ""
echo "📓 Next steps for notebook upload:"
echo "1. Go to https://www.kaggle.com/code"
echo "2. Upload dna_consciousness_training.ipynb"
echo "3. Enable GPU and run training"
echo "4. Download results and integrate back"
echo ""
echo "🎉 Upload process complete!"
"""
        
        script_path = self.project_root / "scripts" / "quick_kaggle_upload.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"✅ Quick upload script created: {script_path}")
        return script_path
    
    def generate_upload_summary(self):
        """Generate a summary of what needs to be uploaded"""
        files = self.list_uploadable_files()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "file_types": {},
            "upload_priority": [
                "dna_consciousness_training.ipynb",
                "unified_kaggle_training.ipynb", 
                "competition_data",
                "competition_evaluation"
            ],
            "files": files
        }
        
        # Count file types
        for file in files:
            file_type = file['type']
            if file_type not in summary['file_types']:
                summary['file_types'][file_type] = 0
            summary['file_types'][file_type] += 1
        
        # Save summary
        with open(self.project_root / "kaggle_upload_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Upload summary saved to kaggle_upload_summary.json")
        return summary
    
    def run_full_setup(self):
        """Run the complete Kaggle upload setup"""
        print("🚀 Setting up Kaggle upload for Quark Brain Simulation")
        print("=" * 60)
        
        # Check prerequisites
        if not self.check_kaggle_installation():
            return False
        
        if not self.check_kaggle_auth():
            return False
        
        # Generate upload files
        self.create_upload_instructions()
        self.create_quick_upload_script()
        summary = self.generate_upload_summary()
        
        # Print summary
        print("\n📋 Upload Summary:")
        print(f"Total files ready: {summary['total_files']}")
        print("\nFile types:")
        for file_type, count in summary['file_types'].items():
            print(f"  - {file_type}: {count} files")
        
        print("\n🎯 Priority upload order:")
        for i, priority in enumerate(summary['upload_priority'], 1):
            print(f"  {i}. {priority}")
        
        print("\n📁 Files ready for upload:")
        for file in summary['files'][:5]:  # Show first 5
            print(f"  - {file['name']} ({file['type']})")
        
        if len(summary['files']) > 5:
            print(f"  ... and {len(summary['files']) - 5} more files")
        
        print("\n✅ Kaggle upload setup complete!")
        print("📖 See KAGGLE_UPLOAD_INSTRUCTIONS.md for detailed steps")
        print("🚀 Run scripts/quick_kaggle_upload.sh for automated upload")
        
        return True

def main():
    """Main function to run Kaggle upload setup"""
    uploader = KaggleUploader()
    success = uploader.run_full_setup()
    
    if success:
        print("\n🎉 Ready to upload to Kaggle!")
        print("Next steps:")
        print("1. Follow KAGGLE_UPLOAD_INSTRUCTIONS.md")
        print("2. Upload notebooks to Kaggle")
        print("3. Run GPU training")
        print("4. Download results and integrate")
    else:
        print("\n❌ Setup incomplete. Please check Kaggle installation and authentication.")

if __name__ == "__main__":
    main()
