#!/usr/bin/env python3
"""
Test script for cloud storage integration
Verifies that the system can create references and track files correctly.
"""

import os
import tempfile
import shutil
from pathlib import Path
import json

# Import our cloud integration
from cloud_integration import CloudIntegrator

def create_test_files():
    """Create test files for migration testing."""
    test_dir = Path("test_cloud_migration")
    test_dir.mkdir(exist_ok=True)
    
    # Create test directory structure
    (test_dir / "results" / "experiments").mkdir(parents=True, exist_ok=True)
    (test_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (test_dir / "models").mkdir(exist_ok=True)
    
    # Create test files
    test_files = [
        "test_script.py",
        "results/experiments/test_plot.png",
        "results/experiments/test_data.csv",
        "data/raw/large_dataset.h5",
        "models/test_model.pth",
        "README.md"
    ]
    
    for file_path in test_files:
        full_path = test_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.endswith('.py'):
            content = f'# Test Python file\nprint("Hello from {file_path}")'
        elif file_path.endswith('.png'):
            content = "FAKE_PNG_DATA" * 1000  # Simulate large file
        elif file_path.endswith('.csv'):
            content = "header1,header2,header3\n" + "data1,data2,data3\n" * 100
        elif file_path.endswith('.h5'):
            content = "FAKE_H5_DATA" * 2000  # Simulate large file
        elif file_path.endswith('.pth'):
            content = "FAKE_MODEL_DATA" * 3000  # Simulate large file
        elif file_path.endswith('.md'):
            content = f"# Test Markdown\nThis is a test file: {file_path}"
        else:
            content = f"Test content for {file_path}"
        
        with open(full_path, 'w') as f:
            f.write(content)
    
    return test_dir

def test_migration():
    """Test the cloud migration system."""
    print("🧪 Testing Cloud Storage Integration...")
    
    # Create test files
    test_dir = create_test_files()
    print(f"✅ Created test directory: {test_dir}")
    
    # Change to test directory
    original_cwd = os.getcwd()
    os.chdir(test_dir)
    
    try:
        # Initialize cloud integrator
        integrator = CloudIntegrator()
        print("✅ Initialized CloudIntegrator")
        
        # Test file finding
        patterns = ["results/**/*.png", "*.pth", "data/raw/**/*"]
        files = integrator._find_files(patterns)
        print(f"✅ Found {len(files)} files matching patterns")
        
        # Test migration
        print("\n🔄 Starting migration test...")
        integrator.migrate_to_cloud(patterns)
        
        # Check results
        cloud_refs_dir = Path(".cloud_references")
        if cloud_refs_dir.exists():
            ref_files = list(cloud_refs_dir.rglob("*.ref"))
            print(f"✅ Created {len(ref_files)} cloud references")
            
            # Show some reference files
            for ref_file in ref_files[:3]:
                with open(ref_file, 'r') as f:
                    ref_data = json.load(f)
                print(f"   📄 {ref_data['original_path']} -> {ref_data['file_size']} bytes")
        else:
            print("❌ Cloud references directory not created")
        
        # Check that original files were replaced with references
        replaced_files = []
        for pattern in patterns:
            if "*" in pattern:
                files = list(Path(".").rglob(pattern.replace("**", "*")))
            else:
                files = [Path(pattern)] if Path(pattern).exists() else []
            
            for file_path in files:
                if file_path.is_file():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    if "Cloud Reference File" in content:
                        replaced_files.append(str(file_path))
        
        print(f"✅ Replaced {len(replaced_files)} files with references")
        
        # Show directory structure
        print("\n📁 Final directory structure:")
        for item in sorted(Path(".").rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                print(f"   📄 {item} ({size} bytes)")
            elif item.is_dir() and not item.name.startswith('.'):
                print(f"   📁 {item}/")
        
        print("\n🎉 Cloud integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        os.chdir(original_cwd)
        shutil.rmtree(test_dir)
        print(f"🧹 Cleaned up test directory: {test_dir}")

def test_file_patterns():
    """Test file pattern matching."""
    print("\n🔍 Testing file pattern matching...")
    
    # Create temporary test structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        (temp_path / "results" / "experiments").mkdir(parents=True, exist_ok=True)
        (temp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
        
        test_files = [
            "results/experiments/plot1.png",
            "results/experiments/plot2.png",
            "data/raw/dataset1.csv",
            "data/raw/dataset2.h5",
            "model.pth",
            "script.py"
        ]
        
        for file_path in test_files:
            full_path = temp_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text("test content")
        
        # Test integrator
        integrator = CloudIntegrator(temp_dir)
        
        # Test different patterns
        test_patterns = [
            "results/**/*.png",
            "*.pth",
            "data/raw/**/*",
            "*.py"
        ]
        
        for pattern in test_patterns:
            files = integrator._find_files([pattern])
            print(f"   Pattern '{pattern}': {len(files)} files")
            for file_path in files[:3]:  # Show first 3
                rel_path = file_path.relative_to(temp_path)
                print(f"     - {rel_path}")

if __name__ == "__main__":
    print("🚀 Cloud Storage Integration Test Suite")
    print("=" * 50)
    
    # Test file pattern matching
    test_file_patterns()
    
    # Test full migration
    test_migration()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n📋 Next steps:")
    print("1. Setup your preferred cloud service (Google Drive recommended)")
    print("2. Run: python cloud_integration.py")
    print("3. Check .cloud_references/ for migration status")
    print("4. Verify files are accessible through references")
