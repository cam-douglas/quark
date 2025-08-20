#!/usr/bin/env python3
"""
Build script for Quark Brain Simulation package
"""

import os, sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning previous build artifacts...")
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"   Removed {path}")
            elif path.is_file():
                path.unlink()
                print(f"   Removed {path}")

def build_package():
    """Build the package"""
    print("📦 Building Quark package...")
    
    # Clean previous builds
    clean_build()
    
    # Build wheel and source distribution
    run_command("python -m build", "Building wheel and source distribution")
    
    # List built files
    dist_dir = Path("dist")
    if dist_dir.exists():
        print("\n📁 Built packages:")
        for file in dist_dir.glob("*"):
            print(f"   {file.name}")

def install_package():
    """Install the package in development mode"""
    print("🔧 Installing Quark in development mode...")
    run_command("pip install -e .", "Installing package")

def test_installation():
    """Test that the package can be imported and run"""
    print("🧪 Testing installation...")
    
    # Test import
    try:
        import quark
        print("✅ Package imports successfully")
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False
    
    # Test command line interface
    try:
        result = subprocess.run(["quark-brain", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Command line interface works")
        else:
            print(f"❌ Command line interface failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Command line test failed: {e}")
        return False
    
    return True

def main():
    """Main build process"""
    print("🚀 Quark Brain Simulation - Build Process")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Build the package
    build_package()
    
    # Install in development mode
    install_package()
    
    # Test the installation
    if test_installation():
        print("\n🎉 Build completed successfully!")
        print("\n📋 Next steps:")
        print("   • To install: pip install dist/quark-*.whl")
        print("   • To run: quark-brain --help")
        print("   • To develop: pip install -e .")
    else:
        print("\n❌ Build completed but tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
