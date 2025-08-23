#!/usr/bin/env python3
"""
Safe, incremental test script for Neuroscience Domain Experts System

This script tests imports and functionality step by step to identify
exactly where any freezes or issues occur.
"""

import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def safe_import_test(module_name, import_statement, timeout=10):
    """Safely test importing a module with timeout protection"""
    print(f"🧪 Testing import: {module_name}")
    
    try:
        # Set a timeout for the import
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Import of {module_name} timed out after {timeout} seconds")
        
        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        # Execute the import
        exec(import_statement)
        
        # Clear the alarm
        signal.alarm(0)
        
        print(f"✅ {module_name} imported successfully")
        return True
        
    except TimeoutError as e:
        print(f"⏰ {e}")
        return False
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Unexpected error importing {module_name}: {e}")
        return False

def test_basic_imports():
    """Test basic Python imports first"""
    print("🧪 Testing basic Python imports...")
    
    basic_modules = [
        ("logging", "import logging"),
        ("requests", "import requests"),
        ("pandas", "import pandas"),
        ("numpy", "import numpy"),
        ("asyncio", "import asyncio"),
        ("aiohttp", "import aiohttp")
    ]
    
    for module_name, import_stmt in basic_modules:
        if not safe_import_test(module_name, import_stmt):
            print(f"❌ Basic import {module_name} failed - stopping here")
            return False
    
    print("✅ All basic imports successful")
    return True

def test_neurodata_imports():
    """Test neurodata module imports step by step"""
    print("\n🧪 Testing neurodata module imports...")
    
    neurodata_modules = [
        ("enhanced_data_resources", "from neurodata.enhanced_data_resources import EnhancedDataResources"),
        ("allen_brain", "from neurodata.allen_brain import AllenBrainInterface"),
        ("bossdb_client", "from neurodata.bossdb_client import BossDBClient"),
        ("dandi_client", "from neurodata.dandi_client import DANDIClient"),
        ("hcp_interface", "from neurodata.hcp_interface import HCPInterface"),
        ("openneuro_client", "from neurodata.openneuro_client import OpenNeuroClient")
    ]
    
    for module_name, import_stmt in neurodata_modules:
        if not safe_import_test(module_name, import_stmt):
            print(f"❌ Neurodata import {module_name} failed - stopping here")
            return False
    
    print("✅ All neurodata imports successful")
    return True

def test_models_imports():
    """Test models module imports step by step"""
    print("\n🧪 Testing models module imports...")
    
    models_modules = [
        ("model_manager", "from models.model_manager import MoEModelManager"),
        ("moe_manager", "from models.moe_manager import MoEManager"),
        ("moe_router", "from models.moe_router import HumanLikeCognitiveRouter"),
        ("neuroscience_experts", "from models.neuroscience_experts import NeuroscienceExpertManager")
    ]
    
    for module_name, import_stmt in models_modules:
        if not safe_import_test(module_name, import_stmt):
            print(f"❌ Models import {module_name} failed - stopping here")
            return False
    
    print("✅ All models imports successful")
    return True

def test_baby_agi_imports():
    """Test baby_agi module imports step by step"""
    print("\n🧪 Testing baby_agi module imports...")
    
    baby_agi_modules = [
        ("agent", "from baby_agi.agent import BabyAGIAgent"),
        ("control", "from baby_agi.control import AgentController"),
        ("runtime", "from baby_agi.runtime import AgentRuntime")
    ]
    
    for module_name, import_stmt in baby_agi_modules:
        if not safe_import_test(module_name, import_stmt):
            print(f"❌ Baby AGI import {module_name} failed - stopping here")
            return False
    
    print("✅ All baby_agi imports successful")
    return True

def main():
    """Run all tests incrementally"""
    print("🧠 SAFE NEUROSCIENCE SYSTEM TEST SUITE")
    print("=" * 50)
    print("Testing imports step by step to identify any freezes...")
    print()
    
    # Test basic imports first
    if not test_basic_imports():
        print("\n❌ Basic imports failed - system may have fundamental issues")
        return False
    
    # Test neurodata imports
    if not test_neurodata_imports():
        print("\n❌ Neurodata imports failed - check neurodata module dependencies")
        return False
    
    # Test models imports
    if not test_models_imports():
        print("\n❌ Models imports failed - check models module dependencies")
        return False
    
    # Test baby_agi imports
    if not test_baby_agi_imports():
        print("\n❌ Baby AGI imports failed - check baby_agi module dependencies")
        return False
    
    print("\n🎉 All imports successful! No freezes detected.")
    print("You can now run the full test suite safely.")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        sys.exit(1)
