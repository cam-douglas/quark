#!/usr/bin/env python3
"""
Test FlyWire Visual Server Integration

This script tests the integration between FlyWire data and the visual server
infrastructure to ensure everything works correctly.
"""

import os, sys
import time
import requests
import json
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flywire_integration import FlyWireDataManager

def test_flywire_data_loading():
    """Test FlyWire data loading functionality."""
    print("🧪 Testing FlyWire Data Loading...")
    
    try:
        # Initialize data manager
        data_manager = FlyWireDataManager()
        
        # Test data creation
        print("  Creating sample data...")
        success = data_manager.download_sample_data()
        if not success:
            print("  ❌ Failed to create sample data")
            return False
        
        # Test data loading
        print("  Loading data...")
        success = data_manager.load_data()
        if not success:
            print("  ❌ Failed to load data")
            return False
        
        # Test data access
        print("  Accessing data...")
        stats = data_manager.get_network_statistics()
        if not stats:
            print("  ❌ Failed to get network statistics")
            return False
        
        print(f"  ✅ Data loaded successfully:")
        print(f"    Neurons: {stats['total_neurons']:,}")
        print(f"    Connections: {stats['total_neurons']:,}")
        print(f"    Cell types: {len(stats['cell_type_distribution'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_visual_server_api():
    """Test the visual server API endpoints."""
    print("\n🌐 Testing Visual Server API...")
    
    base_url = "http://localhost:5000"
    
    try:
        # Test server status
        print("  Testing server status...")
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            print(f"  ✅ Server status: {status_data}")
        else:
            print(f"  ❌ Server status failed: {response.status_code}")
            return False
        
        # Test data loading
        print("  Testing data loading...")
        response = requests.post(f"{base_url}/api/load_data", timeout=30)
        if response.status_code == 200:
            load_data = response.json()
            if load_data.get('success'):
                print("  ✅ Data loaded successfully")
                print(f"    Stats: {load_data.get('stats', {})}")
            else:
                print(f"  ❌ Data loading failed: {load_data.get('error')}")
                return False
        else:
            print(f"  ❌ Data loading request failed: {response.status_code}")
            return False
        
        # Test 3D visualization
        print("  Testing 3D visualization...")
        response = requests.get(f"{base_url}/api/visualization/3d", timeout=10)
        if response.status_code == 200:
            viz_data = response.json()
            if viz_data.get('success'):
                print("  ✅ 3D visualization generated successfully")
                print(f"    Neurons visualized: {viz_data.get('neuron_count')}")
            else:
                print(f"  ❌ 3D visualization failed: {viz_data.get('error')}")
                return False
        else:
            print(f"  ❌ 3D visualization request failed: {response.status_code}")
            return False
        
        # Test network topology
        print("  Testing network topology...")
        response = requests.get(f"{base_url}/api/visualization/topology", timeout=10)
        if response.status_code == 200:
            viz_data = response.json()
            if viz_data.get('success'):
                print("  ✅ Network topology generated successfully")
                print(f"    Neurons visualized: {viz_data.get('neuron_count')}")
            else:
                print(f"  ❌ Network topology failed: {viz_data.get('error')}")
                return False
        else:
            print(f"  ❌ Network topology request failed: {response.status_code}")
            return False
        
        # Test cell type distribution
        print("  Testing cell type distribution...")
        response = requests.get(f"{base_url}/api/visualization/cell_types", timeout=10)
        if response.status_code == 200:
            viz_data = response.json()
            if viz_data.get('success'):
                print("  ✅ Cell type distribution generated successfully")
                cell_types = viz_data.get('cell_type_counts', {})
                print(f"    Cell types: {len(cell_types)}")
            else:
                print(f"  ❌ Cell type distribution failed: {viz_data.get('error')}")
                return False
        else:
            print(f"  ❌ Cell type distribution request failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("  ❌ Could not connect to visual server")
        print("    Make sure the server is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_visualization_components():
    """Test the visualization components directly."""
    print("\n📊 Testing Visualization Components...")
    
    try:
        # Load data
        data_manager = FlyWireDataManager()
        data_manager.download_sample_data()
        data_manager.load_data()
        
        # Import visualizer
        from flywire_simple_visual_server import FlyWireSimpleVisualizer
        
        # Create visualizer
        visualizer = FlyWireSimpleVisualizer(data_manager)
        
        # Test 3D visualization
        print("  Testing 3D brain visualization...")
        result = visualizer.create_3d_brain_visualization()
        if result.get('success'):
            print("  ✅ 3D visualization created successfully")
            print(f"    Neurons visualized: {result.get('neuron_count')}")
        else:
            print(f"  ❌ 3D visualization failed: {result.get('error')}")
            return False
        
        # Test network topology
        print("  Testing network topology...")
        result = visualizer.create_network_topology()
        if result.get('success'):
            print("  ✅ Network topology created successfully")
            print(f"    Neurons visualized: {result.get('neuron_count')}")
        else:
            print(f"  ❌ Network topology failed: {result.get('error')}")
            return False
        
        # Test cell type distribution
        print("  Testing cell type distribution...")
        result = visualizer.create_cell_type_distribution()
        if result.get('success'):
            print("  ✅ Cell type distribution created successfully")
            cell_types = result.get('cell_type_counts', {})
            print(f"    Cell types: {len(cell_types)}")
        else:
            print(f"  ❌ Cell type distribution failed: {result.get('error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_web_interface():
    """Test the web interface accessibility."""
    print("\n🌍 Testing Web Interface...")
    
    base_url = "http://localhost:5000"
    
    try:
        # Test main page
        print("  Testing main page...")
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("  ✅ Main page accessible")
            if "FlyWire Brain Visualization" in response.text:
                print("  ✅ Page content loaded correctly")
            else:
                print("  ⚠️  Page content may not be loading correctly")
        else:
            print(f"  ❌ Main page failed: {response.status_code}")
            return False
        
        # Test HTML template
        print("  Testing HTML template...")
        template_path = "templates/flywire_simple.html"
        if os.path.exists(template_path):
            print("  ✅ HTML template exists")
            with open(template_path, 'r') as f:
                content = f.read()
                if "Plotly" in content and "FlyWire" in content:
                    print("  ✅ HTML template content is correct")
                else:
                    print("  ⚠️  HTML template content may be incomplete")
        else:
            print("  ❌ HTML template not found")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("  ❌ Could not connect to web interface")
        print("    Make sure the server is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧠 FlyWire Visual Server Integration Test")
    print("=" * 50)
    
    tests = [
        ("FlyWire Data Loading", test_flywire_data_loading),
        ("Visual Server API", test_visual_server_api),
        ("Visualization Components", test_visualization_components),
        ("Web Interface", test_web_interface)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The FlyWire Visual Server integration is working correctly.")
        print("\n🚀 Next steps:")
        print("1. Open http://localhost:5000 in your browser")
        print("2. Click 'Load FlyWire Data' to initialize the system")
        print("3. Explore the interactive visualizations")
        print("4. Integrate with your existing neuroscience workflows")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that the visual server is running")
        print("3. Verify FlyWire data files exist")
        print("4. Check server logs for error messages")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
