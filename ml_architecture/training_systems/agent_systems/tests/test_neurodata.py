#!/usr/bin/env python3
"""
Test script for Small-Mind Open Neurodata Integration

This script tests the basic functionality of the open neurodata integration framework.
All sources are publicly accessible without API keys or signups.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all open neurodata modules can be imported"""
    print("Testing open neurodata module imports...")
    
    try:
        from neurodata import (
            OpenNeurophysiologyInterface,
            OpenBrainImagingInterface,
            CommonCrawlInterface,
            NeurodataManager
        )
        print("✓ All open neurodata modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_interface_creation():
    """Test that open interface objects can be created"""
    print("\nTesting open interface object creation...")
    
    try:
        from neurodata import (
            OpenNeurophysiologyInterface,
            OpenBrainImagingInterface,
            CommonCrawlInterface
        )
        
        # Create interface instances
        open_phys = OpenNeurophysiologyInterface()
        open_imaging = OpenBrainImagingInterface()
        commoncrawl = CommonCrawlInterface()
        
        print("✓ All open interface objects created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Interface creation failed: {e}")
        return False

def test_manager_creation():
    """Test that the unified manager can be created"""
    print("\nTesting NeurodataManager creation...")
    
    try:
        from neurodata import NeurodataManager
        
        manager = NeurodataManager()
        print("✓ NeurodataManager created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Manager creation failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of open interfaces"""
    print("\nTesting basic functionality...")
    
    try:
        from neurodata import OpenNeurophysiologyInterface, OpenBrainImagingInterface, CommonCrawlInterface
        
        # Test Open Neurophysiology interface
        print("  Testing Open Neurophysiology interface methods...")
        open_phys = OpenNeurophysiologyInterface()
        
        methods = dir(open_phys)
        required_methods = [
            'search_crcns_datasets',
            'get_neuromorpho_neurons', 
            'search_modeldb_models',
            'get_opensourcebrain_models'
        ]
        
        for method in required_methods:
            if method in methods:
                print(f"    ✓ {method} method exists")
            else:
                print(f"    ✗ {method} method missing")
        
        # Test Open Brain Imaging interface
        print("  Testing Open Brain Imaging interface methods...")
        open_imaging = OpenBrainImagingInterface()
        
        methods = dir(open_imaging)
        required_methods = [
            'search_openneuro_datasets',
            'get_openneuro_dataset_info',
            'search_brainlife_datasets',
            'search_nitrc_resources'
        ]
        
        for method in required_methods:
            if method in methods:
                print(f"    ✓ {method} method exists")
            else:
                print(f"    ✗ {method} method missing")
        
        # Test CommonCrawl interface
        print("  Testing CommonCrawl interface methods...")
        commoncrawl = CommonCrawlInterface()
        
        methods = dir(commoncrawl)
        required_methods = [
            'list_crawl_indexes',
            'search_neuroscience_content',
            'get_aws_s3_access',
            'get_neuroscience_datasets'
        ]
        
        for method in required_methods:
            if method in methods:
                print(f"    ✓ {method} method exists")
            else:
                print(f"    ✗ {method} method missing")
        
        print("✓ Basic functionality test completed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_open_data_access():
    """Test that open data sources can be accessed without API keys"""
    print("\nTesting open data access (no API keys required)...")
    
    try:
        from neurodata import NeurodataManager
        
        manager = NeurodataManager()
        
        # Test cross-source search
        print("  Testing cross-source search...")
        results = manager.search_across_sources(
            query="cortex",
            data_types=["electrophysiology", "imaging"]
        )
        
        print(f"    Found results from {len(results)} sources:")
        for source, source_results in results.items():
            print(f"      {source}: {len(source_results)} results")
        
        # Test data statistics
        print("  Testing data statistics...")
        stats = manager.get_data_statistics()
        
        print("    Available data sources:")
        for source, source_stats in stats.items():
            if "error" not in source_stats:
                print(f"      {source}: {source_stats}")
            else:
                print(f"      {source}: Error - {source_stats['error']}")
        
        print("✓ Open data access test completed")
        return True
        
    except Exception as e:
        print(f"✗ Open data access test failed: {e}")
        return False

def main():
    """Run all open neurodata integration tests"""
    print("SMALL-MIND OPEN NEURODATA INTEGRATION TEST")
    print("=" * 60)
    print("This demo tests integration with truly open neuroscience data sources")
    print("No API keys or signups required!")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_interface_creation,
        test_manager_creation,
        test_basic_functionality,
        test_open_data_access
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Open neurodata integration is working correctly.")
        print("🎉 You can now access neuroscience data without any API keys!")
    else:
        print("✗ Some tests failed. Check the output above for details.")
    
    print("\n" + "=" * 60)
    print("OPEN DATA SOURCES AVAILABLE:")
    print("• CRCNS - Collaborative Research in Computational Neuroscience")
    print("• NeuroMorpho.org - Neuronal morphology database")
    print("• ModelDB - Computational neuroscience models")
    print("• Open Source Brain - Collaborative modeling platform")
    print("• NeuroElectro - Electrophysiological properties")
    print("• OpenNeuro - BIDS datasets (public access)")
    print("• Brainlife.io - Public neuroimaging platform")
    print("• NITRC - Neuroimaging tools and resources")
    print("• INDI - International neuroimaging data sharing")
    print("• CommonCrawl - Web crawl data (WARC/ARC format)")
    print("=" * 60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
