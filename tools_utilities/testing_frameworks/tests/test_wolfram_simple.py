#!/usr/bin/env python3
"""
Simple Wolfram Alpha API Test
============================

Quick test to verify your Wolfram Alpha integration is working.
"""

import requests
import urllib.parse
import xml.etree.ElementTree as ET
import json
from datetime import datetime

# Your Wolfram Alpha credentials
APP_ID = "TYW5HL7G68"
BASE_URL = "http://api.wolframalpha.com/v2"

def test_wolfram_basic():
    """Test basic Wolfram Alpha connectivity"""
    print("🧠 Testing Wolfram Alpha Integration for Quark Brain Simulation")
    print("=" * 60)
    print(f"App ID: {APP_ID}")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    # Test queries relevant to brain simulation
    test_queries = [
        "2+2",  # Basic test
        "eigenvalues of {{1, 0.5}, {0.5, 1}}",  # Matrix analysis
        "solve y' = -y",  # Differential equation
        "integrate sin(x) dx",  # Mathematical computation
        "Hodgkin-Huxley equation"  # Neuroscience query
    ]
    
    results = []
    
    for i, query in enumerate(test_queries):
        print(f"\n🔍 Test {i+1}: {query}")
        
        # Build URL
        params = {
            'appid': APP_ID,
            'input': query,
            'format': 'plaintext',
            'output': 'xml'
        }
        
        url = f"{BASE_URL}/query?" + urllib.parse.urlencode(params)
        
        try:
            # Make request
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse XML
                root = ET.fromstring(response.text)
                success = root.get('success', 'false') == 'true'
                
                if success:
                    print("   ✅ SUCCESS")
                    
                    # Extract results
                    pods = root.findall('pod')
                    print(f"   📊 Pods returned: {len(pods)}")
                    
                    # Show first result
                    for pod in pods[:2]:  # Show first 2 pods
                        title = pod.get('title', 'Unknown')
                        subpods = pod.findall('subpod')
                        if subpods:
                            plaintext = subpods[0].find('plaintext')
                            if plaintext is not None and plaintext.text:
                                result_text = plaintext.text[:100] + "..." if len(plaintext.text) > 100 else plaintext.text
                                print(f"   🎯 {title}: {result_text}")
                    
                    results.append({
                        'query': query,
                        'success': True,
                        'pods': len(pods)
                    })
                else:
                    print("   ❌ Query failed - Wolfram couldn't interpret")
                    results.append({
                        'query': query,
                        'success': False,
                        'error': 'Failed to interpret query'
                    })
                    
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                results.append({
                    'query': query,
                    'success': False,
                    'error': f'HTTP {response.status_code}'
                })
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
            results.append({
                'query': query,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    success_rate = (successful / total) * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 EXCELLENT! Your Wolfram Alpha integration is working perfectly!")
    elif success_rate >= 60:
        print("\n✅ GOOD! Your Wolfram Alpha integration is mostly working.")
    elif success_rate >= 40:
        print("\n⚠️  PARTIAL! Some issues with your Wolfram Alpha integration.")
    else:
        print("\n❌ ISSUES! Your Wolfram Alpha integration needs attention.")
    
    # Save results
    with open('wolfram_test_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'app_id': APP_ID,
            'results': results,
            'summary': {
                'total': total,
                'successful': successful,
                'success_rate': success_rate
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: wolfram_test_results.json")
    
    return results

def test_neural_specific():
    """Test brain simulation specific queries"""
    print("\n" + "=" * 60)
    print("🧠 TESTING BRAIN SIMULATION SPECIFIC QUERIES")
    print("=" * 60)
    
    neural_queries = [
        "action potential",
        "neural network",
        "synaptic transmission",
        "brain connectivity",
        "Hodgkin Huxley"
    ]
    
    for query in neural_queries:
        print(f"\n🔬 Neural Query: {query}")
        
        params = {
            'appid': APP_ID,
            'input': query,
            'format': 'plaintext',
            'output': 'xml'
        }
        
        url = f"{BASE_URL}/query?" + urllib.parse.urlencode(params)
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                success = root.get('success', 'false') == 'true'
                
                if success:
                    pods = root.findall('pod')
                    print(f"   ✅ Found {len(pods)} information pods")
                    
                    # Look for relevant pods
                    for pod in pods[:3]:
                        title = pod.get('title', 'Unknown')
                        if any(keyword in title.lower() for keyword in ['definition', 'basic', 'properties', 'description']):
                            subpods = pod.findall('subpod')
                            if subpods:
                                plaintext = subpods[0].find('plaintext')
                                if plaintext is not None and plaintext.text:
                                    print(f"   📝 {title}: {plaintext.text[:150]}...")
                else:
                    print("   ❌ No results found")
                    
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")

if __name__ == "__main__":
    # Run basic tests
    results = test_wolfram_basic()
    
    # Run neural-specific tests
    test_neural_specific()
    
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS:")
    print("=" * 60)
    print("1. If tests passed, your Wolfram Alpha integration is ready!")
    print("2. Use the full integration: python demo_wolfram_integration.py")
    print("3. Integrate with your brain simulation training pipeline")
    print("4. Explore advanced mathematical computations")
    print("5. Check out WOLFRAM_ALPHA_INTEGRATION_GUIDE.md for details")
