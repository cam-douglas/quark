#!/usr/bin/env python3
"""
Cursor Network Testing Script
Purpose: Test and verify Cursor network connectivity and configuration
Author: Quark Development Team
Dependencies: requests, json, subprocess, asyncio
"""

import asyncio
import json
import os
import requests
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import threading

class CursorNetworkTester:
    """
    Comprehensive testing suite for Cursor network functionality.
    Tests streaming, polling, DNS resolution, and connection stability.
    """
    
    def __init__(self):
        self.cursor_endpoints = [
            "https://api2.cursor.sh",
            "https://api3.cursor.sh", 
            "https://cursor.sh"
        ]
        self.test_results = {
            "timestamp": time.time(),
            "environment": self._get_environment_info(),
            "tests": {}
        }
    
    def _get_environment_info(self) -> Dict[str, any]:
        """Gather environment information."""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "is_docker": Path("/.dockerenv").exists(),
            "environment_vars": {
                key: os.environ.get(key) 
                for key in [
                    "CURSOR_DISABLE_STREAMING",
                    "CURSOR_DOCKER_MODE",
                    "CURSOR_USE_POLLING",
                    "HTTP_PROXY",
                    "HTTPS_PROXY"
                ]
                if os.environ.get(key)
            }
        }
    
    def test_basic_connectivity(self) -> Dict[str, any]:
        """Test basic HTTP connectivity to Cursor endpoints."""
        print("🔌 Testing basic connectivity...")
        results = {}
        
        for endpoint in self.cursor_endpoints:
            try:
                start_time = time.time()
                response = requests.get(
                    endpoint,
                    timeout=10,
                    allow_redirects=True
                )
                end_time = time.time()
                
                results[endpoint] = {
                    "success": True,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "headers": dict(response.headers),
                    "error": None
                }
                print(f"   ✅ {endpoint}: {response.status_code} ({results[endpoint]['response_time']:.2f}s)")
                
            except Exception as e:
                results[endpoint] = {
                    "success": False,
                    "status_code": None,
                    "response_time": None,
                    "headers": None,
                    "error": str(e)
                }
                print(f"   ❌ {endpoint}: {str(e)}")
        
        return results
    
    def test_dns_resolution(self) -> Dict[str, any]:
        """Test DNS resolution for Cursor domains."""
        print("🌐 Testing DNS resolution...")
        results = {}
        
        domains = ["api2.cursor.sh", "api3.cursor.sh", "cursor.sh"]
        
        for domain in domains:
            try:
                start_time = time.time()
                ip_addresses = socket.getaddrinfo(domain, None)
                end_time = time.time()
                
                ips = list(set([addr[4][0] for addr in ip_addresses]))
                
                results[domain] = {
                    "success": True,
                    "ip_addresses": ips,
                    "resolution_time": end_time - start_time,
                    "error": None
                }
                print(f"   ✅ {domain}: {ips[0]} (+{len(ips)-1} more) ({results[domain]['resolution_time']:.3f}s)")
                
            except Exception as e:
                results[domain] = {
                    "success": False,
                    "ip_addresses": None,
                    "resolution_time": None,
                    "error": str(e)
                }
                print(f"   ❌ {domain}: {str(e)}")
        
        return results
    
    def test_streaming_capability(self) -> Dict[str, any]:
        """Test streaming capability and buffering issues."""
        print("📡 Testing streaming capability...")
        results = {}
        
        # Test with a known streaming endpoint
        test_urls = [
            "https://httpbin.org/stream/5",  # Simple streaming test
            "https://api2.cursor.sh/health",  # Cursor health endpoint
        ]
        
        for url in test_urls:
            try:
                start_time = time.time()
                
                # Test streaming response
                response = requests.get(
                    url,
                    stream=True,
                    timeout=15
                )
                
                chunks_received = 0
                first_chunk_time = None
                last_chunk_time = time.time()
                
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        chunks_received += 1
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                        last_chunk_time = time.time()
                        
                        # Only process first few chunks for test
                        if chunks_received >= 3:
                            break
                
                end_time = time.time()
                
                time_to_first_chunk = first_chunk_time - start_time if first_chunk_time else None
                
                results[url] = {
                    "success": True,
                    "chunks_received": chunks_received,
                    "time_to_first_chunk": time_to_first_chunk,
                    "total_time": end_time - start_time,
                    "streaming_works": time_to_first_chunk is not None and time_to_first_chunk < 5.0,
                    "error": None
                }
                
                status = "✅" if results[url]["streaming_works"] else "⚠️"
                print(f"   {status} {url}: {chunks_received} chunks, first in {time_to_first_chunk:.2f}s")
                
            except Exception as e:
                results[url] = {
                    "success": False,
                    "chunks_received": 0,
                    "time_to_first_chunk": None,
                    "total_time": None,
                    "streaming_works": False,
                    "error": str(e)
                }
                print(f"   ❌ {url}: {str(e)}")
        
        return results
    
    def test_websocket_connectivity(self) -> Dict[str, any]:
        """Test WebSocket connectivity (simulated)."""
        print("🔌 Testing WebSocket-like connectivity...")
        results = {}
        
        # Test persistent connection capability
        for endpoint in ["api2.cursor.sh", "api3.cursor.sh"]:
            try:
                # Test multiple rapid requests (simulates WebSocket-like behavior)
                start_time = time.time()
                
                session = requests.Session()
                request_times = []
                
                for i in range(5):
                    req_start = time.time()
                    response = session.get(f"https://{endpoint}/health", timeout=5)
                    req_end = time.time()
                    request_times.append(req_end - req_start)
                
                end_time = time.time()
                avg_request_time = sum(request_times) / len(request_times)
                
                results[endpoint] = {
                    "success": True,
                    "average_request_time": avg_request_time,
                    "total_time": end_time - start_time,
                    "requests_completed": len(request_times),
                    "connection_stable": avg_request_time < 2.0,
                    "error": None
                }
                
                status = "✅" if results[endpoint]["connection_stable"] else "⚠️"
                print(f"   {status} {endpoint}: {len(request_times)} requests, avg {avg_request_time:.2f}s")
                
            except Exception as e:
                results[endpoint] = {
                    "success": False,
                    "average_request_time": None,
                    "total_time": None,
                    "requests_completed": 0,
                    "connection_stable": False,
                    "error": str(e)
                }
                print(f"   ❌ {endpoint}: {str(e)}")
        
        return results
    
    def test_polling_simulation(self) -> Dict[str, any]:
        """Simulate polling behavior (fallback for streaming)."""
        print("🔄 Testing polling simulation...")
        results = {}
        
        endpoint = "https://httpbin.org/get"
        
        try:
            # Simulate polling with intervals
            poll_times = []
            start_time = time.time()
            
            for i in range(3):
                poll_start = time.time()
                response = requests.get(endpoint, timeout=5)
                poll_end = time.time()
                
                poll_times.append(poll_end - poll_start)
                
                if i < 2:  # Don't sleep after last request
                    time.sleep(1)  # 1 second polling interval
            
            end_time = time.time()
            avg_poll_time = sum(poll_times) / len(poll_times)
            
            results["polling_test"] = {
                "success": True,
                "polls_completed": len(poll_times),
                "average_poll_time": avg_poll_time,
                "total_time": end_time - start_time,
                "polling_viable": avg_poll_time < 3.0,
                "error": None
            }
            
            status = "✅" if results["polling_test"]["polling_viable"] else "⚠️"
            print(f"   {status} Polling: {len(poll_times)} polls, avg {avg_poll_time:.2f}s")
            
        except Exception as e:
            results["polling_test"] = {
                "success": False,
                "polls_completed": 0,
                "average_poll_time": None,
                "total_time": None,
                "polling_viable": False,
                "error": str(e)
            }
            print(f"   ❌ Polling test: {str(e)}")
        
        return results
    
    def check_cursor_configuration(self) -> Dict[str, any]:
        """Check current Cursor configuration."""
        print("⚙️  Checking Cursor configuration...")
        results = {}
        
        config_paths = [
            Path.home() / ".cursor" / "settings.json",
            Path.home() / ".config" / "cursor" / "settings.json",
            Path("/opt/cursor/settings.json"),
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Check for Docker-specific settings
                    docker_settings = {
                        "cursor.network.dockerMode": config.get("cursor.network.dockerMode"),
                        "cursor.network.disableStreaming": config.get("cursor.network.disableStreaming"),
                        "cursor.chat.usePolling": config.get("cursor.chat.usePolling"),
                        "cursor.agent.usePolling": config.get("cursor.agent.usePolling"),
                        "cursor.network.streamingFallback": config.get("cursor.network.streamingFallback"),
                        "http.proxySupport": config.get("http.proxySupport")
                    }
                    
                    results[str(config_path)] = {
                        "exists": True,
                        "readable": True,
                        "docker_settings": docker_settings,
                        "has_docker_fix": any(docker_settings.values()),
                        "error": None
                    }
                    
                    status = "✅" if results[str(config_path)]["has_docker_fix"] else "⚠️"
                    print(f"   {status} {config_path}: Docker settings {'found' if results[str(config_path)]['has_docker_fix'] else 'missing'}")
                    break
                    
                except Exception as e:
                    results[str(config_path)] = {
                        "exists": True,
                        "readable": False,
                        "docker_settings": None,
                        "has_docker_fix": False,
                        "error": str(e)
                    }
                    print(f"   ❌ {config_path}: Read error - {str(e)}")
        
        if not results:
            print("   ⚠️  No Cursor configuration found")
            results["no_config"] = {
                "exists": False,
                "readable": False,
                "docker_settings": None,
                "has_docker_fix": False,
                "error": "No configuration file found"
            }
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run all tests and compile results."""
        print("🧪 CURSOR NETWORK COMPREHENSIVE TEST")
        print("=" * 50)
        
        # Run all tests
        self.test_results["tests"]["basic_connectivity"] = self.test_basic_connectivity()
        print()
        
        self.test_results["tests"]["dns_resolution"] = self.test_dns_resolution()
        print()
        
        self.test_results["tests"]["streaming_capability"] = self.test_streaming_capability()
        print()
        
        self.test_results["tests"]["websocket_connectivity"] = self.test_websocket_connectivity()
        print()
        
        self.test_results["tests"]["polling_simulation"] = self.test_polling_simulation()
        print()
        
        self.test_results["tests"]["cursor_configuration"] = self.check_cursor_configuration()
        print()
        
        # Generate summary
        self._generate_summary()
        
        return self.test_results
    
    def _generate_summary(self) -> None:
        """Generate test summary and recommendations."""
        print("📊 TEST SUMMARY & RECOMMENDATIONS")
        print("=" * 50)
        
        issues_found = []
        recommendations = []
        
        # Analyze results
        basic_conn = self.test_results["tests"]["basic_connectivity"]
        streaming = self.test_results["tests"]["streaming_capability"]
        config = self.test_results["tests"]["cursor_configuration"]
        
        # Check basic connectivity
        failed_endpoints = [ep for ep, result in basic_conn.items() if not result["success"]]
        if failed_endpoints:
            issues_found.append(f"Failed to connect to: {', '.join(failed_endpoints)}")
            recommendations.append("Check firewall/proxy settings")
        
        # Check streaming
        streaming_issues = []
        for url, result in streaming.items():
            if not result.get("streaming_works", False):
                streaming_issues.append(url)
        
        if streaming_issues:
            issues_found.append("Streaming not working properly")
            recommendations.append("Enable polling mode in Cursor settings")
        
        # Check configuration
        has_docker_config = any(
            result.get("has_docker_fix", False) 
            for result in config.values()
        )
        
        if not has_docker_config and self.test_results["environment"]["is_docker"]:
            issues_found.append("Running in Docker without proper configuration")
            recommendations.append("Run cursor_network_fix.py to apply Docker settings")
        
        # Output summary
        if issues_found:
            print("❌ Issues found:")
            for issue in issues_found:
                print(f"   • {issue}")
            print()
            
            print("💡 Recommendations:")
            for rec in recommendations:
                print(f"   • {rec}")
        else:
            print("✅ No issues detected - Cursor should work properly!")
        
        print(f"\n📋 Test completed at {time.ctime(self.test_results['timestamp'])}")
    
    def save_results(self, filepath: Optional[Path] = None) -> Path:
        """Save test results to file."""
        if filepath is None:
            timestamp = int(time.time())
            filepath = Path(f"cursor_network_test_results_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {filepath}")
        return filepath

def main():
    """Main execution function."""
    tester = CursorNetworkTester()
    results = tester.run_comprehensive_test()
    
    # Save results
    results_file = tester.save_results()
    
    # Provide next steps
    print("\n🎯 NEXT STEPS:")
    print("   1. If issues found, run: python cursor_network_fix.py")
    print("   2. Restart Cursor completely after applying fixes")
    print("   3. Test Chat and Agent functionality")
    print(f"   4. Review detailed results in: {results_file}")

if __name__ == "__main__":
    main()


