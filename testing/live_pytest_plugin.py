#!/usr/bin/env python3
"""
Live Pytest Plugin - Automatically streams live 3D visualizations for every test!
"""

import pytest
import time
import json
import asyncio
import threading
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Global server instance
LIVE_SERVER = None
LIVE_STREAMING_AVAILABLE = True

def start_live_server():
    """Start the working live 3D server."""
    global LIVE_SERVER
    
    try:
        from testing.visualizations.working_live_3d_final import Working3DServer
        
        # Start server on a unique port
        port = 8007  # Different from the demo server
        LIVE_SERVER = Working3DServer(port=port)
        LIVE_SERVER.start()
        
        print(f"🚀 Live 3D server started on port {port} for pytest")
        return LIVE_SERVER
        
    except Exception as e:
        print(f"⚠️ Could not start live 3D server: {e}")
        return None

def live_series(series_id, value, step):
    """Stream data to the live server."""
    global LIVE_SERVER
    
    if not LIVE_SERVER or not hasattr(LIVE_SERVER, 'clients'):
        return
    
    try:
        # Create message
        message = {
            "series_id": series_id,
            "value": value,
            "step": step,
            "timestamp": time.time()
        }
        
        # Broadcast to all clients
        async def broadcast():
            if LIVE_SERVER.clients:
                await asyncio.gather(
                    *[client.send(json.dumps(message)) for client in LIVE_SERVER.clients],
                    return_exceptions=True
                )
        
        # Schedule broadcast using the server's event loop
        if hasattr(LIVE_SERVER, 'loop') and LIVE_SERVER.loop and LIVE_SERVER.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(broadcast(), LIVE_SERVER.loop)
            try:
                future.result(timeout=2)  # Wait up to 2 seconds
            except Exception as e:
                print(f"⚠️ Broadcast failed: {e}")
                
    except Exception as e:
        print(f"⚠️ Live streaming error: {e}")

def create_3d_test_visualization(test_name, duration, status):
    """Create a 3D visualization for test results."""
    global LIVE_SERVER
    
    if not LIVE_SERVER:
        return
    
    try:
        from testing.visualizations.working_live_3d_final import Working3DServer
        
        # Create test data
        test_data = [{
            "name": test_name,
            "duration": duration,
            "status": status
        }]
        
        # Stream 3D visualization
        fig = LIVE_SERVER.stream_3d_visualization("test_landscape", test_data)
        if fig:
            print(f"🎨 3D visualization created for test: {test_name}")
            
    except Exception as e:
        print(f"⚠️ Could not create 3D visualization: {e}")

# Pytest hooks
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure pytest with live streaming."""
    global LIVE_STREAMING_AVAILABLE
    
    if LIVE_STREAMING_AVAILABLE:
        print("🚀 Initializing live 3D streaming for pytest...")
        server = start_live_server()
        if server:
            print("✅ Live 3D streaming enabled for all tests!")
        else:
            print("❌ Live 3D streaming failed to start")
            LIVE_STREAMING_AVAILABLE = False

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Setup before each test - clear previous test data."""
    if not LIVE_STREAMING_AVAILABLE:
        return
        
    test_name = item.nodeid
    
    # Clear previous test results
    live_series("pytest_clear_previous", "clear", 0)
    
    # Stream test start
    live_series("pytest_test_start", {
        "test": test_name,
        "status": "starting",
        "timestamp": time.time()
    }, 0)
    
    print(f"🎯 Live streaming started for test: {test_name}")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """Stream test outcome and detailed metrics."""
    if not LIVE_STREAMING_AVAILABLE:
        return
        
    test_name = report.nodeid
    
    if report.when == "call":  # Test execution phase
        # Stream test outcome - ONLY show current test result, clear others
        live_series("pytest_clear_other_outcomes", "clear", 0)
        
        if report.passed:
            live_series("pytest_current_test_result", {
                "test": test_name,
                "outcome": "PASSED",
                "status": "completed"
            }, 0)
        elif report.failed:
            live_series("pytest_current_test_result", {
                "test": test_name,
                "outcome": "FAILED",
                "status": "completed"
            }, 0)
            if report.longrepr:
                live_series("pytest_current_test_error", str(report.longrepr)[:200], 0)
        elif report.skipped:
            live_series("pytest_current_test_result", {
                "test": test_name,
                "outcome": "SKIPPED",
                "status": "completed"
            }, 0)
    
    # Stream timing information for current test only
    if hasattr(report, 'duration'):
        live_series("pytest_current_test_duration", report.duration, 0)
    
    # Create 3D visualization when test completes
    if report.when == "call":
        try:
            # Determine test status
            status = "PASSED"
            if report.failed:
                status = "FAILED"
            elif report.skipped:
                status = "SKIPPED"
            
            # Create 3D visualization
            duration = report.duration if hasattr(report, 'duration') else 0.001
            create_3d_test_visualization(test_name, duration, status)
            
        except Exception as e:
            print(f"⚠️ Could not create 3D visualization: {e}")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Cleanup after each test."""
    if not LIVE_STREAMING_AVAILABLE:
        return
        
    test_name = item.nodeid
    
    # Stream test completion
    live_series("pytest_test_complete", {
        "test": test_name,
        "status": "completed",
        "timestamp": time.time()
    }, 0)
    
    print(f"✅ Live streaming completed for test: {test_name}")

@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Cleanup after all tests."""
    global LIVE_SERVER
    
    if LIVE_SERVER:
        print("🛑 Stopping live 3D server...")
        LIVE_SERVER.stop()
        print("✅ Live 3D server stopped")

# Additional hooks for comprehensive coverage
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Modify test collection for live streaming."""
    if LIVE_STREAMING_AVAILABLE:
        print(f"🎯 Live 3D streaming will cover {len(items)} tests")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Protocol for each test run."""
    if LIVE_STREAMING_AVAILABLE:
        test_name = item.nodeid
        print(f"🔄 Live streaming protocol for: {test_name}")

# Performance monitoring
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Make report for test run."""
    if LIVE_STREAMING_AVAILABLE and call.when == "call":
        # Stream performance metrics
        if hasattr(call, 'duration'):
            live_series("pytest_performance", {
                "test": item.nodeid,
                "duration": call.duration,
                "timestamp": time.time()
            }, 0)
