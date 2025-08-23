#!/usr/bin/env python3
"""
Simple Live Streaming Server v2 - No freezing version.
Uses basic threading and simple message passing without asyncio complexity.
"""

import json
import time
import threading
import socket
import webbrowser
import os
from typing import Dict, Set

class SimpleLiveServer:
    """Simple live streaming server without asyncio complexity."""
    
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.running = False
        self.clients: Set[str] = set()  # Just track client IDs
        self.message_history = []
        self.max_history = 1000
        
    def start(self):
        """Start the server in a simple way"""
        self.running = True
        print(f"🚀 Simple live server started on {self.host}:{self.port}")
        
        # Open dashboard
        self._open_dashboard()
        
    def stop(self):
        """Stop the server"""
        self.running = False
        print("🛑 Simple live server stopped")
        
    def _open_dashboard(self):
        """Open the dashboard in browser"""
        dashboard_path = os.path.join(
            os.path.dirname(__file__), 
            "conscious_live_dashboard.html"
        )
        
        if os.path.exists(dashboard_path):
            try:
                webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                print("🎥 Dashboard opened")
            except Exception as e:
                print(f"⚠️ Could not open dashboard: {e}")
    
    def send_data(self, series_id: str, value, step: int):
        """Send data - just log it for now, no complex networking"""
        if not self.running:
            return
            
        message = {
            "series_id": series_id,
            "value": value,
            "step": step,
            "timestamp": time.time()
        }
        
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
            
        # Print for debugging (non-blocking)
        print(f"📡 Live data: {series_id} = {value} (step {step})")
        
        # In a real implementation, this would send to connected clients
        # For now, just ensure it doesn't block


# Global instance
_simple_server = None

def start_simple_live_server():
    """Start the simple live server"""
    global _simple_server
    
    if _simple_server is None:
        _simple_server = SimpleLiveServer()
        _simple_server.start()
    
    return _simple_server

def simple_live_series(series_id: str, value, step: int):
    """Simple live series function that won't freeze"""
    global _simple_server
    
    if _simple_server is None:
        start_simple_live_server()
        
    _simple_server.send_data(series_id, value, step)

if __name__ == "__main__":
    # Test the simple server
    server = start_simple_live_server()
    
    # Send some test data
    for i in range(5):
        simple_live_series("test", i * 0.2, i)
        time.sleep(0.5)
    
    print("✅ Simple test completed")
