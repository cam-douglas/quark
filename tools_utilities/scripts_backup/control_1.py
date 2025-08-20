"""
Baby-AGI Control Interface

This module provides a command-line interface for controlling the Baby-AGI agent
through Unix domain sockets.
"""

import os
import socket
import sys
import json
from pathlib import Path
from typing import Optional


class AgentController:
    """Control interface for Baby-AGI agent."""
    
    def __init__(self, socket_path: Optional[str] = None):
        if socket_path is None:
            self.socket_path = Path("~/.babyagi/control.sock").expanduser()
        else:
            self.socket_path = Path(socket_path)
    
    def send_command(self, cmd: str) -> str:
        """Send a command to the agent and return the response."""
        if not self.socket_path.exists():
            return "error: agent not running (socket not found)"
        
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(self.socket_path))
            sock.send(cmd.encode())
            response = sock.recv(4096).decode()
            sock.close()
            return response
        except Exception as e:
            return f"error: {e}"
    
    def status(self) -> str:
        """Get agent status."""
        return self.send_command("status")
    
    def pause(self) -> str:
        """Pause the agent."""
        return self.send_command("pause")
    
    def resume(self) -> str:
        """Resume the agent."""
        return self.send_command("resume")
    
    def stop(self) -> str:
        """Stop the agent."""
        return self.send_command("stop")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Baby-AGI Control Interface")
        print("Usage: python -m src.baby_agi.control [status|pause|resume|stop]")
        print("\nCommands:")
        print("  status  - Get agent status and configuration")
        print("  pause   - Pause agent execution")
        print("  resume  - Resume agent execution")
        print("  stop    - Stop agent execution")
        sys.exit(1)
    
    cmd = sys.argv[1]
    controller = AgentController()
    
    if cmd == "status":
        response = controller.status()
        try:
            # Pretty print JSON response
            data = json.loads(response)
            print("Agent Status:")
            print(f"  Running: {data.get('running', 'unknown')}")
            print(f"  Paused: {data.get('paused', 'unknown')}")
            if 'config' in data:
                print(f"  Run Directory: {data['config'].get('run_dir', 'unknown')}")
                print(f"  Tick Interval: {data['config'].get('tick_interval', 'unknown')}s")
        except json.JSONDecodeError:
            print(response)
    
    elif cmd == "pause":
        print(controller.pause())
    
    elif cmd == "resume":
        print(controller.resume())
    
    elif cmd == "stop":
        print(controller.stop())
    
    else:
        print(f"Unknown command: {cmd}")
        print("Available commands: status, pause, resume, stop")
        sys.exit(1)


if __name__ == "__main__":
    main()
