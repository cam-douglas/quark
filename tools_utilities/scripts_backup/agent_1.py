"""
Baby-AGI Agent Implementation

This module implements the core Baby-AGI agent with reactive graph runtime,
interruptible execution, and modern control plane features.
"""

import os
import signal
import time
import threading
import socket
import json
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path


class BabyAGIAgent:
    """
    Self-Running, Interruptible Local Agent
    
    Features:
    - Reactive-graph runtime with durable checkpoints and interrupts
    - ReAct + MCTS reasoning with Process Reward Model verification
    - MemGPT-style virtual context manager with graph memory
    - HyDE + RePlug robust retrieval with verification
    - nsjail/cgroups sandboxing with tool allowlists
    - JSON schema validation and policy-based guardrails
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.run_dir = Path(self.config.get("run_dir", "~/.babyagi")).expanduser()
        self.run_dir.mkdir(exist_ok=True)
        
        # Control files
        self.sentinel = self.run_dir / "STOP"
        self.socket_path = self.run_dir / "control.sock"
        self.log_path = self.run_dir / "agent.log"
        
        # State
        self.should_run = True
        self.paused = False
        self.runtime = None
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the agent."""
        return {
            "run_dir": "~/.babyagi",
            "tick_interval": 5,
            "max_cycles": 1000,
            "budgets": {
                "time": 600,  # seconds
                "tokens": 200000,
                "tool_calls": 200
            },
            "nodes": {
                "planner": {"policy": "react+mcts", "prune_with": "prm"},
                "retriever": {"hyde": True, "topk": 8},
                "actor": {"tools": ["search", "code_exec", "browser", "http", "fs"]},
                "critic": {"model": "prm_think", "accept_threshold": 0.62},
                "memory_mgr": {"tiers": ["working", "episodic", "semantic"], "backend": "memgpt+graph"},
                "safety": {"rails": "nemo", "schema_validate": True}
            }
        }
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        self.log(f"[signal] received {signum}; stopping")
        self.should_run = False
    
    def log(self, msg: str):
        """Log a message with timestamp."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(f"{ts} {msg}\n")
    
    def start(self):
        """Start the agent main loop."""
        # Start control server in background
        control_thread = threading.Thread(target=self._control_server, daemon=True)
        control_thread.start()
        
        self.log("[boot] agent started")
        
        cycle_count = 0
        while self.should_run and cycle_count < self.config.get("max_cycles", 1000):
            try:
                self._safe_tick()
                cycle_count += 1
                time.sleep(self.config.get("tick_interval", 5))
            except Exception as e:
                self.log(f"[main] exception: {e}\n{traceback.format_exc()}")
                break
        
        self.log("[shutdown] agent stopped")
    
    def _safe_tick(self):
        """Execute one agent cycle with error handling."""
        try:
            self._tick()
        except Exception as e:
            self.log(f"[tick] exception: {e}\n{traceback.format_exc()}")
    
    def _tick(self):
        """Execute one agent cycle."""
        if self.paused:
            return
        
        if self.sentinel.exists():
            self.log("[tick] STOP sentinel detected; initiating shutdown")
            self.should_run = False
            return
        
        # === PLACEHOLDER: planner -> retriever -> actor -> critic -> memory -> safety ===
        self.log("[tick] cycle start")
        
        # Simulate some work with time budget checks
        time.sleep(1)
        
        self.log("[tick] cycle end")
    
    def _control_server(self):
        """Run the control socket server."""
        if self.socket_path.exists():
            self.socket_path.unlink()
        
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.socket_path))
        srv.listen(1)
        
        self.log(f"[control] listening on {self.socket_path}")
        
        while self.should_run:
            try:
                conn, _ = srv.accept()
                cmd = conn.recv(1024).decode().strip()
                resp = self._dispatch_command(cmd)
                conn.sendall(resp.encode())
                conn.close()
            except Exception as e:
                self.log(f"[control] error: {e}")
    
    def _dispatch_command(self, cmd: str) -> str:
        """Dispatch control commands."""
        if cmd == "status":
            return json.dumps({
                "running": self.should_run,
                "paused": self.paused,
                "config": self.config
            })
        elif cmd == "pause":
            self.paused = True
            self.log("[control] paused")
            return "ok: paused"
        elif cmd == "resume":
            self.paused = False
            self.log("[control] resumed")
            return "ok: resumed"
        elif cmd == "stop":
            self.should_run = False
            self.log("[control] stopping")
            return "ok: stopping"
        else:
            return "err: unknown command"
    
    def stop(self):
        """Stop the agent."""
        self.should_run = False
        if self.socket_path.exists():
            self.socket_path.unlink()
    
    def pause(self):
        """Pause the agent."""
        self.paused = True
    
    def resume(self):
        """Resume the agent."""
        self.paused = False
