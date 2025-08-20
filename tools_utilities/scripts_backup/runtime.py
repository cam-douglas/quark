"""
Baby-AGI Runtime Infrastructure

This module provides the reactive graph runtime infrastructure for the Baby-AGI agent,
including node management, checkpointing, and interrupt handling.
"""

import json
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the reactive graph."""
    PLANNER = "planner"
    RETRIEVER = "retriever"
    ACTOR = "actor"
    CRITIC = "critic"
    MEMORY_MGR = "memory_mgr"
    SAFETY = "safety"
    SCHEDULER = "scheduler"


@dataclass
class NodeState:
    """State of a node in the reactive graph."""
    node_id: str
    node_type: NodeType
    status: str  # "idle", "running", "completed", "failed"
    last_run: Optional[float] = None
    execution_count: int = 0
    total_runtime: float = 0.0
    last_output: Optional[Dict[str, Any]] = None
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class Checkpoint:
    """Checkpoint data for the agent runtime."""
    timestamp: float
    node_states: Dict[str, NodeState]
    memory_state: Dict[str, Any]
    budgets: Dict[str, Any]
    task_state: Dict[str, Any]
    checkpoint_id: str


class AgentRuntime:
    """
    Reactive graph runtime for Baby-AGI agent.
    
    Manages:
    - Node execution and state
    - Checkpointing and recovery
    - Budget enforcement
    - Interrupt handling
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir or "~/.babyagi/checkpoints").expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Node management
        self.nodes: Dict[str, Callable] = {}
        self.node_states: Dict[str, NodeState] = {}
        self._initialize_nodes()
        
        # Runtime state
        self.memory_state: Dict[str, Any] = {}
        self.task_state: Dict[str, Any] = {}
        self.budgets = config.get("budgets", {})
        self.checkpoint_interval = config.get("checkpoint_interval", 10)
        self.last_checkpoint = 0
        
        # Interrupt handling
        self.interrupt_requested = False
        self.interrupt_reason = None
    
    def _initialize_nodes(self):
        """Initialize the node states based on configuration."""
        node_configs = self.config.get("nodes", {})
        
        for node_name, node_config in node_configs.items():
            node_type = NodeType(node_name)
            self.node_states[node_name] = NodeState(
                node_id=node_name,
                node_type=node_type,
                status="idle"
            )
    
    def register_node(self, node_name: str, node_func: Callable):
        """Register a node function."""
        self.nodes[node_name] = node_func
        if node_name not in self.node_states:
            self.node_states[node_name] = NodeState(
                node_id=node_name,
                node_type=NodeType(node_name) if hasattr(NodeType, node_name.upper()) else None,
                status="idle"
            )
    
    def execute_node(self, node_name: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not registered")
        
        node_state = self.node_states[node_name]
        node_state.status = "running"
        node_state.last_run = time.time()
        node_state.execution_count += 1
        
        start_time = time.time()
        inputs = inputs or {}
        
        try:
            # Execute the node
            output = self.nodes[node_name](inputs)
            
            # Update state
            node_state.status = "completed"
            node_state.last_output = output
            node_state.total_runtime += time.time() - start_time
            
            # Checkpoint if needed
            self._maybe_checkpoint()
            
            return output
            
        except Exception as e:
            # Handle errors
            node_state.status = "failed"
            node_state.error_count += 1
            node_state.last_error = str(e)
            node_state.total_runtime += time.time() - start_time
            
            # Checkpoint on error
            self._maybe_checkpoint()
            
            raise
    
    def execute_cycle(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a complete cycle through all nodes."""
        if self.interrupt_requested:
            raise InterruptedError(f"Execution interrupted: {self.interrupt_reason}")
        
        inputs = inputs or {}
        cycle_outputs = {}
        
        # Execute nodes in order (planner -> retriever -> actor -> critic -> memory -> safety)
        node_order = ["planner", "retriever", "actor", "critic", "memory_mgr", "safety"]
        
        for node_name in node_order:
            if node_name in self.nodes:
                try:
                    # Pass outputs from previous nodes as inputs
                    node_inputs = {**inputs, **cycle_outputs}
                    output = self.execute_node(node_name, node_inputs)
                    cycle_outputs[node_name] = output
                except Exception as e:
                    # Log error and continue with next node
                    print(f"Error in node {node_name}: {e}")
                    cycle_outputs[node_name] = {"error": str(e)}
        
        return cycle_outputs
    
    def _maybe_checkpoint(self):
        """Create a checkpoint if enough time has passed."""
        current_time = time.time()
        if current_time - self.last_checkpoint >= self.checkpoint_interval:
            self.create_checkpoint()
    
    def create_checkpoint(self):
        """Create a checkpoint of the current runtime state."""
        checkpoint = Checkpoint(
            timestamp=time.time(),
            node_states=self.node_states.copy(),
            memory_state=self.memory_state.copy(),
            budgets=self.budgets.copy(),
            task_state=self.task_state.copy(),
            checkpoint_id=f"checkpoint_{int(time.time())}"
        )
        
        # Save checkpoint to file
        checkpoint_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
        
        # Convert dataclasses to dicts for JSON serialization
        checkpoint_dict = asdict(checkpoint)
        checkpoint_dict["node_states"] = {
            k: asdict(v) for k, v in checkpoint_dict["node_states"].items()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_dict, f, indent=2)
        
        self.last_checkpoint = time.time()
        print(f"Checkpoint created: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load a checkpoint by ID."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore runtime state
            self.node_states = {
                k: NodeState(**v) for k, v in checkpoint_data["node_states"].items()
            }
            self.memory_state = checkpoint_data["memory_state"]
            self.budgets = checkpoint_data["budgets"]
            self.task_state = checkpoint_data["task_state"]
            
            print(f"Checkpoint loaded: {checkpoint_file}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoint IDs."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        return [f.stem for f in checkpoint_files]
    
    def interrupt(self, reason: str = "User requested"):
        """Request an interrupt of the current execution."""
        self.interrupt_requested = True
        self.interrupt_reason = reason
    
    def clear_interrupt(self):
        """Clear the interrupt flag."""
        self.interrupt_requested = False
        self.interrupt_reason = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current runtime status."""
        return {
            "node_states": {k: asdict(v) for k, v in self.node_states.items()},
            "memory_state": self.memory_state,
            "budgets": self.budgets,
            "task_state": self.task_state,
            "interrupt_requested": self.interrupt_requested,
            "interrupt_reason": self.interrupt_reason,
            "last_checkpoint": self.last_checkpoint
        }
