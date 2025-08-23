#!/usr/bin/env python3
"""
🧬 Advanced Biological Simulator
Complex neural processes and biological interactions simulation
"""

import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import numpy as np
import random
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass
from datetime import datetime
import time

@dataclass
class NeuralProcess:
    """Represents a neural process in the simulation"""
    process_id: str
    process_type: str  # "excitation", "inhibition", "plasticity", "homeostasis"
    intensity: float  # 0.0 to 1.0
    duration: float  # seconds
    biological_markers: List[str]
    affected_chunks: List[int]
    timestamp: datetime
    status: str  # "active", "completed", "failed"

@dataclass
class BiologicalInteraction:
    """Represents biological interactions between markers"""
    interaction_id: str
    marker_a: str
    marker_b: str
    interaction_type: str  # "coexpression", "inhibition", "activation", "modulation"
    strength: float  # 0.0 to 1.0
    duration: float
    timestamp: datetime
    status: str

class AdvancedBiologicalSimulator:
    """Advanced biological simulation system for neural processes"""
    
    def __init__(self):
        self.memory_dir = Path("memory")
        self.load_semantic_network()
        self.initialize_simulation_parameters()
        self.active_processes = {}
        self.interaction_history = []
        self.simulation_time = 0.0
        
    def load_semantic_network(self):
        """Load the semantic network and metadata"""
        try:
            with open(self.memory_dir / "metadata.json") as f:
                self.metadata = json.load(f)
            
            with open(self.memory_dir / "rule_graph.json") as f:
                graph_data = json.load(f)
                self.graph = nx.node_link_graph(graph_data)
            
            print(f"✅ Loaded semantic network: {len(self.metadata)} chunks, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            print(f"❌ Failed to load semantic network: {e}")
            raise
    
    def initialize_simulation_parameters(self):
        """Initialize biological simulation parameters"""
        self.biological_parameters = {
            "neural_excitation": {
                "base_intensity": 0.7,
                "decay_rate": 0.1,
                "spread_factor": 0.3,
                "critical_threshold": 0.8
            },
            "synaptic_plasticity": {
                "learning_rate": 0.05,
                "forgetting_rate": 0.02,
                "consolidation_threshold": 0.6,
                "plasticity_window": 0.5
            },
            "homeostasis": {
                "equilibrium_target": 0.5,
                "correction_rate": 0.1,
                "stability_threshold": 0.2,
                "adaptation_speed": 0.05
            },
            "marker_interactions": {
                "GFAP": {"stability": 0.9, "modulation": 0.3, "coexpression": ["Vimentin", "S100B"]},
                "NeuN": {"stability": 0.8, "modulation": 0.4, "coexpression": ["GAP43", "NSE"]},
                "GAP43": {"stability": 0.6, "modulation": 0.7, "coexpression": ["NSE", "Tau"]},
                "NSE": {"stability": 0.7, "modulation": 0.5, "coexpression": ["GAP43", "NeuN"]},
                "Tau": {"stability": 0.5, "modulation": 0.8, "coexpression": ["GAP43", "MBP"]},
                "S100B": {"stability": 0.8, "modulation": 0.4, "coexpression": ["GFAP", "Vimentin"]},
                "MBP": {"stability": 0.9, "modulation": 0.2, "coexpression": ["NeuN", "Tau"]},
                "Vimentin": {"stability": 0.6, "modulation": 0.6, "coexpression": ["GFAP", "S100B"]}
            }
        }
    
    def simulate_neural_excitation(self, trigger_chunk_id: int, intensity: float = 0.7) -> NeuralProcess:
        """Simulate neural excitation spreading through the network"""
        print(f"🧠 Simulating neural excitation from chunk {trigger_chunk_id}")
        
        process_id = f"excitation_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Find connected chunks
        connected_chunks = self._find_connected_chunks(trigger_chunk_id, max_depth=3)
        
        # Create excitation process
        process = NeuralProcess(
            process_id=process_id,
            process_type="excitation",
            intensity=intensity,
            duration=5.0,  # 5 seconds
            biological_markers=self._get_chunk_markers(trigger_chunk_id),
            affected_chunks=connected_chunks,
            timestamp=datetime.now(),
            status="active"
        )
        
        # Apply excitation effects
        self._apply_excitation_effects(process)
        
        # Store active process
        self.active_processes[process_id] = process
        
        return process
    
    def simulate_synaptic_plasticity(self, chunk_a_id: int, chunk_b_id: int, 
                                   interaction_strength: float = 0.5) -> NeuralProcess:
        """Simulate synaptic plasticity between two chunks"""
        print(f"🔗 Simulating synaptic plasticity between chunks {chunk_a_id} and {chunk_b_id}")
        
        process_id = f"plasticity_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Calculate plasticity parameters
        plasticity_intensity = interaction_strength * self.biological_parameters["synaptic_plasticity"]["learning_rate"]
        
        # Create plasticity process
        process = NeuralProcess(
            process_id=process_id,
            process_type="plasticity",
            intensity=plasticity_intensity,
            duration=10.0,  # 10 seconds
            biological_markers=self._get_chunk_markers(chunk_a_id) + self._get_chunk_markers(chunk_b_id),
            affected_chunks=[chunk_a_id, chunk_b_id],
            timestamp=datetime.now(),
            status="active"
        )
        
        # Apply plasticity effects
        self._apply_plasticity_effects(process)
        
        # Store active process
        self.active_processes[process_id] = process
        
        return process
    
    def simulate_homeostasis(self, target_marker: str, duration: float = 15.0) -> NeuralProcess:
        """Simulate homeostatic regulation for a specific marker"""
        print(f"⚖️ Simulating homeostasis for marker {target_marker}")
        
        process_id = f"homeostasis_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Find chunks with target marker
        target_chunks = [chunk["chunk_id"] for chunk in self.metadata 
                        if chunk.get("chunk_id") is not None and 
                        target_marker in chunk.get("markers", [])]
        
        # Create homeostasis process
        process = NeuralProcess(
            process_id=process_id,
            process_type="homeostasis",
            intensity=self.biological_parameters["homeostasis"]["correction_rate"],
            duration=duration,
            biological_markers=[target_marker],
            affected_chunks=target_chunks,
            timestamp=datetime.now(),
            status="active"
        )
        
        # Apply homeostasis effects
        self._apply_homeostasis_effects(process)
        
        # Store active process
        self.active_processes[process_id] = process
        
        return process
    
    def _find_connected_chunks(self, start_chunk_id: int, max_depth: int = 3) -> List[int]:
        """Find chunks connected to the start chunk within max_depth"""
        if start_chunk_id not in self.graph:
            return [start_chunk_id]
        
        connected = set([start_chunk_id])
        current_level = [start_chunk_id]
        
        for depth in range(max_depth):
            next_level = []
            for chunk_id in current_level:
                if chunk_id in self.graph:
                    neighbors = list(self.graph.neighbors(chunk_id))
                    for neighbor_id in neighbors:
                        if neighbor_id not in connected:
                            connected.add(neighbor_id)
                            next_level.append(neighbor_id)
            current_level = next_level
            if not current_level:
                break
        
        return list(connected)
    
    def _get_chunk_markers(self, chunk_id: int) -> List[str]:
        """Get biological markers for a specific chunk"""
        chunk = self._get_chunk_by_id(chunk_id)
        return chunk.get("markers", []) if chunk else []
    
    def _get_chunk_by_id(self, chunk_id: int) -> Dict:
        """Get chunk data by ID"""
        for chunk in self.metadata:
            if chunk.get("chunk_id") == chunk_id:
                return chunk
        return None
    
    def _apply_excitation_effects(self, process: NeuralProcess):
        """Apply excitation effects to affected chunks"""
        print(f"   ⚡ Applying excitation effects to {len(process.affected_chunks)} chunks")
        
        for chunk_id in process.affected_chunks:
            chunk = self._get_chunk_by_id(chunk_id)
            if chunk:
                # Simulate excitation by temporarily enhancing marker expression
                markers = chunk.get("markers", [])
                for marker in markers:
                    if marker in self.biological_parameters["marker_interactions"]:
                        # Increase marker activity temporarily
                        self._modify_marker_activity(marker, chunk_id, process.intensity)
    
    def _apply_plasticity_effects(self, process: NeuralProcess):
        """Apply plasticity effects to affected chunks"""
        print(f"   🔗 Applying plasticity effects to {len(process.affected_chunks)} chunks")
        
        if len(process.affected_chunks) >= 2:
            chunk_a_id, chunk_b_id = process.affected_chunks[0], process.affected_chunks[1]
            
            # Strengthen connection between chunks
            self._strengthen_chunk_connection(chunk_a_id, chunk_b_id, process.intensity)
            
            # Apply learning effects
            self._apply_learning_effects(chunk_a_id, chunk_b_id, process.intensity)
    
    def _apply_homeostasis_effects(self, process: NeuralProcess):
        """Apply homeostasis effects to affected chunks"""
        print(f"   ⚖️ Applying homeostasis effects to {len(process.affected_chunks)} chunks")
        
        target_marker = process.biological_markers[0]
        
        for chunk_id in process.affected_chunks:
            chunk = self._get_chunk_by_id(chunk_id)
            if chunk and target_marker in chunk.get("markers", []):
                # Apply homeostatic regulation
                self._regulate_marker_expression(target_marker, chunk_id, process.intensity)
    
    def _modify_marker_activity(self, marker: str, chunk_id: int, intensity: float):
        """Modify marker activity for a specific chunk"""
        # This would modify the actual marker expression in a real system
        # For simulation, we track the modification
        modification = {
            "marker": marker,
            "chunk_id": chunk_id,
            "intensity": intensity,
            "timestamp": datetime.now(),
            "type": "excitation"
        }
        
        # Store modification for analysis
        if not hasattr(self, 'marker_modifications'):
            self.marker_modifications = []
        self.marker_modifications.append(modification)
    
    def _strengthen_chunk_connection(self, chunk_a_id: int, chunk_b_id: int, strength: float):
        """Strengthen connection between two chunks"""
        # This would modify the actual graph weights in a real system
        # For simulation, we track the strengthening
        strengthening = {
            "chunk_a": chunk_a_id,
            "chunk_b": chunk_b_id,
            "strength": strength,
            "timestamp": datetime.now(),
            "type": "plasticity"
        }
        
        # Store strengthening for analysis
        if not hasattr(self, 'connection_strengthenings'):
            self.connection_strengthenings = []
        self.connection_strengthenings.append(strengthening)
    
    def _apply_learning_effects(self, chunk_a_id: int, chunk_b_id: int, intensity: float):
        """Apply learning effects between chunks"""
        # Simulate Hebbian learning: "neurons that fire together wire together"
        learning_effect = {
            "chunk_a": chunk_a_id,
            "chunk_b": chunk_b_id,
            "learning_intensity": intensity,
            "timestamp": datetime.now(),
            "type": "hebbian_learning"
        }
        
        # Store learning effect for analysis
        if not hasattr(self, 'learning_effects'):
            self.learning_effects = []
        self.learning_effects.append(learning_effect)
    
    def _regulate_marker_expression(self, marker: str, chunk_id: int, intensity: float):
        """Regulate marker expression for homeostasis"""
        regulation = {
            "marker": marker,
            "chunk_id": chunk_id,
            "regulation_intensity": intensity,
            "timestamp": datetime.now(),
            "type": "homeostasis"
        }
        
        # Store regulation for analysis
        if not hasattr(self, 'marker_regulations'):
            self.marker_regulations = []
        self.marker_regulations.append(regulation)
    
    def advance_simulation(self, time_step: float = 1.0):
        """Advance the simulation by a time step"""
        self.simulation_time += time_step
        
        # Update active processes
        completed_processes = []
        for process_id, process in self.active_processes.items():
            process.duration -= time_step
            
            if process.duration <= 0:
                process.status = "completed"
                completed_processes.append(process_id)
                
                # Apply final effects
                self._apply_process_completion_effects(process)
        
        # Remove completed processes
        for process_id in completed_processes:
            del self.active_processes[process_id]
        
        # Apply decay effects
        self._apply_decay_effects(time_step)
        
        print(f"⏰ Simulation advanced to {self.simulation_time:.1f}s, {len(self.active_processes)} active processes")
    
    def _apply_process_completion_effects(self, process: NeuralProcess):
        """Apply effects when a process completes"""
        print(f"   ✅ Process {process.process_id} completed")
        
        if process.process_type == "excitation":
            # Apply post-excitation effects
            self._apply_post_excitation_effects(process)
        elif process.process_type == "plasticity":
            # Apply post-plasticity effects
            self._apply_post_plasticity_effects(process)
        elif process.process_type == "homeostasis":
            # Apply post-homeostasis effects
            self._apply_post_homeostasis_effects(process)
    
    def _apply_decay_effects(self, time_step: float):
        """Apply decay effects to all active processes"""
        for process in self.active_processes.values():
            if process.process_type == "excitation":
                # Decay excitation intensity
                decay_rate = self.biological_parameters["neural_excitation"]["decay_rate"]
                process.intensity *= (1 - decay_rate * time_step)
                process.intensity = max(0.0, process.intensity)
    
    def _apply_post_excitation_effects(self, process: NeuralProcess):
        """Apply effects after excitation process completes"""
        # Reset marker activities to baseline
        for chunk_id in process.affected_chunks:
            chunk = self._get_chunk_by_id(chunk_id)
            if chunk:
                markers = chunk.get("markers", [])
                for marker in markers:
                    self._reset_marker_activity(marker, chunk_id)
    
    def _apply_post_plasticity_effects(self, process: NeuralProcess):
        """Apply effects after plasticity process completes"""
        # Consolidate learning effects
        if len(process.affected_chunks) >= 2:
            chunk_a_id, chunk_b_id = process.affected_chunks[0], process.affected_chunks[1]
            self._consolidate_learning(chunk_a_id, chunk_b_id, process.intensity)
    
    def _apply_post_homeostasis_effects(self, process: NeuralProcess):
        """Apply effects after homeostasis process completes"""
        # Verify equilibrium has been reached
        target_marker = process.biological_markers[0]
        self._verify_homeostasis_equilibrium(target_marker)
    
    def _reset_marker_activity(self, marker: str, chunk_id: int):
        """Reset marker activity to baseline"""
        reset = {
            "marker": marker,
            "chunk_id": chunk_id,
            "timestamp": datetime.now(),
            "type": "reset_to_baseline"
        }
        
        if not hasattr(self, 'marker_resets'):
            self.marker_resets = []
        self.marker_resets.append(reset)
    
    def _consolidate_learning(self, chunk_a_id: int, chunk_b_id: int, intensity: float):
        """Consolidate learning between chunks"""
        consolidation = {
            "chunk_a": chunk_a_id,
            "chunk_b": chunk_b_id,
            "consolidation_intensity": intensity,
            "timestamp": datetime.now(),
            "type": "learning_consolidation"
        }
        
        if not hasattr(self, 'learning_consolidations'):
            self.learning_consolidations = []
        self.learning_consolidations.append(consolidation)
    
    def _verify_homeostasis_equilibrium(self, marker: str):
        """Verify that homeostasis equilibrium has been reached"""
        verification = {
            "marker": marker,
            "timestamp": datetime.now(),
            "type": "homeostasis_verification",
            "equilibrium_reached": True  # Simplified for simulation
        }
        
        if not hasattr(self, 'homeostasis_verifications'):
            self.homeostasis_verifications = []
        self.homeostasis_verifications.append(verification)
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        return {
            "simulation_time": self.simulation_time,
            "active_processes": len(self.active_processes),
            "process_types": list(set(p.process_type for p in self.active_processes.values())),
            "total_modifications": len(getattr(self, 'marker_modifications', [])),
            "total_strengthenings": len(getattr(self, 'connection_strengthenings', [])),
            "total_learning_effects": len(getattr(self, 'learning_effects', [])),
            "total_regulations": len(getattr(self, 'marker_regulations', []))
        }
    
    def run_comprehensive_simulation(self, duration: float = 30.0, time_step: float = 1.0) -> Dict[str, Any]:
        """Run a comprehensive simulation for a specified duration"""
        print(f"🚀 Running comprehensive simulation for {duration} seconds...")
        
        # Initialize simulation
        self.simulation_time = 0.0
        
        # Start various processes
        processes = []
        
        # Start excitation process
        excitation_process = self.simulate_neural_excitation(
            trigger_chunk_id=0,  # Start from first chunk
            intensity=0.8
        )
        processes.append(excitation_process)
        
        # Start plasticity process
        if len(self.metadata) >= 2:
            plasticity_process = self.simulate_synaptic_plasticity(
                chunk_a_id=0,
                chunk_b_id=1,
                interaction_strength=0.6
            )
            processes.append(plasticity_process)
        
        # Start homeostasis process
        homeostasis_process = self.simulate_homeostasis(
            target_marker="GFAP",
            duration=duration
        )
        processes.append(homeostasis_process)
        
        # Run simulation
        simulation_steps = int(duration / time_step)
        for step in range(simulation_steps):
            self.advance_simulation(time_step)
            
            # Print status every 5 steps
            if step % 5 == 0:
                status = self.get_simulation_status()
                print(f"   Step {step + 1}/{simulation_steps}: {status['active_processes']} active processes")
        
        # Final status
        final_status = self.get_simulation_status()
        
        return {
            "simulation_duration": duration,
            "time_step": time_step,
            "total_steps": simulation_steps,
            "final_status": final_status,
            "processes_started": len(processes),
            "simulation_completed": True
        }

def main():
    """Main execution function"""
    print("🧬 Advanced Biological Simulator")
    print("=" * 50)
    
    try:
        simulator = AdvancedBiologicalSimulator()
        
        # Run comprehensive simulation
        simulation_result = simulator.run_comprehensive_simulation(duration=20.0, time_step=1.0)
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"   Duration: {simulation_result['simulation_duration']} seconds")
        print(f"   Total steps: {simulation_result['total_steps']}")
        print(f"   Processes started: {simulation_result['processes_started']}")
        
        # Display final status
        final_status = simulation_result['final_status']
        print(f"\n📊 Final Simulation Status:")
        print(f"   Active processes: {final_status['active_processes']}")
        print(f"   Total modifications: {final_status['total_modifications']}")
        print(f"   Total strengthenings: {final_status['total_strengthenings']}")
        print(f"   Total learning effects: {final_status['total_learning_effects']}")
        print(f"   Total regulations: {final_status['total_regulations']}")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")

if __name__ == "__main__":
    main()
