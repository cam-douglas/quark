#!/usr/bin/env python3
"""
🧠 Enhanced Brain Launcher v4 - Neural Dynamics Integration
Integrates actual neural dynamics with biological validation for Pillar 1

**Features:**
- Real neural dynamics with spiking neurons and plasticity
- Biological validation against neuroscience benchmarks
- Cortical-subcortical loop implementation
- Message-to-spike and spike-to-message conversion
- Comprehensive validation reporting

**Usage:**
  python brain_launcher_v4.py --connectome connectome_v3.yaml --steps 100 --stage F --validate
"""

import argparse
import random
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np

# Import neural integration components
try:
    from ................................................neural_integration_layer import NeuralIntegrationLayer, CorticalSubcorticalLoop
    from ................................................biological_validator import BiologicalValidator
    from ................................................neural_components import SpikingNeuron, HebbianSynapse, STDP
except ImportError:
    # Fallback for direct execution
    from neural_integration_layer import NeuralIntegrationLayer, CorticalSubcorticalLoop
    from biological_validator import BiologicalValidator
    from neural_components import SpikingNeuron, HebbianSynapse, STDP

# Import quantum integration
try:
    from ................................................quantum_integration import QuantumIntegration, QuantumConfig
    from ................................................advanced_quantum_integration import AdvancedQuantumIntegration, AdvancedQuantumConfig
except ImportError:
    # Fallback for direct execution
    from quantum_integration import QuantumIntegration, QuantumConfig
    from advanced_quantum_integration import AdvancedQuantumIntegration, AdvancedQuantumConfig

# Import existing brain launcher components
try:
    from ................................................brain_launcher_v3 import Brain, Module, Message, msg
except ImportError:
    # Fallback for direct execution
    from brain_launcher_v3 import Brain, Module, Message, msg

class NeuralEnhancedModule(Module):
    """Enhanced module with neural dynamics integration"""
    
    def __init__(self, name: str, spec: Dict[str, Any], neural_layer: NeuralIntegrationLayer):
        super().__init__(name, spec)
        self.neural_layer = neural_layer
        self.neural_metrics = {}

    def step(self, inbox: List[Message], ctx: Dict[str, Any]) -> Tuple[List[Message], Dict[str, Any]]:
        """Enhanced step with neural dynamics"""
        
        # Convert brain messages to neural inputs
        brain_messages = [{"kind": m.kind, "src": m.src, "dst": m.dst, 
                          "priority": m.priority, "payload": m.payload} for m in inbox]
        
        # Process through neural dynamics
        neural_outputs, neural_metrics = self.neural_layer.step(brain_messages)
        
        # Convert neural outputs back to brain messages
        outbox = []
        for neural_output in neural_outputs:
            message = Message(
                kind=neural_output["kind"],
                src=neural_output["src"],
                dst=neural_output["dst"],
                priority=neural_output["priority"],
                payload=neural_output["payload"]
            )
            outbox.append(message)
        
        # Get biological metrics for validation
        self.neural_metrics = self.neural_layer.get_biological_metrics()
        
        # Enhanced telemetry with neural data
        telemetry = {
            "confidence": min(1.0, 0.3 + 0.4 * self.neural_metrics["loop_stability"] + 0.3 * random.random()),
            "demand": 0.1 + 0.2 * self.neural_metrics["integration_metrics"]["loop_activity"],
            "neural_metrics": self.neural_metrics
        }
        
        return outbox, telemetry

class NeuralEnhancedPFC(NeuralEnhancedModule):
    """PFC with neural dynamics"""
    
    def step(self, inbox: List[Message], ctx: Dict[str, Any]) -> Tuple[List[Message], Dict[str, Any]]:
        # Get neural metrics for PFC-specific processing
        neural_metrics = self.neural_layer.get_biological_metrics()
        pfc_firing_rate = neural_metrics["firing_rates"]["pfc"]
        
        # PFC executive function based on neural activity
        if pfc_firing_rate > 20.0:  # High activity - executive control
            plan = {"seq": [("execute", 1), ("monitor", 1)], "goal": "high_control"}
        elif pfc_firing_rate > 5.0:  # Moderate activity - planning
            plan = {"seq": [("plan", 1), ("rehearse", 1)], "goal": "planning"}
        else:  # Low activity - maintenance
            plan = {"seq": [("maintain", 1)], "goal": "maintenance"}
        
        outbox = [msg("Command", self.name, "working_memory", action="process", plan=plan)]
        
        # Enhanced telemetry
        telemetry = {
            "confidence": min(1.0, 0.2 + 0.3 * pfc_firing_rate / 50.0 + 0.5 * random.random()),
            "demand": 0.1 + 0.3 * pfc_firing_rate / 50.0,
            "neural_metrics": neural_metrics,
            "executive_state": "high_control" if pfc_firing_rate > 20.0 else "planning" if pfc_firing_rate > 5.0 else "maintenance"
        }
        
        return outbox, telemetry

class NeuralEnhancedBasalGanglia(NeuralEnhancedModule):
    """Basal Ganglia with neural dynamics"""
    
    def step(self, inbox: List[Message], ctx: Dict[str, Any]) -> Tuple[List[Message], Dict[str, Any]]:
        neural_metrics = self.neural_layer.get_biological_metrics()
        bg_firing_rate = neural_metrics["firing_rates"]["bg"]
        
        # BG gating based on neural activity
        if bg_firing_rate > 50.0:  # High activity - strong gating
            gate_strength = 0.9
            action = "strong_gate"
        elif bg_firing_rate > 20.0:  # Moderate activity - moderate gating
            gate_strength = 0.6
            action = "moderate_gate"
        else:  # Low activity - weak gating
            gate_strength = 0.2
            action = "weak_gate"
        
        outbox = [msg("Modulation", self.name, "thalamus", gate_strength=gate_strength, action=action)]
        
        telemetry = {
            "confidence": min(1.0, 0.3 + 0.4 * gate_strength + 0.3 * random.random()),
            "demand": 0.1 + 0.2 * bg_firing_rate / 100.0,
            "neural_metrics": neural_metrics,
            "gating_state": action
        }
        
        return outbox, telemetry

class NeuralEnhancedThalamus(NeuralEnhancedModule):
    """Thalamus with neural dynamics"""
    
    def step(self, inbox: List[Message], ctx: Dict[str, Any]) -> Tuple[List[Message], Dict[str, Any]]:
        neural_metrics = self.neural_layer.get_biological_metrics()
        thalamus_firing_rate = neural_metrics["firing_rates"]["thalamus"]
        
        # Thalamic relay based on neural activity
        if thalamus_firing_rate > 100.0:  # High activity - burst mode
            relay_mode = "burst"
            relay_efficiency = 0.8
        elif thalamus_firing_rate > 20.0:  # Moderate activity - tonic mode
            relay_mode = "tonic"
            relay_efficiency = 0.6
        else:  # Low activity - sleep mode
            relay_mode = "sleep"
            relay_efficiency = 0.2
        
        outbox = [msg("Relay", self.name, "pfc", mode=relay_mode, efficiency=relay_efficiency)]
        
        telemetry = {
            "confidence": min(1.0, 0.2 + 0.4 * relay_efficiency + 0.4 * random.random()),
            "demand": 0.1 + 0.3 * thalamus_firing_rate / 200.0,
            "neural_metrics": neural_metrics,
            "relay_mode": relay_mode
        }
        
        return outbox, telemetry

class NeuralEnhancedBrain(Brain):
    """Enhanced brain with neural dynamics and biological validation"""
    
    def __init__(self, connectome_path: str, stage: str = "F", validate: bool = False):
        # Load connectome config
        with open(connectome_path, "r", encoding="utf-8") as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Initialize curriculum if present
        cur_cfg = config.get("curriculum", {})
        schedule = cur_cfg.get("schedule", [])
        tpw = int(cur_cfg.get("ticks_per_week", 50))
        from ................................................brain_launcher_v3 import Curriculum
        curriculum = Curriculum(schedule, tpw) if schedule else None
        
        super().__init__(config, stage, curriculum=curriculum)
        
        # Initialize neural integration layer
        self.neural_layer = NeuralIntegrationLayer(stage)
        
        # Initialize biological validator
        self.validator = BiologicalValidator()
        self.validate = validate
        
        # Initialize quantum integration
        quantum_config = QuantumConfig(enable_quantum=True, code_distance=5)
        self.quantum_integration = QuantumIntegration(quantum_config)
        
        # Initialize advanced quantum integration
        advanced_quantum_config = AdvancedQuantumConfig(
            enable_quantum=True,
            enable_cross_talk=True,
            enable_leakage=True,
            enable_measurement_errors=True,
            enable_multi_distance=True,
            enable_hybrid_processing=True,
            enable_consciousness_quantization=True
        )
        self.advanced_quantum_integration = AdvancedQuantumIntegration(advanced_quantum_config)
        
        # Replace modules with neural-enhanced versions
        self._create_neural_enhanced_modules()
        
        # Validation tracking
        self.validation_results = []
        self.biological_metrics_history = []
        
        # Quantum metrics tracking
        self.quantum_metrics_history = []
        self.advanced_quantum_metrics_history = []
        
    def _create_neural_enhanced_modules(self):
        """Create neural-enhanced versions of brain modules"""
        
        # Replace existing modules with neural-enhanced versions
        if "pfc" in self.modules:
            self.modules["pfc"] = NeuralEnhancedPFC("pfc", self.modules["pfc"].spec, self.neural_layer)
        
        if "basal_ganglia" in self.modules:
            self.modules["basal_ganglia"] = NeuralEnhancedBasalGanglia("basal_ganglia", 
                                                                     self.modules["basal_ganglia"].spec, 
                                                                     self.neural_layer)
        
        if "thalamus" in self.modules:
            self.modules["thalamus"] = NeuralEnhancedThalamus("thalamus", 
                                                             self.modules["thalamus"].spec, 
                                                             self.neural_layer)
    
    async def step(self, ticks_per_week: int = 50) -> Dict[str, Any]:
        """Enhanced step with neural dynamics and validation"""
        
        # Run standard brain step
        result = super().step(ticks_per_week)
        
        # Get biological metrics
        biological_metrics = self.neural_layer.get_biological_metrics()
        self.biological_metrics_history.append(biological_metrics)
        
        # Perform biological validation if enabled
        if self.validate:
            validation_result = self.validator.validate_neural_dynamics(biological_metrics)
            self.validation_results.append(validation_result)
            
            # Add validation info to result
            result["biological_validation"] = validation_result
            result["biological_realism"] = validation_result["biological_realism"]
        
        # Add neural metrics to result
        result["neural_metrics"] = biological_metrics
        result["cortical_loop_stability"] = biological_metrics["loop_stability"]
        
        # Process quantum data if available
        if hasattr(self, 'quantum_integration') and self.quantum_integration.quantum_available:
            quantum_data = {
                'quantum_state': {
                    'state_id': f'step_{len(self.biological_metrics_history)}',
                    'logical_state': '0',
                    'neural_activity': biological_metrics
                },
                'error_syndromes': [],
                'logical_qubits': ['consciousness_qubit']
            }
            
            processed_quantum_data = self.quantum_integration.process_quantum_data(quantum_data)
            result["quantum_processed"] = processed_quantum_data.get('quantum_processed', False)
            result["quantum_timestamp"] = processed_quantum_data.get('quantum_timestamp', 'N/A')
            
            # Track quantum metrics
            quantum_status = self.quantum_integration.get_status()
            self.quantum_metrics_history.append(quantum_status)
        
        # Process advanced quantum data if available
        if hasattr(self, 'advanced_quantum_integration') and self.advanced_quantum_integration.quantum_available:
            # Create consciousness data for advanced processing
            consciousness_data = {
                'executive_control': biological_metrics.get('firing_rates', {}).get('pfc', 0.5) / 50.0,
                'working_memory': biological_metrics.get('loop_stability', 0.5),
                'attention': biological_metrics.get('feedback_strength', 0.5),
                'self_awareness': 0.7,  # Default value
                'emotional_state': 0.5,  # Default value
                'cognitive_load': 1.0 - biological_metrics.get('loop_stability', 0.5),
                'creativity': 0.6,  # Default value
                'decision_making': biological_metrics.get('feedback_strength', 0.5)
            }
            
            # Process through advanced quantum integration
            processed_advanced_quantum = await self.advanced_quantum_integration.process_advanced_quantum(consciousness_data)
            result["advanced_quantum_processed"] = processed_advanced_quantum.get('advanced_quantum_processed', False)
            result["consciousness_quantized"] = 'quantized_consciousness' in processed_advanced_quantum
            result["quantum_entanglement"] = processed_advanced_quantum.get('quantized_consciousness', {}).get('quantum_entanglement', {})
            
            # Track advanced quantum metrics
            advanced_quantum_status = self.advanced_quantum_integration.get_advanced_status()
            self.advanced_quantum_metrics_history.append(advanced_quantum_status)
        
        return result
    
    def get_validation_report(self) -> str:
        """Get comprehensive validation report"""
        if not self.validate or not self.validation_results:
            return "Biological validation not enabled or no results available."
        
        return self.validator.get_validation_report()
    
    def export_validation_data(self, filename: str):
        """Export validation data"""
        if self.validate:
            self.validator.export_validation_data(filename)
    
    def get_neural_summary(self) -> Dict[str, Any]:
        """Get summary of neural dynamics"""
        if not self.biological_metrics_history:
            return {"error": "No neural metrics available"}
        
        latest_metrics = self.biological_metrics_history[-1]
        
        return {
            "firing_rates": latest_metrics["firing_rates"],
            "synchrony": latest_metrics["synchrony"],
            "oscillation_power": latest_metrics["oscillation_power"],
            "loop_stability": latest_metrics["loop_stability"],
            "feedback_strength": latest_metrics["feedback_strength"],
            "integration_metrics": latest_metrics["integration_metrics"],
            "biological_realism": self.validation_results[-1]["biological_realism"] if self.validation_results else False
        }
    
    def get_quantum_summary(self) -> Dict[str, Any]:
        """Get summary of quantum integration"""
        if not hasattr(self, 'quantum_integration'):
            return {"error": "Quantum integration not available"}
        
        if not self.quantum_metrics_history:
            return {"error": "No quantum metrics available"}
        
        latest_quantum_metrics = self.quantum_metrics_history[-1]
        
        return {
            "quantum_available": latest_quantum_metrics.get('quantum_available', False),
            "surface_code": latest_quantum_metrics.get('surface_code', False),
            "quantum_decoder": latest_quantum_metrics.get('quantum_decoder', False),
            "code_distance": latest_quantum_metrics.get('code_distance', 0),
            "quantum_states_processed": len(self.quantum_metrics_history),
            "quantum_integration_status": "Active" if latest_quantum_metrics.get('quantum_available') else "Inactive"
        }
    
    def get_advanced_quantum_summary(self) -> Dict[str, Any]:
        """Get summary of advanced quantum integration"""
        if not hasattr(self, 'advanced_quantum_integration'):
            return {"error": "Advanced quantum integration not available"}
        
        if not self.advanced_quantum_metrics_history:
            return {"error": "No advanced quantum metrics available"}
        
        latest_advanced_metrics = self.advanced_quantum_metrics_history[-1]
        
        return {
            "advanced_quantum_available": latest_advanced_metrics.get('quantum_available', False),
            "advanced_components": latest_advanced_metrics.get('advanced_components', {}),
            "enhancements_enabled": latest_advanced_metrics.get('enhancements_enabled', []),
            "consciousness_quantization": latest_advanced_metrics.get('configuration', {}).get('consciousness_quantization', False),
            "quantum_classical_hybrid": latest_advanced_metrics.get('configuration', {}).get('hybrid_processing', False),
            "multi_distance_training": latest_advanced_metrics.get('configuration', {}).get('multi_distance', False),
            "advanced_quantum_states_processed": len(self.advanced_quantum_metrics_history),
            "advanced_quantum_integration_status": "Active" if latest_advanced_metrics.get('quantum_available') else "Inactive"
        }

async def main():
    parser = argparse.ArgumentParser(description="Neural Dynamics Enhanced Brain Launcher")
    parser.add_argument("--connectome", required=True, help="Connectome YAML file")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--stage", choices=["F", "N0", "N1"], default="F", help="Developmental stage")
    parser.add_argument("--validate", action="store_true", help="Enable biological validation")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--log_csv", help="CSV log file")
    parser.add_argument("--validation_log", help="Validation log file")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Create enhanced brain
    brain = NeuralEnhancedBrain(args.connectome, args.stage, args.validate)
    
    print(f"🧠 Neural Dynamics Enhanced Brain Launcher v4")
    print(f"Stage: {args.stage}")
    print(f"Validation: {'Enabled' if args.validate else 'Disabled'}")
    print(f"Steps: {args.steps}")
    
    # Display quantum integration status
    quantum_status = brain.quantum_integration.get_status()
    print(f"🔬 Quantum Integration: {'Active' if quantum_status['quantum_available'] else 'Inactive'}")
    if quantum_status['quantum_available']:
        print(f"   Surface Code: {'✅' if quantum_status['surface_code'] else '❌'}")
        print(f"   Quantum Decoder: {'✅' if quantum_status['quantum_decoder'] else '❌'}")
        print(f"   Code Distance: {quantum_status['code_distance']}")
    
    # Display advanced quantum integration status
    advanced_quantum_status = brain.advanced_quantum_integration.get_advanced_status()
    print(f"🚀 Advanced Quantum Integration: {'Active' if advanced_quantum_status['quantum_available'] else 'Inactive'}")
    if advanced_quantum_status['quantum_available']:
        print(f"   Consciousness Quantization: {'✅' if advanced_quantum_status['configuration']['consciousness_quantization'] else '❌'}")
        print(f"   Quantum-Classical Hybrid: {'✅' if advanced_quantum_status['configuration']['hybrid_processing'] else '❌'}")
        print(f"   Multi-Distance Training: {'✅' if advanced_quantum_status['configuration']['multi_distance'] else '❌'}")
        print(f"   Advanced Error Models: {'✅' if advanced_quantum_status['configuration']['cross_talk'] else '❌'}")
    
    print("=" * 60)
    
    # Run simulation
    start_time = time.time()
    
    for step in range(args.steps):
        result = await brain.step()
        
        # Print progress every 10 steps
        if step % 10 == 0:
            neural_summary = brain.get_neural_summary()
            print(f"Step {step:3d}: PFC={neural_summary['firing_rates']['pfc']:.1f}Hz, "
                  f"BG={neural_summary['firing_rates']['bg']:.1f}Hz, "
                  f"Thal={neural_summary['firing_rates']['thalamus']:.1f}Hz, "
                  f"Stability={neural_summary['loop_stability']:.3f}")
            
            # Show quantum progress
            if result.get('quantum_processed'):
                print(f"       🔬 Quantum: Active, Advanced: {'✅' if result.get('advanced_quantum_processed') else '❌'}")
    
    end_time = time.time()
    
    print("\n" + "=" * 60)
    print("🎯 SIMULATION COMPLETE")
    print(f"Runtime: {end_time - start_time:.2f} seconds")
    
    # Final neural summary
    final_summary = brain.get_neural_summary()
    print(f"\n📊 FINAL NEURAL SUMMARY:")
    print(f"PFC Firing Rate: {final_summary['firing_rates']['pfc']:.2f} Hz")
    print(f"BG Firing Rate: {final_summary['firing_rates']['bg']:.2f} Hz")
    print(f"Thalamus Firing Rate: {final_summary['firing_rates']['thalamus']:.2f} Hz")
    print(f"Loop Stability: {final_summary['loop_stability']:.3f}")
    print(f"Feedback Strength: {final_summary['feedback_strength']:.3f}")
    
    if args.validate:
        print(f"Biological Realism: {'✅ ACHIEVED' if final_summary['biological_realism'] else '❌ FAILED'}")
        
        # Print validation report
        print("\n" + brain.get_validation_report())
        
        # Export validation data
        if args.validation_log:
            brain.export_validation_data(args.validation_log)
            print(f"Validation data exported to: {args.validation_log}")
    
    # Final quantum summary
    quantum_summary = brain.get_quantum_summary()
    print(f"\n🔬 FINAL QUANTUM SUMMARY:")
    print(f"Quantum Integration: {quantum_summary.get('quantum_integration_status', 'Unknown')}")
    print(f"Surface Code: {'✅ Active' if quantum_summary.get('surface_code') else '❌ Inactive'}")
    print(f"Quantum Decoder: {'✅ Active' if quantum_summary.get('quantum_decoder') else '❌ Inactive'}")
    print(f"Code Distance: {quantum_summary.get('code_distance', 0)}")
    print(f"Quantum States Processed: {quantum_summary.get('quantum_states_processed', 0)}")
    
    # Final advanced quantum summary
    advanced_quantum_summary = brain.get_advanced_quantum_summary()
    print(f"\n🚀 FINAL ADVANCED QUANTUM SUMMARY:")
    print(f"Advanced Quantum Integration: {advanced_quantum_summary.get('advanced_quantum_integration_status', 'Unknown')}")
    print(f"Consciousness Quantization: {'✅ Active' if advanced_quantum_summary.get('consciousness_quantization') else '❌ Inactive'}")
    print(f"Quantum-Classical Hybrid: {'✅ Active' if advanced_quantum_summary.get('quantum_classical_hybrid') else '❌ Inactive'}")
    print(f"Multi-Distance Training: {'✅ Active' if advanced_quantum_summary.get('multi_distance_training') else '❌ Inactive'}")
    print(f"Advanced Quantum States Processed: {advanced_quantum_summary.get('advanced_quantum_states_processed', 0)}")
    
    # Export neural metrics to CSV if requested
    if args.log_csv:
        import csv
        with open(args.log_csv, 'w', newline='') as csvfile:
            fieldnames = ['step', 'pfc_rate', 'bg_rate', 'thalamus_rate', 'loop_stability', 'feedback_strength']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, metrics in enumerate(brain.biological_metrics_history):
                writer.writerow({
                    'step': i,
                    'pfc_rate': metrics['firing_rates']['pfc'],
                    'bg_rate': metrics['firing_rates']['bg'],
                    'thalamus_rate': metrics['firing_rates']['thalamus'],
                    'loop_stability': metrics['loop_stability'],
                    'feedback_strength': metrics['feedback_strength']
                })
        print(f"Neural metrics exported to: {args.log_csv}")

if __name__ == "__main__":
    asyncio.run(main())
