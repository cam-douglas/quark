#!/usr/bin/env python3
"""
Comprehensive Live Demo - Multiple experiments running simultaneously
Shows all Phase-1 prototypes in action with real-time streaming.
"""

import time
import threading
import random
from testing.visualizations.simple_live_server import start_live_server, live_series
from brain_architecture.neural_core.proto_cortex.layer_sheet import LayerSheet
from brain_architecture.neural_core.hippocampus.stdp_synapse import STDPSynapse
from brain_architecture.neural_core.thalamus.relay_nucleus import ThalamicRelay
from brain_architecture.neural_core.basal_ganglia.simple_gate import ActionGate

def run_layer_sheet_experiment():
    """LayerSheet homeostatic convergence"""
    sheet = LayerSheet(n=50)
    for step in range(200):
        sheet.step()
        live_series("layer_sheet_activity", sheet.mean_activity(), step)
        time.sleep(0.05)  # Slow down for visibility

def run_stdp_experiment():
    """STDP synaptic weight evolution"""
    syn = STDPSynapse(weight=0.5)
    for step in range(100):
        # Alternate LTP/LTD
        if step % 2 == 0:
            syn.pre_spike(step * 10)
            syn.post_spike(step * 10 + 5)
        else:
            syn.post_spike(step * 10)
            syn.pre_spike(step * 10 + 5)
        
        live_series("stdp_weight", syn.weight(), step)
        time.sleep(0.1)

def run_relay_experiment():
    """Thalamic relay gating behavior"""
    relay = ThalamicRelay(gate_open=True)
    for step in range(150):
        # Close gate at step 50, open at step 100
        if step == 50:
            relay.close_gate()
        elif step == 100:
            relay.open_gate()
        
        packet = random.random()
        out = relay.receive(packet)
        throughput = 1 if out is not None else 0
        
        live_series("relay_throughput", throughput, step)
        time.sleep(0.08)

def run_action_gate_experiment():
    """Basal ganglia action selection"""
    gate = ActionGate(threshold=0.6)
    for step in range(120):
        # Generate varying salience levels
        salience = {
            "action_a": random.random() * 0.8,
            "action_b": random.random() * 0.9,
            "action_c": random.random() * 0.7
        }
        
        result = gate.select(salience)
        selected = 1 if result.selected_channel else 0
        
        live_series("action_selection", selected, step)
        live_series("salience_a", salience["action_a"], step)
        live_series("salience_b", salience["action_b"], step)
        live_series("salience_c", salience["action_c"], step)
        time.sleep(0.06)

def main():
    print("🧠 Quark Comprehensive Live Demo")
    print("=" * 50)
    
    # Start live server and open dashboard
    start_live_server()
    
    print("🚀 Starting all experiments simultaneously...")
    print("📊 Watch the conscious dashboard for real-time updates!")
    
    # Run all experiments in parallel threads
    threads = [
        threading.Thread(target=run_layer_sheet_experiment, name="LayerSheet"),
        threading.Thread(target=run_stdp_experiment, name="STDP"),
        threading.Thread(target=run_relay_experiment, name="Relay"),
        threading.Thread(target=run_action_gate_experiment, name="ActionGate")
    ]
    
    # Start all threads
    for thread in threads:
        thread.start()
        print(f"✅ Started {thread.name} experiment")
    
    # Wait for all to complete
    for thread in threads:
        thread.join()
    
    print("🎉 All experiments completed!")
    print("📈 Check the dashboard for final results")

if __name__ == "__main__":
    main()
