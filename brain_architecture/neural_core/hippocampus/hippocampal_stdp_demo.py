from brain_architecture.neural_core.neural_dynamics.synaptic_plasticity import PlasticityManager
import time

"""
Hippocampal STDP Demo:
- Simulates a minimal CA3->CA1 pathway with STDP updates
- Prints synaptic weight changes over a few spike events
"""

def run_demo():
    print("\n🧠 Hippocampal STDP Demo (CA3 -> CA1)")
    plasticity = PlasticityManager()

    # Define a simple mapping of neuron IDs: CA3 neurons (0..4) to CA1 neurons (10..14)
    pairs = [(i, i+10) for i in range(5)]

    # Add synapses for CA3 -> CA1
    for pre, post in pairs:
        plasticity.add_synapse(pre, post, initial_weight=1.0)

    t0 = time.time()

    # Simulate spike timing for LTP/LTD patterns
    # Pair 0: Pre then Post (LTP)
    plasticity.handle_pre_spike(0, 10, t0)
    time.sleep(0.005)
    plasticity.handle_post_spike(0, 10, t0 + 0.005)

    # Pair 1: Post then Pre (LTD)
    plasticity.handle_post_spike(1, 11, t0)
    time.sleep(0.005)
    plasticity.handle_pre_spike(1, 11, t0 + 0.005)

    # Pair 2: Two LTP events
    plasticity.handle_pre_spike(2, 12, t0)
    plasticity.handle_post_spike(2, 12, t0 + 0.003)
    plasticity.handle_pre_spike(2, 12, t0 + 0.010)
    plasticity.handle_post_spike(2, 12, t0 + 0.013)

    # Show final weights
    stats = plasticity.get_plasticity_stats()
    print("\n📊 Final STDP Weights (subset):")
    for pre, post in pairs:
        w = stats['stdp_weights'].get((pre, post), None)
        if w is not None:
            print(f"  CA3 {pre} -> CA1 {post}: {w:.3f}")

    print("\n✅ Hippocampal STDP demo completed.")

if __name__ == "__main__":
    run_demo()
