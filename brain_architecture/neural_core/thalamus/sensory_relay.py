from brian2 import *
try:
    # Support module execution
    from .thalamus_model import create_thalamic_nucleus  # type: ignore
except Exception:
    # Fallback for direct script execution
    from thalamus_model import create_thalamic_nucleus
import sys, os

# Ensure project root is on path for direct execution
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools_utilities.automation.experiment_logger import ExperimentLogger
import matplotlib.pyplot as plt
import os

# --- 1. Simulation Setup ---
# Set the duration of the simulation
duration = 200*ms

# --- 2. Create Network Components ---
# Create the thalamic nucleus using our model function
thalamic_nucleus = create_thalamic_nucleus(n_neurons=100)

# Make thalamic neurons easier to spike for live demo
thalamic_nucleus.thresh = -55*mV
thalamic_nucleus.tau = 30*ms

# Create a sensory input layer (e.g., from the retina)
# We use a PoissonGroup, which generates random spikes at a specified rate.
# This is a common way to model external input in computational neuroscience.
n_inputs = 100
input_firing_rate = 120*Hz  # boosted rate for visible relay
sensory_input = PoissonGroup(n_inputs, rates=input_firing_rate)

# --- 3. Connect the Network ---
# Create synapses to connect the sensory input to the thalamic nucleus.
# We'll use a one-to-one connection for simplicity in this initial model.
synapses = Synapses(sensory_input, thalamic_nucleus, on_pre='v_post += 15*mV')  # stronger jump for demo
synapses.connect(j='i') # Connects neuron i in sensory_input to neuron i in thalamic_nucleus

# --- 4. Set up Monitors ---
# Record spikes from both the input and the thalamic nucleus to visualize the relay.
input_monitor = SpikeMonitor(sensory_input)
thalamus_monitor = SpikeMonitor(thalamic_nucleus)

# --- 5. Run the Simulation ---
print("Starting sensory relay simulation...")
run(duration)
print("Simulation finished.")

# --- 6. Plot Results ---
# A raster plot is an excellent way to visualize the spike times of multiple neurons.
fig = plt.figure(figsize=(10, 6))

# Plot spikes from the sensory input layer
plt.subplot(2, 1, 1)
plt.plot(input_monitor.t/ms, input_monitor.i, '.k', markersize=5)
plt.xlabel('Time (ms)')
plt.ylabel('Sensory Neuron Index')
plt.title('Sensory Input Activity')
plt.xlim(0, duration/ms)

# Plot spikes from the thalamic nucleus
plt.subplot(2, 1, 2)
plt.plot(thalamus_monitor.t/ms, thalamus_monitor.i, '.r', markersize=5)
plt.xlabel('Time (ms)')
plt.ylabel('Thalamic Neuron Index')
plt.title('Thalamic Nucleus Activity (Relayed Spikes)')
plt.xlim(0, duration/ms)

plt.tight_layout()

# --- 7. Log with MLflow ---
logger = ExperimentLogger()
params = {
    "n_inputs": n_inputs,
    "input_firing_rate_hz": float(input_firing_rate/Hz),
    "n_thalamic_neurons": len(thalamic_nucleus),
    "thalamic_thresh_mV": float(thalamic_nucleus.thresh[0]/mV),
    "thalamic_tau_ms": float(thalamic_nucleus.tau[0]/ms),
    "synaptic_jump_mV": 15.0,
}
metrics = {
    "sensory_spikes": int(len(input_monitor.t)),
    "thalamic_spikes": int(len(thalamus_monitor.t))
}

with logger.start_run(run_name="thalamus_sensory_relay", experiment_name="QuarkBrainSimulation"):
    logger.log_params(params)
    logger.log_metrics(metrics)

    # Save raster plot
    out_path = "thalamus_relay_raster.png"
    fig.savefig(out_path)
    logger.log_artifact(out_path)
    if os.path.exists(out_path):
        os.remove(out_path)

plt.show()
