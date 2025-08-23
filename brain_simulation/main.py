from brian2 import *

# --- 1. Simulation Setup ---
# Set the duration of the simulation
duration = 100*ms

# --- 2. Neuron Model ---
# Define the neuron model using differential equations.
# This is a simple Leaky Integrate-and-Fire (LIF) model.
# dv/dt: Change in membrane potential
# tau: Membrane time constant
# El: Resting potential
# eqs describe the model's dynamics.
tau = 10*ms
El = -65*mV
thresh = -50*mV
reset = -70*mV
eqs = '''
dv/dt = (El - v) / tau : volt (unless refractory)
'''

# --- 3. Neuron Group ---
# Create a group of neurons. Here, we create a single neuron.
# 'threshold=' defines the condition for firing a spike.
# 'reset=' defines the membrane potential after a spike.
# 'refractory=' defines the period after a spike where the neuron cannot fire again.
# 'method=' specifies the numerical integration method.
G = NeuronGroup(1, eqs,
                threshold='v > thresh',
                reset='v = reset',
                refractory=5*ms,
                method='exact')

# Set the initial membrane potential
G.v = El

# --- 4. Input Stimulus ---
# Provide a constant input current to the neuron to make it spike.
# We will create a "PoissonGroup" which generates spikes with a certain frequency.
# And then connect it to our neuron.
# For simplicity in this first step, we will directly set a constant input
# by adding it to the differential equation.

# Let's modify the equation to include a constant input current I
I = 20*mV  # This represents a constant driving force
eqs_with_input = '''
dv/dt = (El - v + I) / tau : volt (unless refractory)
'''
G_input = NeuronGroup(1, eqs_with_input,
                      threshold='v > thresh',
                      reset='v = reset',
                      refractory=5*ms,
                      method='exact')
G_input.v = El


# --- 5. Monitors ---
# Set up monitors to record the neuron's state and spikes.
state_monitor = StateMonitor(G_input, 'v', record=True)
spike_monitor = SpikeMonitor(G_input)

# --- 6. Run Simulation ---
print("Starting simulation...")
run(duration)
print("Simulation finished.")

# --- 7. Plot Results ---
# Plot the membrane potential and the spike times.
if len(state_monitor.t) > 0:
    plot(state_monitor.t/ms, state_monitor.v[0]/mV, label='Membrane Potential')
    if len(spike_monitor.t) > 0:
        plot(spike_monitor.t/ms, [thresh/mV]*len(spike_monitor.t), 'r|', ms=10, label='Spikes')
    xlabel('Time (ms)')
    ylabel('Voltage (mV)')
    title('Simulation of a Single LIF Neuron')
    legend()
    show()
else:
    print("No data recorded. There might be an issue with the simulation setup.")

print(f"Number of spikes: {spike_monitor.num_spikes}")
