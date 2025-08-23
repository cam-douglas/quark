from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    # Module execution
    from .thalamus_model import create_thalamic_nucleus  # type: ignore
except Exception:
    # Script execution
    from thalamus_model import create_thalamic_nucleus

"""
Realtime Thalamic Relay Demo
- Runs the thalamic relay in small time chunks
- Updates a raster plot live to show relayed spikes in realtime
"""


def run_realtime(total_duration_ms: float = 2000.0, step_ms: float = 20.0):
    defaultclock.dt = 0.1*ms

    # Network setup
    thalamic_nucleus = create_thalamic_nucleus(n_neurons=100)
    thalamic_nucleus.thresh = -55*mV
    thalamic_nucleus.tau = 30*ms

    n_inputs = 100
    input_firing_rate = 100*Hz
    sensory_input = PoissonGroup(n_inputs, rates=input_firing_rate)

    syn = Synapses(sensory_input, thalamic_nucleus, on_pre='v_post += 12*mV')
    syn.connect(j='i')

    mon_in = SpikeMonitor(sensory_input)
    mon_th = SpikeMonitor(thalamic_nucleus)

    # Live plotting setup
    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_in, ax_th = axes
    ax_in.set_title('Sensory Input (live)')
    ax_th.set_title('Thalamic Nucleus (relayed spikes, live)')
    ax_th.set_xlabel('Time (ms)')
    ax_in.set_ylabel('Neuron')
    ax_th.set_ylabel('Neuron')

    # Containers for plotted data
    in_times, in_inds = [], []
    th_times, th_inds = [], []

    # Convert total to steps
    num_steps = int(np.ceil(total_duration_ms / step_ms))
    t_start = time.time()
    sim_time_ms = 0.0

    for step in range(num_steps):
        run(step_ms*ms)
        sim_time_ms += step_ms
        
        # Append new spikes
        in_times = list(mon_in.t/ms)
        in_inds = list(mon_in.i)
        th_times = list(mon_th.t/ms)
        th_inds = list(mon_th.i)

        # Clear and redraw
        ax_in.cla()
        ax_th.cla()
        ax_in.set_title('Sensory Input (live)')
        ax_th.set_title('Thalamic Nucleus (relayed spikes, live)')
        ax_th.set_xlabel('Time (ms)')
        ax_in.set_ylabel('Neuron')
        ax_th.set_ylabel('Neuron')
        ax_in.plot(in_times, in_inds, '.k', markersize=3)
        ax_th.plot(th_times, th_inds, '.r', markersize=3)
        ax_in.set_xlim(max(0, sim_time_ms-500), sim_time_ms)
        ax_th.set_xlim(max(0, sim_time_ms-500), sim_time_ms)
        ax_in.set_ylim(-1, n_inputs+1)
        ax_th.set_ylim(-1, len(thalamic_nucleus)+1)
        plt.pause(0.01)

    elapsed = time.time() - t_start
    print(f"Realtime demo completed. Simulated {total_duration_ms:.0f} ms in {elapsed:.2f} s.")
    print(f"Sensory spikes: {len(mon_in.t)}, Thalamic spikes: {len(mon_th.t)}")
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    run_realtime(total_duration_ms=3000.0, step_ms=25.0)
