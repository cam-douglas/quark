from brian2 import NeuronGroup, mV, ms


def create_thalamic_nucleus(n_neurons=100):
    """
    Creates a model of a thalamic nucleus.

    This function returns a Brian2 NeuronGroup representing a simplified
    population of thalamic relay neurons. For this initial model, we use
    standard Leaky Integrate-and-Fire (LIF) neurons.

    In future iterations, these could be replaced with more complex models
    that exhibit characteristic thalamic behaviors, such as bursting dynamics
    (e.g., using Izhikevich neurons or including low-threshold calcium channels).

    Args:
        n_neurons (int): The number of neurons in the nucleus.

    Returns:
        NeuronGroup: A Brian2 NeuronGroup representing the thalamic nucleus.
    """
    # Standard LIF neuron parameters
    tau = 10 * ms
    El = -65 * mV
    thresh = -50 * mV
    reset = -70 * mV

    eqs = f'''
    dv/dt = (El - v) / tau : volt (unless refractory)
    El : volt
    tau : second
    thresh : volt
    reset : volt
    '''

    # Create the NeuronGroup
    thalamus = NeuronGroup(
        n_neurons,
        eqs,
        threshold='v > thresh',
        reset='v = reset',
        refractory=5 * ms,
        method='exact'
    )

    # Initialize membrane parameters
    thalamus.v = El
    thalamus.El = El
    thalamus.tau = tau
    thalamus.thresh = thresh
    thalamus.reset = reset

    return thalamus


if __name__ == '__main__':
    # --- Example Usage ---
    # This demonstrates how to create the thalamic nucleus and check its properties.

    nucleus = create_thalamic_nucleus(50)

    print("--- Thalamic Nucleus Model ---")
    print(f"Number of neurons: {len(nucleus)}")
    print(f"Model equations:\n{nucleus.equations}")
    print(f"Initial membrane potential (first 5 neurons): {nucleus.v[:5]}")

    # This part is just for demonstration and won't run a full simulation.
    # The actual simulation will be handled in the sensory_relay.py script.
