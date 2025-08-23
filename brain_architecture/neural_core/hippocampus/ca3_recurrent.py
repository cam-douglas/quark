# Purpose: Build a hippocampal microcircuit with DG→CA3→CA1 pathway plus CA3 recurrent collaterals with STDP and facilitation.
# Inputs: n_neurons (int) optional
# Outputs: panels (SpikeMonitor tuple), labels (tuple[str,str]), step_fn callable, objs list for Network management

from brian2 import *  # noqa: F401,F403

__all__ = ["build_ca3_recurrent"]

def build_ca3_recurrent(n_neurons: int = 100):
    """Return (panels, labels, step_fn, objs) for live viewer integration."""
    defaultclock.dt = 0.1 * ms

    # Base parameters
    El0 = -65 * mV
    tau0 = 20 * ms
    thresh0 = -55 * mV
    reset0 = -70 * mV

    eqs = """
    dv/dt = (El - v)/tau : volt (unless refractory)
    El : volt
    tau : second
    thresh : volt
    reset : volt
    """

    DG = NeuronGroup(n_neurons, eqs, threshold="v>thresh", reset="v=reset", refractory=5*ms, method="exact")
    CA3 = NeuronGroup(n_neurons, eqs, threshold="v>thresh", reset="v=reset", refractory=5*ms, method="exact")
    CA1 = NeuronGroup(n_neurons, eqs, threshold="v>thresh", reset="v=reset", refractory=5*ms, method="exact")

    for grp in (DG, CA3, CA1):
        grp.v = El0
        grp.El = El0
        grp.tau = tau0
        grp.thresh = thresh0
        grp.reset = reset0
    # Make CA1 slightly more excitable
    CA1.thresh = -60 * mV

    # External Poisson input to DG
    inp = PoissonGroup(n_neurons, rates=80 * Hz)
    syn_in = Synapses(inp, DG, on_pre="v_post += 10*mV")
    syn_in.connect(j="i")

    # DG → CA3 mossy fibers (strong but facilitating)
    syn_dg_ca3 = Synapses(DG, CA3, model="""
        w: volt
        du/dt = (0.2 - u)/100*ms : 1 (event-driven)  # facilitation variable
        """, on_pre="""
        v_post += w*u
        u += 0.1*(1-u)
        """)
    syn_dg_ca3.connect(j="i")
    syn_dg_ca3.w = 10 * mV

    # CA3 recurrent collaterals with STDP
    stdp_model = """
    w : volt
    dapre/dt = -apre/20*ms : 1 (event-driven)
    dapost/dt = -apost/20*ms : 1 (event-driven)
    A_plus : 1 (shared)
    A_minus : 1 (shared)
    w_max : volt (shared)
    """
    on_pre = "v_post += w; apre += 1; w = clip(w + A_plus*apost*mV, 0*mV, w_max)"
    on_post = "apost += 1; w = clip(w - A_minus*apre*mV, 0*mV, w_max)"

    syn_ca3_rec = Synapses(CA3, CA3, model=stdp_model, on_pre=on_pre, on_post=on_post, method="euler")
    syn_ca3_rec.connect(condition="i!=j", p=0.1)
    syn_ca3_rec.w = 5 * mV
    syn_ca3_rec.A_plus = 0.4
    syn_ca3_rec.A_minus = 0.4
    syn_ca3_rec.w_max = 15 * mV

    # CA3 → CA1 Schaffer collaterals (plastic as before)
    syn_ca3_ca1 = Synapses(CA3, CA1, model=stdp_model, on_pre=on_pre, on_post=on_post, method="euler")
    syn_ca3_ca1.connect(j="i")
    syn_ca3_ca1.w = 10 * mV
    syn_ca3_ca1.A_plus = 0.5
    syn_ca3_ca1.A_minus = 0.5
    syn_ca3_ca1.w_max = 20 * mV

    mon_ca3 = SpikeMonitor(CA3)
    mon_ca1 = SpikeMonitor(CA1)

    net = Network(DG, CA3, CA1, inp, syn_in, syn_dg_ca3, syn_ca3_rec, syn_ca3_ca1, mon_ca3, mon_ca1)

    panels = (mon_ca3, mon_ca1)
    labels = ("CA3 i", "CA1 i")
    base_rate = 80 * Hz

    def step_fn(step_ms: float, *, _inp=inp):
        rate_scale = getattr(step_fn, "rate_scale", 1.0)
        _inp.rates = max(1*Hz, base_rate * rate_scale)
        net.run(step_ms * ms)

    objs = [DG, CA3, CA1, inp, syn_in, syn_dg_ca3, syn_ca3_rec, syn_ca3_ca1, mon_ca3, mon_ca1, net]
    return panels, labels, step_fn, objs
