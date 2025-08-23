# Purpose: CA3 attractor benchmark - store sparse patterns in CA3 recurrent network and test pattern completion.
# Inputs: CLI args or defaults; patterns defined as subsets of CA3 neurons.
# Outputs: MLflow runs with retrieval metrics; returns metrics dict in library mode.

from typing import Dict, List, Tuple
import argparse
import sys
import numpy as np
from brian2 import *  # noqa: F401,F403

from .ca3_recurrent import build_ca3_recurrent

try:
    from tools_utilities.automation.experiment_logger import ExperimentLogger
except Exception:
    ExperimentLogger = None  # type: ignore


def _select_patterns(n_neurons: int, num_patterns: int, active_frac: float, rng: np.random.Generator) -> List[np.ndarray]:
    patterns: List[np.ndarray] = []
    k = max(1, int(active_frac * n_neurons))
    for _ in range(num_patterns):
        idx = rng.choice(n_neurons, size=k, replace=False)
        mask = np.zeros(n_neurons, dtype=bool)
        mask[idx] = True
        patterns.append(mask)
    return patterns


def _imprint_patterns_on_recurrent_weights(syn_rec: Synapses, patterns: List[np.ndarray], delta_mV: float, w_max_mV: float) -> None:
    # For each pattern, reinforce connections within the active set
    for mask in patterns:
        active = np.where(mask)[0]
        if len(active) == 0:
            continue
        # Strengthen all i->j where both are in pattern and i!=j
        # Caution: large masks can be slow; operate via current weights view
        w = syn_rec.w[:] / mV
        # synapses are in compressed sparse format; we adjust by rule using indices
        I = syn_rec.i[:]
        J = syn_rec.j[:]
        in_pat = np.isin(I, active) & np.isin(J, active) & (I != J)
        w[in_pat] = np.clip(w[in_pat] + delta_mV, 0.0, w_max_mV)
        syn_rec.w[:] = w * mV


def _overlap_score(spike_monitor: SpikeMonitor, pattern_mask: np.ndarray, window_ms: float, t_end_ms: float) -> float:
    # Compute binary activity per neuron in last window and overlap with pattern
    if len(spike_monitor.t) == 0:
        return 0.0
    t = spike_monitor.t / ms
    i = spike_monitor.i
    t_start = max(0.0, float(t_end_ms - window_ms))
    sel = (t >= t_start) & (t <= t_end_ms)
    if not np.any(sel):
        return 0.0
    active_neurons = np.unique(i[sel])
    n = int(spike_monitor.source.N)
    act = np.zeros(n, dtype=bool)
    act[active_neurons] = True
    # normalized overlap m in [0,1]
    if pattern_mask.sum() == 0:
        return 0.0
    m = float(np.logical_and(act, pattern_mask).sum()) / float(pattern_mask.sum())
    return m


def run_attractor_benchmark(num_patterns: int = 5,
                            active_frac: float = 0.1,
                            cue_frac: float = 0.5,
                            imprint_delta_mV: float = 3.0,
                            sim_imprint_ms: float = 200.0,
                            sim_cue_ms: float = 150.0,
                            sim_recall_ms: float = 500.0,
                            seed_val: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed_val)
    start_scope()
    panels, labels, step_fn, objs = build_ca3_recurrent(n_neurons=100)

    # Identify core components
    net = next(o for o in objs if isinstance(o, Network))
    ca3 = None
    inp = None
    syn_rec = None
    for o in objs:
        if isinstance(o, NeuronGroup) and int(o.N) == 100:
            # heuristic: CA3 is the group that has a recurrent synapse; we will confirm later
            if ca3 is None:
                ca3 = o
        if isinstance(o, PoissonGroup):
            inp = o
        if isinstance(o, Synapses) and hasattr(o, 'w'):
            try:
                if o.source is o.target:
                    syn_rec = o
            except Exception:
                pass
    if ca3 is None or inp is None or syn_rec is None:
        raise RuntimeError('Could not locate CA3, input, or recurrent synapses')

    # Patterns
    patterns = _select_patterns(int(ca3.N), num_patterns, active_frac, rng)

    # Imprint
    syn_rec.w_max = max(getattr(syn_rec, 'w_max', 20*mV), 15*mV)
    _imprint_patterns_on_recurrent_weights(syn_rec, patterns, delta_mV=imprint_delta_mV, w_max_mV=float((syn_rec.w_max / mV)))
    net.run(sim_imprint_ms * ms)

    # Choose a pattern and cue a subset of its neurons via extra input rates
    target_idx = rng.integers(0, num_patterns)
    target_mask = patterns[target_idx]
    cue_mask = target_mask.copy()
    # deactivate some to simulate partial cue
    active_indices = np.where(cue_mask)[0]
    rng.shuffle(active_indices)
    keep = int(max(1, cue_frac * len(active_indices)))
    cue_indices = set(active_indices[:keep])

    base_rate = float(inp.rates[0] / Hz) if len(inp.rates) else 80.0
    high_rate = base_rate * 3.0

    # Temporarily increase rates for cue neurons by mapping 1-1 Poisson sources
    # Note: build_ca3_recurrent connects inp→DG; CA3 is driven through DG→CA3; we approximate cue by scaling global input
    # and lowering threshold for cued neurons briefly
    orig_thresh = ca3.thresh[:]
    thr = ca3.thresh[:]
    for idx in cue_indices:
        thr[idx] = (-58.0) * mV  # slightly easier to spike for cue
    ca3.thresh[:] = thr
    inp.rates = high_rate * Hz
    net.run(sim_cue_ms * ms)

    # Restore baseline and observe recall
    ca3.thresh[:] = orig_thresh
    inp.rates = base_rate * Hz
    net.run(sim_recall_ms * ms)

    # Metrics
    mon_ca3, mon_ca1 = panels
    overlap = _overlap_score(mon_ca3, target_mask, window_ms=sim_recall_ms, t_end_ms=sim_imprint_ms + sim_cue_ms + sim_recall_ms)
    duration_s = (sim_imprint_ms + sim_cue_ms + sim_recall_ms) / 1000.0
    mean_rate_ca3 = float(np.mean(mon_ca3.count) / duration_s) if len(mon_ca3.count) else 0.0
    mean_rate_ca1 = float(np.mean(mon_ca1.count) / duration_s) if len(mon_ca1.count) else 0.0

    return {
        'overlap': float(overlap),
        'mean_rate_ca3_hz': float(mean_rate_ca3),
        'mean_rate_ca1_hz': float(mean_rate_ca1),
        'num_patterns': float(num_patterns),
        'active_frac': float(active_frac),
        'cue_frac': float(cue_frac),
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--patterns', type=int, default=5)
    parser.add_argument('--active_frac', type=float, default=0.1)
    parser.add_argument('--cue_frac', type=float, default=0.5)
    parser.add_argument('--imprint_delta_mV', type=float, default=3.0)
    parser.add_argument('--sim_imprint_ms', type=float, default=200.0)
    parser.add_argument('--sim_cue_ms', type=float, default=150.0)
    parser.add_argument('--sim_recall_ms', type=float, default=500.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    metrics = run_attractor_benchmark(num_patterns=args.patterns,
                                      active_frac=args.active_frac,
                                      cue_frac=args.cue_frac,
                                      imprint_delta_mV=args.imprint_delta_mV,
                                      sim_imprint_ms=args.sim_imprint_ms,
                                      sim_cue_ms=args.sim_cue_ms,
                                      sim_recall_ms=args.sim_recall_ms,
                                      seed_val=args.seed)
    if ExperimentLogger is not None:
        try:
            logger = ExperimentLogger()
            with logger.start_run(run_name='ca3_attractor', experiment_name='QuarkBrainSimulation'):
                logger.log_params({
                    'module': 'ca3_attractor',
                    'patterns': args.patterns,
                    'active_frac': args.active_frac,
                    'cue_frac': args.cue_frac,
                    'imprint_delta_mV': args.imprint_delta_mV,
                    'sim_imprint_ms': args.sim_imprint_ms,
                    'sim_cue_ms': args.sim_cue_ms,
                    'sim_recall_ms': args.sim_recall_ms,
                })
                logger.log_metrics(metrics)
        except Exception:
            pass
    print(metrics)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
