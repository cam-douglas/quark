# Purpose: CA3 plasticity parameter sweep with biologically grounded ranges.
# Inputs: CLI args or defaults; writes progress to a JSON status file; logs runs to MLflow via ExperimentLogger.
# Outputs: MLflow runs with params/metrics/artifacts; status JSON updated per run.

import argparse
import io
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from brian2 import *  # noqa: F401,F403

from .ca3_recurrent import build_ca3_recurrent

try:
    from tools_utilities.automation.experiment_logger import ExperimentLogger
except Exception:
    ExperimentLogger = None  # type: ignore

try:
    from testing.testing_frameworks.scientific_validation import ScientificValidator
except Exception:
    ScientificValidator = None  # type: ignore


def _ensure_status(status_path: str) -> None:
    os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
    if not os.path.exists(status_path):
        with open(status_path, "w") as f:
            json.dump({"state": "idle", "total": 0, "completed": 0, "history": []}, f)


def _write_status(status_path: str, payload: Dict) -> None:
    tmp = status_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, status_path)


def _render_weights_histogram(weights_mv: np.ndarray) -> bytes:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4))
    if len(weights_mv):
        ax.hist(weights_mv, bins=20, color='#4cc9f0', edgecolor='#22304a')
        ax.set_title('CA3 recurrent weights (mV)')
        ax.set_xlabel('w (mV)'); ax.set_ylabel('count')
    else:
        ax.text(0.5, 0.5, 'No weights', ha='center', va='center'); ax.axis('off')
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
    return buf.read()


def run_sweep(status_path: str, parallel: int = 1, sim_ms: float = 2000.0) -> None:
    _ensure_status(status_path)

    # Biologically grounded ranges (see prior message)
    a_plus_vals = [0.2, 0.35, 0.5, 0.65]
    a_minus_vals = [0.15, 0.3, 0.45, 0.6]
    w0_vals_mv = [3.0, 5.0, 7.0]
    rate_scales = [0.5, 1.0, 2.0]

    grid: List[Tuple[float, float, float, float]] = []
    for ap in a_plus_vals:
        for am in a_minus_vals:
            for w0 in w0_vals_mv:
                for rs in rate_scales:
                    grid.append((ap, am, w0, rs))

    status = {"state": "running", "total": len(grid), "completed": 0, "history": []}
    _write_status(status_path, status)

    validator = ScientificValidator() if ScientificValidator else None

    for idx, (ap, am, w0, rs) in enumerate(grid, start=1):
        try:
            start_scope()
            panels, labels, step_fn, objs = build_ca3_recurrent(n_neurons=100)
            # Identify structures
            net = next(o for o in objs if isinstance(o, Network))
            inp = next(o for o in objs if isinstance(o, PoissonGroup))
            syn_ca3_rec = None
            for o in objs:
                if isinstance(o, Synapses) and hasattr(o, 'w'):
                    try:
                        if o.source is o.target:
                            syn_ca3_rec = o
                            break
                    except Exception:
                        pass
            if syn_ca3_rec is None:
                raise RuntimeError('CA3 recurrent synapse not found')

            # Apply params
            syn_ca3_rec.w = float(w0) * mV
            syn_ca3_rec.A_plus = float(ap)
            syn_ca3_rec.A_minus = float(am)
            inp.rates = max(1*Hz, 80*Hz * float(rs))

            # Run
            net.run(sim_ms * ms)

            # Metrics
            mon_ca3, mon_ca1 = panels
            duration_s = sim_ms / 1000.0
            mean_rate_ca3 = float(np.mean(mon_ca3.count) / duration_s) if len(mon_ca3.count) else 0.0
            mean_rate_ca1 = float(np.mean(mon_ca1.count) / duration_s) if len(mon_ca1.count) else 0.0
            w_mv = np.array(syn_ca3_rec.w[:] / mV)
            w_mean = float(np.mean(w_mv)) if len(w_mv) else 0.0
            w_std = float(np.std(w_mv)) if len(w_mv) else 0.0
            w_cv = float(w_std / w_mean) if w_mean > 1e-9 else 0.0

            brain_score = 0.0
            if validator is not None:
                try:
                    agi_data = {
                        'spikes_ca3': list(zip(mon_ca3.t/ms, mon_ca3.i)),
                        'spikes_ca1': list(zip(mon_ca1.t/ms, mon_ca1.i)),
                    }
                    res = validator.run_validation(agi_data, 'brain_score')
                    brain_score = float(res.get('score', 0.0))
                except Exception:
                    brain_score = 0.0

            # Log
            if ExperimentLogger is not None:
                logger = ExperimentLogger()
                with logger.start_run(run_name="ca3_sweep", experiment_name="QuarkBrainSimulation"):
                    logger.log_params({
                        'module': 'ca3_sweep',
                        'A_plus': ap,
                        'A_minus': am,
                        'w0_mV': w0,
                        'rate_scale': rs,
                        'sim_ms': sim_ms,
                    })
                    logger.log_metrics({
                        'mean_rate_ca3_hz': mean_rate_ca3,
                        'mean_rate_ca1_hz': mean_rate_ca1,
                        'w_mean_mV': w_mean,
                        'w_std_mV': w_std,
                        'w_cv': w_cv,
                        'brain_score': brain_score,
                    })
                    # Artifact: weight histogram for interesting runs
                    if mean_rate_ca1 > 2.0 or w_cv < 1.0:
                        png = _render_weights_histogram(w_mv)
                        tmp = 'ca3_weights_hist.png'
                        with open(tmp, 'wb') as f:
                            f.write(png)
                        logger.log_artifact(tmp)
                        os.remove(tmp)

            # Update status
            status["completed"] = idx
            status["last"] = {
                'A_plus': ap,
                'A_minus': am,
                'w0_mV': w0,
                'rate_scale': rs,
                'mean_rate_ca3_hz': mean_rate_ca3,
                'mean_rate_ca1_hz': mean_rate_ca1,
                'w_cv': w_cv,
                'brain_score': brain_score,
            }
            status["history"].append(status["last"])  # grow log
            _write_status(status_path, status)
        except Exception as e:
            status["completed"] = idx
            status["last_error"] = str(e)
            _write_status(status_path, status)
            continue

    status["state"] = "done"
    _write_status(status_path, status)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--status_path', type=str, default='logs/ca3_sweep_status.json')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--sim_ms', type=float, default=2000.0)
    args = parser.parse_args(argv)

    try:
        run_sweep(status_path=args.status_path, parallel=args.parallel, sim_ms=args.sim_ms)
        return 0
    except Exception as e:
        _ensure_status(args.status_path)
        _write_status(args.status_path, {"state": "error", "error": str(e)})
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
