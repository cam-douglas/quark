# Neural Core Live Viewer & CA3 Sweep

## Live Viewer
- Start server:
  ```bash
  python3 -m brain_architecture.neural_core.live_viewer
  ```
- Open viewer: http://127.0.0.1:8011
- Tabs: Thalamus, Hippocampus, Hippocampus+CA3R
- Controls: input rate, threshold, STDP A+/A-, motivational bias; validation runs automatically.
- MLflow dashboard: http://127.0.0.1:8050 (launch separately)

## CA3 Plasticity Sweep
- CLI:
  ```bash
  python3 -m brain_architecture.neural_core.hippocampus.ca3_sweep --sim_ms 2000 --parallel 1
  ```
- From viewer: click "Start CA3 Sweep"; progress shows near the mode label; status JSON at `logs/ca3_sweep_status.json`.
- Logged metrics: CA3/CA1 mean rates, CA3 weight stats, placeholder Brain-Score.
- Artifacts: CA3 weights histogram for interesting runs.

## Biological Notes & Citations
- STDP windows and magnitudes for hippocampal pyramidal neurons suggest A⁺≈0.2–0.6, A⁻≈0.15–0.5 with τ≈20 ms.
  - Debanne D, Inglebert Y, Russier M. Plasticity of intrinsic neuronal excitability. Curr Opin Neurobiol. 2019.
  - Sjöström PJ, Rancz EA, Roth A, Häusser M. Dendritic excitability and synaptic plasticity. Physiol Rev. 2008.
  - Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W. Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. Science. 2011.
- DG→CA3 (mossy fiber) synapses exhibit facilitation and large EPSPs; CA3 recurrent collaterals are weaker but plastic.
  - Nicoll RA, Schmitz D. Synaptic plasticity at hippocampal mossy fibre synapses. Nat Rev Neurosci. 2005.
- Granule cell spontaneous firing is sparse (sub-Hz to a few Hz), modeled via ensemble Poisson input scaling in simulations.

Calibration: We initialize DG→CA3 around 8–12 mV and CA3 recurrent 3–7 mV, then sweep A⁺/A⁻ and rate scales to map stable regimes.
