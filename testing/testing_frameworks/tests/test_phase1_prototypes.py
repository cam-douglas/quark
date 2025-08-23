"""Unit tests for Phase-1 neural prototypes."""

# Ensure repo root on sys.path
import os, sys, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------
# Thalamic Relay
# ---------------------------------------------------------------------

def test_thalamic_relay_gate():
    from brain_architecture.neural_core.thalamus.relay_nucleus import ThalamicRelay

    relay = ThalamicRelay(gate_open=False)
    assert relay.receive("ping") is None  # closed gate

    relay.open_gate()
    out = relay.receive("ping")
    assert out == "ping"
    stats = relay.stats()
    assert stats.total_received == 2 and stats.total_forwarded == 1


# ---------------------------------------------------------------------
# Hippocampal STDP
# ---------------------------------------------------------------------


def test_stdp_weight_bounds():
    from brain_architecture.neural_core.hippocampus.stdp_synapse import STDPSynapse, STDPParams

    params = STDPParams(A_plus=1.0, A_minus=1.0, w_min=0.0, w_max=1.0)
    syn = STDPSynapse(weight=0.5, params=params)

    # Strong LTP then LTD to drive to bounds
    syn.pre_spike(0.0)
    syn.post_spike(1.0)  # Δt positive -> LTP
    assert 0.5 <= syn.weight() <= 1.0

    syn.post_spike(10.0)
    syn.pre_spike(11.0)  # Δt negative -> LTD
    assert 0.0 <= syn.weight() <= 1.0


# ---------------------------------------------------------------------
# Basal Ganglia Gate
# ---------------------------------------------------------------------


def test_action_gate_selection():
    from brain_architecture.neural_core.basal_ganglia.simple_gate import ActionGate

    gate = ActionGate(threshold=0.6)
    salience = {"a": 0.5, "b": 0.9, "c": 0.4}
    res = gate.select(salience)
    assert res.selected_channel == "b"

    salience = {"x": 0.2, "y": 0.3}
    res = gate.select(salience)
    assert res.selected_channel is None


# ---------------------------------------------------------------------
# Proto-cortex Sheet
# ---------------------------------------------------------------------


def test_layer_sheet_homeostasis():
    from brain_architecture.neural_core.proto_cortex.layer_sheet import LayerSheet

    sheet = LayerSheet(n=50)
    before = sheet.mean_activity()
    for _ in range(300):
        sheet.step()
    after = sheet.mean_activity()
    # Activity should move towards 0.1 target
    assert abs(after - 0.1) < abs(before - 0.1)
