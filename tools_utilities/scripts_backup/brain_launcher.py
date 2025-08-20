#!/usr/bin/env python3
"""
brain_launcher.py — newborn cognitive brain scaffold

- Loads a connectome.yaml (provided earlier)
- Instantiates stub modules for PFC, BG, Thalamus, WM, HC, DMN, Attention, and ArchitectureAgent
- Runs a simple tick loop with a message bus and prints activation traces

Dependencies: pyyaml
Usage:
  python brain_launcher.py --connectome connectome.yaml --steps 10
"""

import argparse
import time
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import yaml

# ---------------------------
# Message types
# ---------------------------
@dataclass
class Message:
    kind: str              # Observation | Plan | Command | Reward | Modulation | Telemetry
    src: str
    dst: str
    priority: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

# Convenience constructors
def msg(kind, src, dst, **payload):
    return Message(kind=kind, src=src, dst=dst, payload=payload)

# ---------------------------
# Base Module
# ---------------------------
class Module:
    def __init__(self, name: str, spec: Dict[str, Any]):
        self.name = name
        self.spec = spec
        self.state: Dict[str, Any] = {}

    def step(self, inbox: List[Message], ctx: Dict[str, Any]) -> Tuple[List[Message], Dict[str, Any]]:
        """
        Process input messages and produce output messages.
        Return (outbox, telemetry). Telemetry must include "confidence" [0..1] and "demand".
        """
        raise NotImplementedError

# ---------------------------
# Stub Implementations (lightweight, deterministic-ish with randomness)
# ---------------------------
class PFC(Module):
    def step(self, inbox, ctx):
        wm_conf = ctx.get("wm_confidence", 0.5)
        arousal = ctx["global"].get("arousal", 0.5)
        plan = {"seq": [("think", 1), ("rehearse", 1)], "goal": "homeostasis"}
        out = [msg("Command", self.name, "working_memory", action="rehearse", plan=plan)]
        telemetry = {
            "confidence": min(1.0, 0.4 + 0.4*wm_conf + 0.2*arousal),
            "demand": 0.1,
        }
        return out, telemetry

class BasalGanglia(Module):
    def step(self, inbox, ctx):
        # Gate WM update based on DA (reward signal) and arousal
        da = ctx["modulators"]["DA"]
        gate_update = da > 0.4 and random.random() < (0.3 + 0.5*da)
        out = []
        if gate_update:
            out.append(msg("Command", self.name, "thalamus", route="pfc->working_memory"))
        telemetry = {"confidence": 0.5 + 0.5*da, "demand": 0.05}
        return out, telemetry

class Thalamus(Module):
    def step(self, inbox, ctx):
        # Simple relay: route commands/data among known modules; respect capacity/attention crudely
        attention_bias = ctx["attention"].get("bias", 1.0)
        out = []
        for m in inbox:
            if m.kind in ("Command", "Observation"):
                # Forward to declared outputs or PFC/WM by default
                if "working_memory" in ctx["topology"]["modules"]:
                    out.append(msg(m.kind, self.name, "working_memory", **m.payload))
                out.append(msg("Telemetry", self.name, "architecture_agent", routed=True))
        telemetry = {"confidence": min(1.0, 0.6 * attention_bias), "demand": 0.06}
        return out, telemetry

class WorkingMemory(Module):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.slots = [{"vec": [0.0]*8, "age": 0, "precision": 0.5} for _ in range(3)]

    def step(self, inbox, ctx):
        rehearsed = False
        for m in inbox:
            if m.kind == "Command" and m.payload.get("action") == "rehearse":
                for s in self.slots:
                    s["precision"] = min(1.0, s["precision"] + 0.05)
                    s["age"] = 0
                rehearsed = True
        for s in self.slots:
            s["age"] += 1
            s["precision"] = max(0.1, s["precision"] - 0.01)
        telemetry = {
            "confidence": sum(s["precision"] for s in self.slots) / (len(self.slots)+1e-6),
            "demand": 0.05,
            "rehearsed": rehearsed,
        }
        out = [msg("Telemetry", self.name, "architecture_agent", wm=telemetry)]
        return out, telemetry

class Hippocampus(Module):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.buffer: List[Dict[str, Any]] = []

    def step(self, inbox, ctx):
        # Append simple episodic traces
        self.buffer.append({"t": ctx["global"]["t"], "arousal": ctx["global"]["arousal"]})
        if len(self.buffer) > 64:
            self.buffer = self.buffer[-64:]
        telemetry = {"confidence": 0.6, "demand": 0.04, "episodes": len(self.buffer)}
        out = [msg("Telemetry", self.name, "architecture_agent", hc=len(self.buffer))]
        return out, telemetry

class DMN(Module):
    def step(self, inbox, ctx):
        mode = ctx["global"].get("mode", "internal")
        conf = 0.6 if mode == "internal" else 0.3
        out = [msg("Plan", self.name, "pfc", idea="self-talk")] if mode == "internal" else []
        telemetry = {"confidence": conf, "demand": 0.05, "mode": mode}
        return out, telemetry

class Attention(Module):
    def step(self, inbox, ctx):
        # Compute a crude bias toward task-positive based on NE
        ne = ctx["modulators"]["NE"]
        bias = 0.8 if ne > 0.5 else 0.6
        out = [msg("Telemetry", self.name, "architecture_agent", bias=bias)]
        telemetry = {"confidence": bias, "demand": 0.02, "bias": bias}
        return out, telemetry

class ArchitectureAgent(Module):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.modulators = {"DA": 0.3, "NE": 0.4, "ACh": 0.5, "5HT": 0.5}
        self.arousal = 0.5
        self.mode = "internal"  # internal vs task-positive

    def step(self, inbox, ctx):
        # Aggregate telemetry
        for m in inbox:
            if m.src == "attention" and "bias" in m.payload:
                ctx["attention"]["bias"] = m.payload["bias"]
            if m.src == "working_memory" and "wm" in m.payload:
                ctx["wm_confidence"] = m.payload["wm"]["confidence"]
        # Simple salience switch policy
        if ctx["attention"]["bias"] > 0.7:
            self.mode = "task-positive"
        else:
            self.mode = "internal"
        # Modulator drift
        self.modulators["DA"] = max(0.0, min(1.0, self.modulators["DA"] + random.uniform(-0.02, 0.02)))
        self.modulators["NE"] = max(0.0, min(1.0, self.modulators["NE"] + (0.02 if self.mode=="task-positive" else -0.01)))
        # Emit global modulation
        out = [
            msg("Modulation", self.name, "attention", NE=self.modulators["NE"]),
            msg("Modulation", self.name, "basal_ganglia", DA=self.modulators["DA"]),
            msg("Command", self.name, "pfc", mode=self.mode),
        ]
        telemetry = {
            "confidence": 0.7,
            "demand": 0.03,
            "modulators": self.modulators.copy(),
            "mode": self.mode,
        }
        return out, telemetry

# ---------------------------
# Loader & Orchestrator
# ---------------------------
MODULE_IMPLS = {
    "architecture_agent": ArchitectureAgent,
    "pfc": PFC,
    "basal_ganglia": BasalGanglia,
    "thalamus": Thalamus,
    "working_memory": WorkingMemory,
    "hippocampus": Hippocampus,
    "dmn": DMN,
    "attention": Attention,
}

class Brain:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.modules: Dict[str, Module] = {}
        self.mailboxes: Dict[str, List[Message]] = {}
        # Collect module specs: "modules" block + top-level "attention" + "architecture_agent"
        to_instantiate = {}
        to_instantiate["architecture_agent"] = config.get("architecture_agent", {"role":"AA"})
        for k, v in config.get("modules", {}).items():
            to_instantiate[k] = v
        if "attention" in config:
            to_instantiate["attention"] = config["attention"]
        # Instantiate
        for name, spec in to_instantiate.items():
            impl = MODULE_IMPLS.get(name)
            if impl is None:
                # Default generic Module if unknown
                impl = Module  # will error on step; but this helps detect typos early
            self.modules[name] = impl(name, spec) if impl is not Module else Module(name, spec)
            self.mailboxes[name] = []
        # Activate initial modulation broadcast
        self.t = 0

    def step(self):
        self.t += 1
        ctx = {
            "global": {"t": self.t, "arousal": 0.5, "mode": "internal"},
            "modulators": {"DA": 0.3, "NE": 0.4, "ACh": 0.5, "5HT": 0.5},
            "attention": {"bias": 0.6},
            "wm_confidence": 0.5,
            "topology": {"modules": list(self.modules.keys())},
        }
        # Let AA set the true global values by running first
        aa = self.modules["architecture_agent"]
        aa_out, aa_tel = aa.step(self.mailboxes["architecture_agent"], ctx)
        self.mailboxes["architecture_agent"].clear()
        ctx["modulators"] = aa_tel["modulators"]
        ctx["global"]["mode"] = aa_tel["mode"]

        # Post AA messages
        for m in aa_out:
            self.mailboxes[m.dst].append(m)

        # Run the rest in a fixed order
        order = [k for k in self.modules.keys() if k != "architecture_agent"]
        telemetry_log = {"t": self.t, "mode": ctx["global"]["mode"], "mods": ctx["modulators"]}
        for name in order:
            mod = self.modules[name]
            inbox = self.mailboxes[name]
            self.mailboxes[name] = []
            out, tel = mod.step(inbox, ctx)
            telemetry_log[name] = tel
            for m in out:
                if m.dst in self.mailboxes:
                    self.mailboxes[m.dst].append(m)
        return telemetry_log

def load_connectome(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", default="src/config/connectome.yaml", help="Path to connectome.yaml")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    cfg = load_connectome(args.connectome)
    brain = Brain(cfg)

    print("=== newborn brain scaffold ===")
    for t in range(args.steps):
        tel = brain.step()
        # Pretty trace
        mods = tel["mods"]
        print(f"[t={tel['t']:02d}] mode={tel['mode']:<13} DA={mods['DA']:.2f} NE={mods['NE']:.2f}  "
              f"PFC.conf={tel.get('pfc',{}).get('confidence',0):.2f}  "
              f"WM.conf={tel.get('working_memory',{}).get('confidence',0):.2f}  "
              f"BG.conf={tel.get('basal_ganglia',{}).get('confidence',0):.2f}  "
              f"Thal.conf={tel.get('thalamus',{}).get('confidence',0):.2f}  "
              f"DMN.conf={tel.get('dmn',{}).get('confidence',0):.2f}  "
              f"Att.bias={tel.get('attention',{}).get('bias',0):.2f}")

if __name__ == "__main__":
    main()
