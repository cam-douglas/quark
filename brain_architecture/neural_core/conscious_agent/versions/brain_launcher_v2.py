#!/usr/bin/env python3
"""
brain_launcher_v2.py — newborn cognitive brain scaffold (Stage F / N0 / N1)

Features
- Developmental stages: F (fetal), N0 (newborn), N1 (1–3 months)
- Working-memory (WM) slots expansion by stage (F=3, N0=3, N1=4)
- Basal Ganglia sparse Top-K gating (moe_k by stage: F=1, N0=2, N1=2)
- Salience Switch (SN) for internal ↔ task-positive mode
- Sleep & Consolidation Engine (SCE) with NREM/REM replay
- Cerebellar (CB) timing boost (N1)

YAML compatibility
- Works with *either* lowercase module keys (pfc, working_memory, …)
  or uppercase aliases (PFC, WM, BG, Thalamus, DMN, SN, SCE, CB)
- Expects a top-level "architecture_agent" block for sleep_period/length.
  If missing, safe defaults are used.

Usage
  pip install pyyaml
  python brain_launcher_v2.py --connectome connectome_v2.yaml --steps 60 --stage N1
"""

import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import yaml

# ---------------------------
# Messages
# ---------------------------
@dataclass
class Message:
    kind: str              # Observation | Plan | Command | Reward | Modulation | Telemetry | Replay
    src: str
    dst: str
    priority: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

def msg(kind, src, dst, **payload):
    return Message(kind=kind, src=src, dst=dst, payload=payload)

# ---------------------------
# Utilities / helpers
# ---------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def norm_modname(name: str) -> str:
    """
    Normalize module names so the launcher accepts either lowercase IDs
    used earlier ('pfc','working_memory',...) or uppercase aliases that
    appear in some configs ('PFC','WM','BG','Thalamus','DMN','SN','SCE','CB').
    """
    if not name:
        return name
    table = {
        "PFC": "pfc",
        "WM": "working_memory",
        "BG": "basal_ganglia",
        "THALAMUS": "thalamus",
        "DMN": "dmn",
        "SN": "salience",
        "SCE": "sleeper",
        "CB": "cerebellum",
    }
    key = name.upper()
    return table.get(key, name.lower())

# ---------------------------
# Base Module
# ---------------------------
class Module:
    def __init__(self, name: str, spec: Dict[str, Any]):
        self.name = name
        self.spec = spec
        self.state: Dict[str, Any] = {}

    def step(self, inbox: List[Message], ctx: Dict[str, Any]) -> Tuple[List[Message], Dict[str, Any]]:
        raise NotImplementedError

# ---------------------------
# Implementations
# ---------------------------
class PFC(Module):
    def step(self, inbox, ctx):
        wm_conf = ctx.get("wm_confidence", 0.5)
        arousal = ctx["global"].get("arousal", 0.5)
        mode = ctx["global"].get("mode", "internal")
        plan = {"seq": [("think", 1), ("rehearse", 1 if mode=="task-positive" else 0)], "goal": "homeostasis"}
        out = [msg("Command", self.name, "working_memory", action="rehearse", plan=plan)]
        telemetry = {"confidence": clamp01(0.35 + 0.45*wm_conf + 0.2*arousal), "demand": 0.10}
        return out, telemetry

class BasalGanglia(Module):
    def step(self, inbox, ctx):
        da = ctx["modulators"]["DA"]
        ach = ctx["modulators"]["ACh"]
        fatigue = ctx["global"]["fatigue"]
        moe_k = ctx["policy"]["moe_k"]
        candidates = ["pfc", "working_memory", "dmn", "hippocampus"]
        if "cerebellum" in ctx["topology"]["modules"]:
            candidates.append("cerebellum")
        scored = []
        for c in candidates:
            base = 0.5
            if c == "working_memory":
                base += 0.2*ach
            if c == "dmn":
                base += 0.2*(1.0 - ctx["attention"]["task_bias"])
            if c == "cerebellum":
                base += 0.1
            score = base + 0.4*da - 0.2*fatigue + random.uniform(-0.05, 0.05)
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = [c for c,_ in scored[:moe_k]]
        out = [msg("Command", self.name, "thalamus", activate=chosen)]
        telemetry = {"confidence": clamp01(0.5 + 0.4*da - 0.2*fatigue), "demand": 0.05, "chosen": chosen}
        return out, telemetry

class Thalamus(Module):
    def step(self, inbox, ctx):
        out = []
        for m in inbox:
            if m.kind == "Command" and "activate" in m.payload:
                out.append(msg("Telemetry", self.name, "architecture_agent", activated=m.payload["activate"]))
            if m.kind in ("Command", "Observation", "Plan"):
                if "working_memory" in ctx["topology"]["modules"]:
                    out.append(msg(m.kind, self.name, "working_memory", **m.payload))
        telemetry = {"confidence": clamp01(0.6 * ctx["attention"].get("task_bias", 0.6)), "demand": 0.06}
        return out, telemetry

class WorkingMemory(Module):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        slots = int(spec.get("slots", 3))
        self.slots = [{"vec": [0.0]*8, "age": 0, "precision": 0.45} for _ in range(slots)]

    def configure(self, slots: int):
        current = len(self.slots)
        if slots > current:
            for _ in range(slots - current):
                self.slots.append({"vec": [0.0]*8, "age": 0, "precision": 0.45})
        elif slots < current:
            self.slots = self.slots[:slots]

    def step(self, inbox, ctx):
        rehearsed = False
        for m in inbox:
            if m.kind == "Command" and m.payload.get("action") == "rehearse":
                for s in self.slots:
                    s["precision"] = min(1.0, s["precision"] + 0.05)
                    s["age"] = 0
                rehearsed = True
            if m.kind == "Replay":
                for s in self.slots:
                    s["precision"] = min(1.0, s["precision"] + 0.02)
        for s in self.slots:
            s["age"] += 1
            s["precision"] = max(0.1, s["precision"] - 0.01)
        wm_conf = sum(s["precision"] for s in self.slots) / max(1, len(self.slots))
        telemetry = {"confidence": wm_conf, "demand": 0.05, "rehearsed": rehearsed, "slots": len(self.slots)}
        out = [msg("Telemetry", self.name, "architecture_agent", wm=telemetry)]
        return out, telemetry

class Hippocampus(Module):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.buffer: List[Dict[str, Any]] = []

    def step(self, inbox, ctx):
        if ctx["global"]["state"] == "wake":
            self.buffer.append({"t": ctx["global"]["t"], "mode": ctx["global"]["mode"]})
            if len(self.buffer) > 256:
                self.buffer = self.buffer[-256:]
        telemetry = {"confidence": 0.6, "demand": 0.04, "episodes": len(self.buffer)}
        out = [msg("Telemetry", self.name, "architecture_agent", hc=len(self.buffer))]
        return out, telemetry

class DMN(Module):
    def step(self, inbox, ctx):
        mode = ctx["global"].get("mode", "internal")
        conf = 0.6 if mode == "internal" else 0.3
        out = []
        if mode == "internal" and ctx["global"]["state"] == "wake":
            out.append(msg("Plan", self.name, "pfc", idea="self-talk"))
        telemetry = {"confidence": conf, "demand": 0.05, "mode": mode}
        return out, telemetry
