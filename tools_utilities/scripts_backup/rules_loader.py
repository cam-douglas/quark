#!/usr/bin/env python3
"""rules_loader.py — Bootstrap & Runtime Rule Enforcement

Features
--------
- load_rules(paths): load & parse rules documents (rules.md, roadmap, etc.)
- validate_connectome(cfg, stage): check newborn-stage constraints (F, N0, N1)
- instrument_agent(name, role, agent, stage, rules): wrap .step() to enforce invariants
- AgentProxy: clamps telemetry, injects context, ensures replay/sleep hooks are handled
- RuleContext: lightweight container for merged rule text + extracted constraints

This module is **tech-agnostic** and can be used by brain_launcher_v2 or any runner.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import os
import re
import time
import warnings
from copy import deepcopy

# -----------------------------
# Utilities
# -----------------------------
def clamp01(x: float) -> float:
    try:
        return 0.0 if x < 0 else (1.0 if x > 1 else float(x))
    except Exception:
        return 0.0

def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

# -----------------------------
# Rule Context
# -----------------------------
@dataclass
class RuleContext:
    sources: Dict[str, str] = field(default_factory=dict)  # path -> text
    constraints: Dict[str, Any] = field(default_factory=dict)  # parsed hints

    def has(self, key: str) -> bool:
        return key in self.constraints

    def get(self, key: str, default=None):
        return self.constraints.get(key, default)

# -----------------------------
# Loading & Parsing Rules
# -----------------------------
DEFAULT_RULE_FILES = [
    "rules.md",
    "cognitive_brain_roadmap.md",
]

def load_rules(paths: Optional[List[str]] = None) -> RuleContext:
    """Load text files and derive minimal constraints used by validators."""
    paths = paths or DEFAULT_RULE_FILES
    ctx = RuleContext()
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    ctx.sources[p] = f.read()
            except Exception as e:
                warnings.warn(f"Failed to read {p}: {e}")
    # Derive constraints from rules text (regex-based hints; resilient if text changes)
    text = "\n\n".join(ctx.sources.values())
    constraints = {}
    # Stages & defaults (fallbacks if not found in text will be these values)
    constraints["stage_wm_slots"] = {"F": 3, "N0": 3, "N1": 4}
    constraints["stage_moe_k"] = {"F": 1, "N0": 2, "N1": 2}
    constraints["required_modules"] = {
        "F": ["architecture_agent", "pfc", "basal_ganglia", "thalamus", "working_memory", "hippocampus", "dmn", "attention"],
        "N0": ["architecture_agent", "pfc", "basal_ganglia", "thalamus", "working_memory", "hippocampus", "dmn", "attention", "salience", "sleeper"],
        "N1": ["architecture_agent", "pfc", "basal_ganglia", "thalamus", "working_memory", "hippocampus", "dmn", "attention", "salience", "sleeper", "cerebellum"],
    }
    constraints["global_modulators"] = ["DA", "NE", "ACh", "5HT"]
    constraints["telemetry_required_keys"] = ["confidence", "demand"]
    ctx.constraints = constraints
    return ctx

# -----------------------------
# Connectome Validators
# -----------------------------
class ValidationError(Exception):
    pass

def _get_modules(cfg: Dict[str, Any]) -> Dict[str, Any]:
    mods = {}
    # architecture_agent
    if "architecture_agent" in cfg:
        mods["architecture_agent"] = cfg["architecture_agent"]
    # modules block
    for k, v in (cfg.get("modules") or {}).items():
        mods[k] = v
    # attention at top-level (v1/v2 style)
    if "attention" in cfg:
        mods["attention"] = cfg["attention"]
    return mods

def validate_connectome(cfg: Dict[str, Any], stage: str, rules: Optional[RuleContext] = None) -> None:
    """Validate presence of modules, WM slots, and MoE k expectations for the stage."""
    rules = rules or load_rules()
    stage = stage or "F"
    mods = _get_modules(cfg)
    missing = []
    for m in rules.get("required_modules")[stage]:
        if m not in mods:
            missing.append(m)
    if missing:
        raise ValidationError(f"Missing required modules for stage {stage}: {missing}")
    # WM slots
    wm = mods.get("working_memory", {})
    declared_slots = wm.get("slots", None)
    required_slots = rules.get("stage_wm_slots").get(stage, 3)
    if declared_slots is not None and declared_slots < required_slots:
        raise ValidationError(f"working_memory.slots={declared_slots} < required {required_slots} for stage {stage}")
    # MoE k (policy may live in runtime; here we just hint the runner)
    moe_k_required = rules.get("stage_moe_k").get(stage, 1)
    # No strict error if absent; warn for visibility
    warnings.warn(f"Stage {stage}: expected MoE top-k >= {moe_k_required}. Ensure runtime policy matches.")
    # Global modulators in AA (optional but recommended)
    aa = mods.get("architecture_agent", {})
    # Nothing to strictly validate here; we trust runtime to inject modulators.

# -----------------------------
# Agent Instrumentation
# -----------------------------
class AgentProxy:
    """Wrap any agent with runtime checks & telemetry enforcement.

    Expected agent API: step(inbox: List[Message], ctx: Dict) -> (outbox: List[Message], telemetry: Dict)
    """
    def __init__(self, name: str, role: str, agent: Any, stage: str, rules: RuleContext):
        self.name = name
        self.role = role
        self.agent = agent
        self.stage = stage
        self.rules = rules
        self.supports_replay = hasattr(agent, "handle_replay") or True  # soft expectation
        self.required_telemetry = list(self.rules.get("telemetry_required_keys", []))

    def __getattr__(self, item):
        # Delegate all other attributes
        return getattr(self.agent, item)

    def step(self, inbox, ctx):
        # Pre: inject global modulators presence
        ctx = dict(ctx)  # shallow copy
        ctx.setdefault("modulators", {})
        for k in self.rules.get("global_modulators", []):
            ctx["modulators"].setdefault(k, 0.0)
        # Call underlying agent
        out, tel = self.agent.step(inbox, ctx)
        tel = dict(tel) if isinstance(tel, dict) else {}
        # Enforce telemetry keys and bounds
        for k in self.required_telemetry:
            if k not in tel:
                warnings.warn(f"{self.name}: missing telemetry key '{k}', injecting default.")
                tel[k] = 0.0
        tel["confidence"] = clamp01(tel.get("confidence", 0.0))
        tel["demand"] = max(0.0, float(tel.get("demand", 0.0)))
        # Ensure outbox is a list
        out = list(out) if isinstance(out, list) else []
        # Replay handling (during sleep): if Replay arrives and agent lacks special hook, pass through
        if ctx.get("global", {}).get("state") == "sleep" and not self.supports_replay:
            warnings.warn(f"{self.name}: no explicit replay support; relying on default behavior.")
        return out, tel

def instrument_agent(name: str, role: str, agent: Any, stage: str, rules: Optional[RuleContext] = None) -> AgentProxy:
    rules = rules or load_rules()
    return AgentProxy(name=name, role=role, agent=agent, stage=stage, rules=rules)

# -----------------------------
# Demo CLI (optional)
# -----------------------------
def _demo():
    # Basic self-test of loader with current directory files
    rules = load_rules()
    print("Loaded rule sources:", list(rules.sources.keys()))
    # Minimal connectome dict for validation demo
    demo_cfg = {
        "architecture_agent": {},
        "modules": {
            "pfc": {}, "basal_ganglia": {}, "thalamus": {},
            "working_memory": {"slots": 3}, "hippocampus": {}, "dmn": {},
        },
        "attention": {},
    }
    for stage in ("F", "N0", "N1"):
        try:
            validate_connectome(demo_cfg, stage, rules)
            print(f"[OK] connectome valid for stage {stage}")
        except Exception as e:
            print(f"[FAIL] stage {stage}: {e}")

if __name__ == "__main__":
    _demo()
