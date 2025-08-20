# connectome/schema.py
# Minimal schema & helpers for reading connectome.yaml without heavy deps beyond PyYAML.

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
import os

@dataclass
class ModuleSpec:
    id: str
    label: str
    role: str
    population: int
    excitatory_ratio: float
    required_links: List[str] = field(default_factory=list)
    gate_targets: List[str] = field(default_factory=list)
    activation: bool = True  # True for active modules, False for dormant

@dataclass
class Defaults:
    small_world: Dict[str, Any]
    inter_module: Dict[str, Any]
    neuromodulators: Dict[str, float]

@dataclass
class Policies:
    topology: Dict[str, Any]
    e_i_balance: Dict[str, Any]
    gating: Dict[str, Any]
    routing: Dict[str, Any]
    weights: Dict[str, Any]

@dataclass
class Exports:
    dir: str
    formats: List[str]

@dataclass
class ConnectomeConfig:
    version: int
    metadata: Dict[str, Any]
    defaults: Defaults
    modules: List[ModuleSpec]
    policies: Policies
    exports: Exports

def _dict_to_defaults(d: Dict[str, Any]) -> Defaults:
    return Defaults(
        small_world=d.get("small_world", {}),
        inter_module=d.get("inter_module", {}),
        neuromodulators=d.get("neuromodulators", {}),
    )

def _dict_to_policies(d: Dict[str, Any]) -> Policies:
    return Policies(
        topology=d.get("topology", {}),
        e_i_balance=d.get("e_i_balance", {}),
        gating=d.get("gating", {}),
        routing=d.get("routing", {}),
        weights=d.get("weights", {}),
    )

def _dict_to_exports(d: Dict[str, Any]) -> Exports:
    return Exports(
        dir=d.get("dir", "connectome/exports"),
        formats=d.get("formats", ["graphml", "json", "manifests"]),
    )

def _dict_to_module(m: Dict[str, Any]) -> ModuleSpec:
    return ModuleSpec(
        id=m["id"],
        label=m["label"],
        role=m["role"],
        population=int(m["population"]),
        excitatory_ratio=float(m["excitatory_ratio"]),
        required_links=list(m.get("required_links", [])),
        gate_targets=list(m.get("gate_targets", [])),
        activation=bool(m.get("activation", True)),
    )

def load_config(path: str = "connectome/connectome.yaml") -> ConnectomeConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Connectome config not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    modules = [_dict_to_module(m) for m in raw.get("modules", [])]
    cfg = ConnectomeConfig(
        version=int(raw.get("version", 1)),
        metadata=raw.get("metadata", {}),
        defaults=_dict_to_defaults(raw.get("defaults", {})),
        modules=modules,
        policies=_dict_to_policies(raw.get("policies", {})),
        exports=_dict_to_exports(raw.get("exports", {})),
    )
    os.makedirs(cfg.exports.dir, exist_ok=True)
    return cfg
