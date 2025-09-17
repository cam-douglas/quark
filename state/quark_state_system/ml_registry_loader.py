"""Loader for ML component registry produced by build_ml_index.py"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, List

INDEX_PATH = Path(__file__).resolve().parents[2] / "brain" / "ml" / "ML_INDEX.yaml"

_cache: List[dict] | None = None

def get_ml_components() -> List[Dict]:
    """Return cached list of ML component metadata.

Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""
    global _cache
    if _cache is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"ML index not found: {INDEX_PATH}")
        _cache = yaml.safe_load(INDEX_PATH.read_text())['modules']
    return _cache
