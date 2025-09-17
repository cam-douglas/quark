"""Human Data Utilities

Helper functions for accessing and interpolating quantitative human embryonic
metrics stored in `human_experimental_data.py`.

All functions are pure and ≤50 lines to remain architecture-compliant.
"""

from typing import Dict, List, Tuple

import numpy as np

from .proliferation_validation_types import ExperimentalData
from .human_experimental_data import load_human_experimental_data


_HUMAN_DB: Dict[str, ExperimentalData] = load_human_experimental_data()


def _group_by_metric(metric_name: str) -> List[Tuple[float, ExperimentalData]]:
    """Return (pcw, data) sorted list for a given metric."""
    items = []
    for data in _HUMAN_DB.values():
        if data.metric_name == metric_name and data.developmental_stage.endswith("pcw"):
            stage_str = data.developmental_stage.replace("pcw", "")
            if "-" in stage_str:
                start, end = map(float, stage_str.split("-", 1))
                pcw = (start + end) / 2.0
            else:
                pcw = float(stage_str)
            items.append((pcw, data))
    return sorted(items, key=lambda t: t[0])


def interpolate(metric_name: str, pcw: float) -> float:
    """Linear-interpolate `metric_name` at a given post-conception week (pcw).

    Falls back to nearest neighbour if pcw is outside known range.
    """
    points = _group_by_metric(metric_name)
    if not points:
        raise ValueError(f"No human data for metric {metric_name}")

    xs, ys = zip(*[(p, d.expected_value) for p, d in points])
    if pcw <= xs[0]:
        return ys[0]
    if pcw >= xs[-1]:
        return ys[-1]
    return float(np.interp(pcw, xs, ys))
