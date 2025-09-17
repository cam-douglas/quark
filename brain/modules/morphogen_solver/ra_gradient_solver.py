"""Retinoic Acid (RA) Gradient Solver

Simple 3-D steady-state analytic model for dorsal–ventral RA concentration in the
neural tube.  The profile is adapted from Diez del Corral et al. 2003 (Cell) and
Liu et al. 2020 (Development): exponential decay from dorsal roof plate with
rostral-caudal attenuation factor.

This lightweight solver covers the early human embryo window (3–10 pcw).
It exposes the same public interface as existing `shh_gradient_system` etc. so
foundation-layer integrators pick it up automatically.
"""

from __future__ import annotations

from typing import Tuple
import math

# Calibration constants (µM) – dorsal roof-plate concentration @ 3 pcw
RA_MAX_CONC = 1.0  # µM
# Decay length-scale along DV axis (µm)
DV_LENGTH_SCALE = 60.0
# Rostral-caudal attenuation (unitless per mm)
RC_ATTENUATION = 0.15


class RAGradientSolver:
    """Compute RA concentration at any 3-D point (µm units)."""

    def __init__(self,
                 ra_max: float = RA_MAX_CONC,
                 dv_lambda: float = DV_LENGTH_SCALE,
                 rc_attenuation: float = RC_ATTENUATION):
        self.ra_max = ra_max
        self.dv_lambda = dv_lambda
        self.rc_k = rc_attenuation

    # ------------------------------------------------------------------
    # Public API (mirrors other *gradient_system* modules)
    # ------------------------------------------------------------------
    def concentration(self, pos: Tuple[float, float, float]) -> float:
        """Return RA concentration (µM) at *pos*=(x,y,z).

        Coordinate convention: z = 0 at dorsal roof plate, positive towards
        ventral floor plate; y = 0 at rostral; positive towards caudal.
        """
        x, y, z = pos
        dv_decay = math.exp(-z / self.dv_lambda)
        rc_factor = math.exp(-self.rc_k * (y / 1000.0))  # y in µm → mm
        return self.ra_max * dv_decay * rc_factor

    # Convenience sampler ------------------------------------------------
    def line_profile(self, n_points: int = 50,
                     max_depth: float = 200.0,
                     y_offset: float = 0.0) -> list[float]:
        """Return concentration values along the DV axis (z) for diagnostics."""
        step = max_depth / (n_points - 1)
        return [self.concentration((0.0, y_offset, step * i)) for i in range(n_points)]
