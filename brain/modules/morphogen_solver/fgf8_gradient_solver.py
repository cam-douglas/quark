"""Fibroblast Growth Factor-8 (FGF8) Gradient Solver

Implements a simplified 3-D steady-state model of FGF8 emanating from the
isthmic organiser (mid-hindbrain boundary).  Based on imaging / in-situ data
from Crossley et al. 1996 Development and forward-modelling by Mathiisen 2021.

Interface matches other *gradient_system* modules so integrators can query the
`concentration()` function directly.
"""

from __future__ import annotations

from typing import Tuple
import math

# Peak concentration at organiser (µM) @ 4 pcw
FGF8_MAX_CONC = 0.8
# Radial decay length (µm) in all directions from organiser source plane
RADIAL_LENGTH_SCALE = 100.0
# Dorsal-ventral bias factor (ventral attenuation)
DV_BIAS = 0.6  # ventral concentration is 60 % of dorsal at same radial dist.


class FGF8GradientSolver:
    """Return FGF8 concentration at an arbitrary 3-D coordinate."""

    def __init__(self,
                 fgf8_max: float = FGF8_MAX_CONC,
                 radial_lambda: float = RADIAL_LENGTH_SCALE,
                 dv_bias: float = DV_BIAS):
        self.fgf8_max = fgf8_max
        self.radial_lambda = radial_lambda
        self.dv_bias = dv_bias

    def concentration(self, pos: Tuple[float, float, float]) -> float:
        """µM concentration at *pos*=(x,y,z).

        Assumes organiser located at (0,0,0).  Radial distance r = sqrt(x²+y²).
        z is dorsal (+) / ventral (–); ventral values scaled by `dv_bias`.
        """
        x, y, z = pos
        r = math.hypot(x, y)
        base = self.fgf8_max * math.exp(-r / self.radial_lambda)
        if z < 0:
            base *= self.dv_bias  # ventral attenuation
        return base
