import math
from brain.modules.morphogen_solver import RAGradientSolver, FGF8GradientSolver

def test_ra_gradient_decay():
    solver = RAGradientSolver()
    dorsal = solver.concentration((0,0,0))
    ventral = solver.concentration((0,0,200))
    assert dorsal > ventral
    assert math.isclose(ventral/dorsal, math.exp(-200/solver.dv_lambda), rel_tol=0.1)

def test_fgf8_radial_decay():
    solver = FGF8GradientSolver()
    centre = solver.concentration((0,0,0))
    off = solver.concentration((100,0,0))
    assert off < centre
    assert off > 0
