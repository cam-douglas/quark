"""
Physics Simulation Module for SmallMind

Integrates MuJoCo, NEST, and PyVista for comprehensive brain development simulation:
- MuJoCo: Physical brain development (tissue growth, mechanics)
- NEST: Neural network development (connectivity, activity)
- PyVista: Modern 3D visualization capabilities
- Hybrid: Combined physical and neural simulation
"""

try:
    from ................................................brain_physics import BrainPhysicsSimulator
    from ................................................mujoco_interface import MuJoCoInterface
    from ................................................nest_interface import NESTInterface
    __all__ = ['BrainPhysicsSimulator', 'MuJoCoInterface', 'NESTInterface']
except ImportError as e:
    print(f"Warning: Some physics simulation modules not available: {e}")
    __all__ = []
