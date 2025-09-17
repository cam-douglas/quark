"""Master Brain Simulator
Orchestrates all brain modules, managing their interactions and the flow of information.
This is the central hub for the integrated brain simulation.

Integration: Primary simulator entry; integrates all neural subsystems at runtime.
Rationale: Primary simulator entry point for neural runtime.
"""

import sys
import os

# Ensure parent directories are in the path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Updated path: default mode network moved under neural_core/default_mode_network

# AlphaGenome Integration for biologically compliant brain construction
# AlphaGenome integrations are optional and can be disabled via env.
try:
    from brain.modules.alphagenome_integration.cell_constructor import CellType
except Exception:
    class _CellTypeFallback:
        class NEURON:
            value = "neuron"
    CellType = _CellTypeFallback
# Adapters (feature-flagged)
# Perception / SLAM / Dynamics / Math adapters
# Optimizers / backends adapters (stubs)
# These provide availability and configuration surfaces; core usage will be wired per-module.
try:
	from brain.architecture.integrations.optim.casadi_adapter import CasadiAdapter
	from brain.architecture.integrations.optim.osqp_adapter import OSQPAdapter
	from brain.architecture.integrations.optim.ceres_adapter import CeresAdapter
	from brain.architecture.integrations.optim.qpsolvers_adapter import QPSolversAdapter
	from brain.architecture.integrations.optim.nlopt_adapter import NLOptAdapter
	from brain.architecture.integrations.optim.ipopt_adapter import IpoptAdapter
	from brain.architecture.integrations.optim.proxsuite_adapter import ProxSuiteAdapter
	from brain.architecture.integrations.optim.pagmo_adapter import PagmoAdapter
	from brain.architecture.integrations.optim.pymoo_adapter import PyMooAdapter
	from brain.architecture.integrations.optim.trajopt_adapter import TrajOptAdapter
except Exception:
	CasadiAdapter = OSQPAdapter = CeresAdapter = QPSolversAdapter = NLOptAdapter = IpoptAdapter = ProxSuiteAdapter = PagmoAdapter = PyMooAdapter = TrajOptAdapter = None

# Motion extra
try:
	from brain.architecture.integrations.motion.topico_adapter import TopiCoAdapter
except Exception:
	TopiCoAdapter = None

# Perception fusion
try:
	from brain.architecture.integrations.perception.fuse_adapter import FuseAdapter
except Exception:
	FuseAdapter = None

from brain.tools.goal_poll import log_next_goal

log_next_goal()
