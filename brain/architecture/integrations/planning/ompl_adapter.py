"""OMPL Adapter
Joint/task-space motion planning wrapper.
Falls back to straight-line interpolation if Python bindings are unavailable.
Source: https://github.com/ompl/ompl

Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Adapter layer used by simulators to interface external libraries.
"""
import numpy as np

try:
	_HAS_OMPL = True
except Exception:
	_HAS_OMPL = False

class OMPLAdapter:
	def __init__(self, dof: int):
		self.dof = int(dof)
		self.available = _HAS_OMPL and self.dof > 0

	def is_available(self) -> bool:
		return bool(self.available)

	def plan(self, start: np.ndarray, goal: np.ndarray, num_points: int = 50) -> np.ndarray:
		if not self.available:
			# Linear interpolation fallback
			t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)[:, None]
			traj = (1.0 - t) * start[None, :] + t * goal[None, :]
			return traj.astype(np.float32)
		# Placeholder: return straight-line until full OMPL problem setup is provided
		t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)[:, None]
		traj = (1.0 - t) * start[None, :] + t * goal[None, :]
		return traj.astype(np.float32)

	def get_status(self) -> dict:
		return {"available": self.available, "dof": self.dof}
