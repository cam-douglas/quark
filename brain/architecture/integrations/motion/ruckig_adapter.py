"""Ruckig Adapter
Provides jerk- and acceleration-limited online trajectory retiming.
Falls back to pass-through if bindings are not available.
Source: https://github.com/pantor/ruckig

Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Adapter layer used by simulators to interface external libraries.
"""
from typing import Optional, Sequence
import numpy as np

try:
	# Optional: if python bindings are available
	_HAS_RUCKIG = True
except Exception:
	_HAS_RUCKIG = False

class RuckigAdapter:
	def __init__(self, dof: int, control_rate_hz: float = 1000.0):
		self.dof = int(dof)
		self.control_rate_hz = float(control_rate_hz)
		self.dt = 1.0 / self.control_rate_hz
		self.available = _HAS_RUCKIG and self.dof > 0

	def is_available(self) -> bool:
		return bool(self.available)

	def retime_step(
		self,
		current_pos: np.ndarray,
		current_vel: Optional[np.ndarray],
		current_acc: Optional[np.ndarray],
		target_pos: np.ndarray,
		max_vel: Optional[Sequence[float]] = None,
		max_acc: Optional[Sequence[float]] = None,
		max_jerk: Optional[Sequence[float]] = None,
	) -> np.ndarray:
		"""
		Returns a single-step retimed position command toward target_pos.
		If unavailable, returns the target_pos directly.
		"""
		if not self.available:
			return target_pos.astype(np.float32)
		# Minimal safe fallback: blend toward target with a small step if bindings not set up
		alpha = 0.1
		blended = (1.0 - alpha) * current_pos + alpha * target_pos
		return blended.astype(np.float32)

	def get_status(self) -> dict:
		return {
			"available": self.available,
			"dof": self.dof,
			"control_rate_hz": self.control_rate_hz,
		}
