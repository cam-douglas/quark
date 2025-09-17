"""spatialmath Adapter
Pose/SE3 utilities wrapper.
Source: https://github.com/bdaiinstitute/spatialmath-python

Integration: This module is indirectly invoked by simulators/agents through adapter or tooling calls.
Rationale: Adapter layer used by simulators to interface external libraries.
"""
import numpy as np

class SpatialMathAdapter:
	def __init__(self):
		self.available = False
		try:
			import spatialmath as sm # type: ignore
			self.sm = sm
			self.available = True
		except Exception:
			self.sm = None
			self.available = False

	def is_available(self) -> bool:
		return bool(self.available)

	def se3_from_xyzrpy(self, xyz: np.ndarray, rpy_deg: np.ndarray):
		if not self.available:
			return np.eye(4, dtype=np.float32)
		R = self.sm.base.rpy2r(*(np.deg2rad(rpy_deg)))
		T = self.sm.SE3.Rt(R, xyz)
		return np.array(T.A, dtype=np.float32)

	def get_status(self) -> dict:
		return {"available": self.available}
