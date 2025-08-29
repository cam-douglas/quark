"""
manif Adapter
Lie groups utilities placeholder.
Source: https://github.com/artivis/manif
"""
import numpy as np

class ManifAdapter:
	def __init__(self):
		self.available = False
		try:
			import manif # type: ignore
			self.available = True
		except Exception:
			self.available = False

	def is_available(self) -> bool:
		return bool(self.available)

	def se3_exp(self, xi: np.ndarray) -> np.ndarray:
		return np.eye(4, dtype=np.float32)

	def get_status(self) -> dict:
		return {"available": self.available}
