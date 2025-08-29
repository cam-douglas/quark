"""
Sophus Adapter
Lie groups utilities placeholder.
Source: https://github.com/strasdat/Sophus
"""
import numpy as np

class SophusAdapter:
	def __init__(self):
		self.available = False
		try:
			import sophuspy as sp # type: ignore
			self.available = True
		except Exception:
			self.available = False

	def is_available(self) -> bool:
		return bool(self.available)

	def se3_log(self, T: np.ndarray) -> np.ndarray:
		return np.zeros(6, dtype=np.float32)

	def get_status(self) -> dict:
		return {"available": self.available}
