"""
Drake Adapter
Rigid-body dynamics, planning tools placeholder.
Source: https://github.com/RobotLocomotion/drake
"""

class DrakeAdapter:
	def __init__(self):
		self.available = False
		try:
			import pydrake # type: ignore
			self.available = True
		except Exception:
			self.available = False

	def is_available(self) -> bool:
		return bool(self.available)

	def get_status(self) -> dict:
		return {"available": self.available}
