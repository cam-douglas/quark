"""
DART Adapter
Dynamics and kinematics placeholder.
Source: https://github.com/dartsim/dart
"""

class DARTAdapter:
	def __init__(self):
		self.available = False
		try:
			import dartpy # type: ignore
			self.available = True
		except Exception:
			self.available = False

	def is_available(self) -> bool:
		return bool(self.available)

	def get_status(self) -> dict:
		return {"available": self.available}
