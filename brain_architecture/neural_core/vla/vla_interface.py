from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np


class VLAInterface:
    """
    Minimal interface contract for a Vision-Language-Action module.
    """

    def infer(self, image: np.ndarray, text_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Produce high-level intent from vision and language.

        Returns a dict, e.g.:
        {
          "target_velocity": [0.0, 0.3, 0.0],
          "gait": {"frequency": 0.6, "amplitude": 0.4}
        }
        """
        raise NotImplementedError


