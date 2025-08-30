"""Package marker so setuptools includes 'brain' top-level package."""
from __future__ import annotations

# Step 2 of 8: feature flag for E8 lattice memory
import os
USE_E8_MEMORY = os.getenv("USE_E8_MEMORY", "false").lower() in {"1", "true", "yes"}
