"""Configuration constants and environment variables for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import os
from typing import Dict, Any

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")

# --- Timeouts and Intervals ---
POOL_WORKER_TIMEOUT = int(os.getenv("POOL_WORKER_TIMEOUT", "20"))
POOL_RESULT_TIMEOUT = int(os.getenv("POOL_RESULT_TIMEOUT", "60"))
LLM_CALL_TIMEOUT_SEC = int(os.getenv("LLM_CALL_TIMEOUT_SEC", "30"))
EMBEDDING_TIMEOUT_SEC = int(os.getenv("EMBEDDING_TIMEOUT_SEC", "15"))
DREAM_MIN_INTERVAL_SEC = 30
CONSOLE_EXPORT_EVERY_STEPS = 100
CONSOLE_EXPORT_FORMAT = "both"  # "text", "json", or "both"

# --- Model Dimensions ---
EMBED_DIM = int(os.getenv("E8_EMBED_DIM", "1536"))
DIMENSIONAL_SHELL_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]
AUTOENCODER_LAYER_SIZES = [EMBED_DIM] + DIMENSIONAL_SHELL_SIZES

# --- Action Layout for RL Agent ---
ACTION_LAYOUT = [
    {"dim": 3, "biv_start": 0, "biv_len": 3, "angle_idx": 3},
    {"dim": 5, "biv_start": 4, "biv_len": 10, "angle_idx": 14},
    {"dim": 8, "biv_start": 15, "biv_len": 28, "angle_idx": 43},
]
ACTION_SIZE_NO_LOCK = sum(d["biv_len"] + 1 for d in ACTION_LAYOUT)

# --- Cognitive Cycle Timing ---
TEACHER_ASK_EVERY = 25
TEACHER_OFFSET = 5
EXPLORER_OFFSET = 15
TEACHER_STEP_TIMEOUT = 15.0
EXPLORER_STEP_TIMEOUT = 20.0

# --- Black Hole Event Parameters ---
BLACK_HOLE_COOLDOWN_STEPS = 50
BH_PRESSURE_THRESHOLD = 0.4
BH_SPREAD_FRAC = 0.5
BH_BG_FRAC = 0.2
BH_DIFFUSION_ETA = 0.15
BH_FIELD_LEAK = 0.02
BLACK_HOLE_K = 16  # Number of KNN links for remnant

# --- Memory and Temperature ---
CONSOLIDATE_MIN = 20
TEMP_HALF_LIFE_VIVID = 8
TEMP_HALF_LIFE_HOT = 24
TEMP_HALF_LIFE_WARM = 72
TEMP_HALF_LIFE_COLD = 240

# --- Miscellaneous ---
SEMANTIC_DOMAIN = "E8_holographic_Conscioussness"
DREAM_MODE_ENABLED = True
LOCAL_GEN_WORKERS = int(os.getenv("LOCAL_GEN_WORKERS", "1"))
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "1337"))

# --- Data Sources ---
DATA_SOURCES: Dict[str, Any] = {}

# --- System Constants ---
LAST_INTRINSIC: Dict[str, Any] = {}
