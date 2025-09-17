"""Utility functions and helper classes for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import json
import time
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from .config import LAST_INTRINSIC

def get_run_id() -> str:
    """Generate unique run identifier."""
    return f"run_{int(time.time())}"

def get_path(rel: str, run_id: str) -> str:
    """Generate file path with run ID."""
    from .config import RUNTIME_DIR
    import os
    abs_dir = os.path.join(RUNTIME_DIR, run_id)
    os.makedirs(abs_dir, exist_ok=True)
    return os.path.join(abs_dir, rel)

def mood_get(mood_vector: dict, key: str, default: float = 0.5) -> float:
    """Safely extract mood value with default."""
    return float(mood_vector.get(key, default))

def sanitize_line(text: str, max_chars: int = 80) -> str:
    """Sanitize text to single line."""
    if not text or not text.strip():
        return ""
    clean = " ".join(text.strip().split())
    return clean[:max_chars] + ("..." if len(clean) > max_chars else "")

def sanitize_block(text: str, max_sentences: int = 5, max_chars: int = 500) -> str:
    """Sanitize text block."""
    if not text or not text.strip():
        return ""
    sentences = [s.strip() for s in text.strip().replace('\n', ' ').split('.') if s.strip()]
    truncated = sentences[:max_sentences]
    result = ". ".join(truncated)
    if len(truncated) == max_sentences and len(sentences) > max_sentences:
        result += "..."
    return result[:max_chars]

def safe_json_write(filepath: str, data: Any):
    """Safely write JSON with error handling."""
    try:
        import os
        import tempfile
        import shutil
        dirname = os.path.dirname(filepath)
        os.makedirs(dirname, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dirname) as tmp:
            json.dump(data, tmp, cls=NumpyEncoder, ensure_ascii=False, separators=(',', ':'))
            temp_path = tmp.name

        shutil.move(temp_path, filepath)
    except Exception:
        pass

def safe_json_read(filepath: str, default: Any = None) -> Any:
    """Safely read JSON with fallback to default."""
    try:
        import os
        if not os.path.exists(filepath):
            return default
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def _parse_json_object(text: str) -> Dict:
    """Parse JSON object from text."""
    import re
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}

def classify_geometry_theme(delta_vector: np.ndarray) -> List[str]:
    """Classify geometric themes from delta vector."""
    if np.linalg.norm(delta_vector) < 0.01:
        return ["stasis"]
    return ["integration", "growth"]

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Helper function to ensure vectors have unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

def clamp_action(vec: np.ndarray, max_norm: float = 0.04) -> np.ndarray:
    """Clamp action vector to maximum norm."""
    n = float(np.linalg.norm(vec))
    if n == 0.0 or n <= max_norm:
        return vec
    return (vec * (max_norm / n)).astype(np.float32)

def shaped_reward_components(bh: float, bh_ma50: Optional[float], action: np.ndarray,
                           prev_action: np.ndarray, extras: Dict[str, Any]) -> Dict[str, float]:
    """Calculate shaped reward components."""
    w_grad, w20, w40, w60, w_act, w_smooth = 0.8, 0.02, 0.05, 0.10, 0.5, 0.25
    grad_term = w_grad * max(0.0, bh - (bh_ma50 or 0.0))
    dwell = (w20 if bh > 0.20 else 0.0) + (w40 if bh > 0.40 else 0.0) + (w60 if bh > 0.60 else 0.0)
    force_pen = w_act * (float(np.linalg.norm(action)) ** 2)
    smooth_pen = w_smooth * float(np.sum((action - prev_action) ** 2))
    goal_term = 0.4 * float(extras.get('goal_resonance', 0.0))
    tension_term = 0.1 * float(extras.get('avg_tension', 0.0))
    valence_term = 0.1 * float(extras.get('valence', 0.0))
    surprise_term = 0.4 * float(extras.get('surprise', 0.0))

    try:
        fe = float(extras.get('free_energy', LAST_INTRINSIC.get('free_energy', 0.0)))
        epi = float(extras.get('epistemic', LAST_INTRINSIC.get('epistemic', 0.0)))
        topo = float(extras.get('topo', LAST_INTRINSIC.get('topo', 0.0)))
    except Exception:
        fe = epi = topo = 0.0

    fe_term = 0.2 * fe
    epi_term = 0.3 * epi
    topo_term = 0.3 * topo

    return {
        'grad': grad_term, 'dwell': dwell, 'force_pen': -force_pen, 'smooth_pen': -smooth_pen,
        'goal': goal_term, 'tension': tension_term, 'valence': valence_term, 'surprise': surprise_term,
        'free_energy': fe_term, 'epistemic': epi_term, 'topo': topo_term
    }

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

@dataclass
class EmergenceSeed:
    """Emergence seed for black hole events."""
    remnant_id: str
    embedding_vector: np.ndarray
    projected_vector: np.ndarray
    mass: float
    absorbed_ids: List[str]
    step_created: int

class UniversalEmbeddingAdapter:
    """Universal embedding dimension adapter."""
    def __init__(self, in_dim: int, out_dim: int):
        from .config import GLOBAL_SEED
        self.in_dim, self.out_dim = in_dim, out_dim
        if in_dim == out_dim:
            self.W = np.eye(in_dim, dtype=np.float32)
        else:
            rng = np.random.default_rng(GLOBAL_SEED)
            self.W = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
            self.W /= np.linalg.norm(self.W, axis=0, keepdims=True)

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape[0] != self.in_dim:
            padded_vec = np.zeros(self.in_dim, dtype=np.float32)
            size_to_copy = min(vector.shape[0], self.in_dim)
            padded_vec[:size_to_copy] = vector[:size_to_copy]
            vector = padded_vec
        return vector @ self.W

class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration."""
    def __init__(self, size: int, theta: float = 0.05, sigma: float = 0.06):
        self.size = size
        self.theta = theta
        self._sigma0 = sigma
        self.sigma = sigma
        self.state = np.zeros(self.size, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.size, dtype=np.float32)

    def sample(self) -> np.ndarray:
        dx = self.theta * (-self.state) + self.sigma * np.random.randn(self.size).astype(np.float32)
        self.state = self.state + dx
        return self.state

class Bar:
    """Simple data container."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
