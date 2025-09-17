"""Proximity engines and attention mechanisms for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .e8_mind_core import E8Mind
    from .geometric import DimensionalShell

# Optional dependencies for efficient nearest neighbors
try:
    from sklearn.neighbors import KDTree as _SKKDTree
    from sklearn.metrics.pairwise import cosine_distances as _sk_cosine_distances
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity
except ImportError:
    _SKKDTree = None
    _sk_cosine_distances = None
    _sk_cosine_similarity = None

try:
    from scipy.spatial import KDTree as _SPKDTree
except ImportError:
    _SPKDTree = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class KDTree:
    """A wrapper for scikit-learn/scipy KDTree with a pure NumPy fallback."""
    def __init__(self, data):
        if _SKKDTree is not None:
            self._impl = _SKKDTree(np.asarray(data, dtype=np.float32))
            self.n = self._impl.data.shape[0]
            self._is_fallback = False
        elif _SPKDTree is not None:
            self._impl = _SPKDTree(np.asarray(data, dtype=np.float32))
            self.n = self._impl.n
            self._is_fallback = False
        else:
            # Fallback initialization
            self._impl = np.asarray(data, dtype=np.float32)
            self.n = self._impl.shape[0]
            self._is_fallback = True

    def query(self, q, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Find k-nearest neighbors for query vector(s)."""
        q_arr = np.asarray(q, dtype=np.float32)
        is_single_query = q_arr.ndim == 1
        q_2d = np.atleast_2d(q_arr)

        if not self._is_fallback:
            # Use efficient scikit-learn or scipy implementation
            d, i = self._impl.query(q_2d, k=k)
        else:
            # NumPy fallback for both single and batch queries
            all_dists = []
            all_indices = []
            data_points = self._impl

            for query_vector in q_2d:
                # Calculate Euclidean distances
                distances = np.sqrt(np.sum((data_points - query_vector)**2, axis=1))

                # Get k smallest distances efficiently
                if k < self.n:
                    nearest_idx = np.argpartition(distances, k)[:k]
                    sorted_partition_indices = np.argsort(distances[nearest_idx])
                    idx = nearest_idx[sorted_partition_indices]
                else:
                    idx = np.argsort(distances)[:k]

                all_indices.append(idx)
                all_dists.append(distances[idx])

            d = np.array(all_dists, dtype=np.float32)
            i = np.array(all_indices, dtype=np.int64)

        # Return results in expected shape
        if is_single_query:
            return d.ravel(), i.ravel()
        else:
            return d, i

def cosine_distances(A, B):
    """Calculate cosine distances between matrices."""
    if _sk_cosine_distances is not None:
        return _sk_cosine_distances(A, B)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    def _norm(x):
        return np.sqrt((x*x).sum(axis=1, keepdims=True)) + 1e-12

    A2, B2 = A / _norm(A), B / _norm(B)
    return 1.0 - (A2 @ B2.T)

def cosine_similarity(A, B):
    """Calculate cosine similarity between matrices."""
    if _sk_cosine_similarity is not None:
        return _sk_cosine_similarity(A, B)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)

    def _norm(x):
        return np.sqrt((x*x).sum(axis=1, keepdims=True)) + 1e-12

    A2, B2 = A / _norm(A), B / _norm(B)
    return A2 @ B2.T

class ProximityEngine:
    """Engine for finding approximate nearest neighbors within and across dimensional shells."""
    def __init__(self, shell_dims: List[int], mind_instance: 'E8Mind', console):
        self.console = console
        self.shell_dims = shell_dims
        self.mind = mind_instance
        self.indices: Dict[int, Optional[KDTree]] = {dim: None for dim in shell_dims}
        self.id_maps: Dict[int, List[str]] = {dim: [] for dim in shell_dims}

    def update_shell_index(self, dim: int, shell: 'DimensionalShell'):
        """Rebuilds the KDTree index for a specific dimensional shell."""
        if dim not in self.indices:
            return

        matrix, node_ids = shell.get_all_vectors_as_matrix()
        if matrix is not None and node_ids and matrix.shape[0] > 0:
            try:
                self.indices[dim] = KDTree(matrix)
                self.id_maps[dim] = node_ids
            except Exception as e:
                self.console.log(f"[ProximityEngine] Failed to build KDTree for dim {dim}: {e}")
                self.indices[dim] = None
                self.id_maps[dim] = []
        else:
            self.indices[dim] = None
            self.id_maps[dim] = []

    def find_similar_in_shell(self, query_vector: np.ndarray, dim: int, k: int = 5) -> List[Tuple[str, float]]:
        """Finds k-nearest neighbors for a query vector within its own shell."""
        kdtree = self.indices.get(dim)
        id_map = self.id_maps.get(dim)
        if kdtree is None or not id_map:
            return []

        num_points = kdtree.n
        if k > num_points:
            k = num_points
        if k == 0:
            return []

        distances, indices = kdtree.query(query_vector, k=k)

        if k == 1 and isinstance(indices, (int, np.integer)):
            return [(id_map[int(indices)], float(distances))]
        return [(id_map[int(i)], float(d)) for d, i in zip(distances, indices)]

    def cross_dimensional_query(self, query_vector: np.ndarray, source_dim: int,
                               target_dim: int, k: int = 1) -> List[Tuple[str, float]]:
        """Finds nearest neighbors for a vector from a source shell in a target shell."""
        if not TORCH_AVAILABLE or self.mind.autoencoder is None:
            return []
        if source_dim == target_dim:
            return self.find_similar_in_shell(query_vector, target_dim, k)

        with torch.no_grad():
            source_tensor = torch.from_numpy(query_vector).float().unsqueeze(0)
            projected_tensor = self.mind.autoencoder.project_between_dim(
                source_tensor, source_dim=source_dim, target_dim=target_dim
            )

            if projected_tensor is None:
                return []

            projected_vector = projected_tensor.squeeze(0).numpy()

        return self.find_similar_in_shell(projected_vector, target_dim, k)

class ShellAttention:
    """Attention mechanism for dimensional shells."""
    def __init__(self, out_dim: int = 32, keep_k: int = 3):
        self.out_dim = int(out_dim)
        self.keep_k = int(max(1, keep_k))

    @staticmethod
    def _ten(vec: np.ndarray) -> np.ndarray:
        """Convert vector to 10-dimensional representation."""
        if vec is None or vec.size == 0:
            return np.zeros(10, dtype=np.float32)
        if vec.size >= 10:
            return vec[:10].astype(np.float32)
        out = np.zeros(10, dtype=np.float32)
        out[:vec.size] = vec.astype(np.float32)
        return out

    def _weights(self, tensions: Dict, mood) -> Dict:
        """Calculate attention weights based on tensions and mood."""
        eps = 1e-6
        coh = float(mood.mood_vector.get("coherence", 0.5)) if hasattr(mood, "mood_vector") else 0.5
        raw = {d: coh / (eps + float(t)) for d, t in tensions.items()}
        if not raw:
            return {}

        xs = np.array(list(raw.values()), dtype=np.float32)
        xs = np.exp(xs - xs.max())
        xs /= (xs.sum() + 1e-12)
        return {d: float(w) for d, w in zip(raw.keys(), xs.tolist())}

    def build(self, mind: 'E8Mind', out_dim: Optional[int] = None,
              keep_k: Optional[int] = None) -> np.ndarray:
        """Build attention vector based on shell tensions."""
        out_dim = int(out_dim or self.out_dim)
        keep_k = int(keep_k or self.keep_k)
        tensions = {}

        for dim, shell in mind.dimensional_shells.items():
            try:
                M, _ = shell.get_all_vectors_as_matrix()
                if M is not None and M.shape[0] > 1:
                    tensions[dim] = float(np.linalg.norm(M - M.mean(0), axis=1).mean())
                else:
                    tensions[dim] = 0.0
            except Exception:
                tensions[dim] = 0.0

        ws = self._weights(tensions, mind.mood)
        top = sorted(ws.items(), key=lambda kv: -kv[1])[:keep_k]
        parts = []

        for dim, w in top:
            try:
                M, _ = mind.dimensional_shells[dim].get_all_vectors_as_matrix()
                v = M.mean(0).astype(np.float32) if (M is not None and M.size > 0) else np.zeros(dim, dtype=np.float32)
            except Exception:
                v = np.zeros(max(1, int(dim)), dtype=np.float32)
            parts.append(self._ten(v) * float(w))

        head = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)
        need = 10 * keep_k
        if head.size < need:
            head = np.pad(head, (0, need - head.size))
        elif head.size > need:
            head = head[:need]

        gten = float(sum(tensions.values()) / len(tensions)) if tensions else 0.0
        coh = float(mind.mood.mood_vector.get("coherence", 0.5))
        out = np.concatenate([head.astype(np.float32), np.array([gten, coh], dtype=np.float32)])

        if out.size < out_dim:
            out = np.pad(out, (0, out_dim - out.size))
        elif out.size > out_dim:
            out = out[:out_dim]

        return out.astype(np.float32)

class ArbiterGate:
    """Decision gate for cognitive processing."""
    def __init__(self):
        self._last_tv = 0.0

    def decide(self, telemetry: Dict, mood_vec: Dict) -> float:
        """Make gating decision based on telemetry and mood."""
        tv = float((telemetry or {}).get("tv", 0.0))
        energy = float((telemetry or {}).get("energy", 0.0))
        norm = float((telemetry or {}).get("norm", 1.0))
        coh = float((mood_vec or {}).get("coherence", 0.5))
        ent = float((mood_vec or {}).get("entropy", 0.5))

        d_tv = tv - self._last_tv
        self._last_tv = tv

        g = (0.5 + 0.35 * (0.3 * coh - 0.7 * max(0.0, d_tv))
             - 0.15 * abs(1.0 - norm) + 0.05 * (0.5 - ent))
        return float(np.clip(g, 0.0, 1.0))
