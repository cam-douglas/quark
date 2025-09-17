"""Memory management and consolidation for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import asyncio
import glob
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

from .config import EMBED_DIM
from .graph_db import GraphDB
from .proximity import KDTree
from .async_infrastructure import InstrumentedLock
from .utils import safe_json_write

class TinyCompressor:
    """Tiny compressor for memory efficiency."""
    def __init__(self, in_dim: int, code_dim: int = 8):
        self.in_dim = in_dim
        self.code_dim = code_dim
        rng = np.random.default_rng(42)
        self.W = rng.standard_normal((in_dim, code_dim)).astype(np.float32)
        self.W /= np.linalg.norm(self.W, axis=0, keepdims=True)

    def compress(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W

    def approx_decompress(self, z: np.ndarray) -> np.ndarray:
        return z @ self.W.T

class MemoryManager:
    """Core memory management system for E8 Mind."""
    def __init__(self, embedding_fn, mood, subconscious, run_id, probe,
                 llm_caller, mind_instance, **kwargs):
        self.embedding_fn = embedding_fn
        self.mood = mood
        self.subconscious = subconscious
        self.run_id = run_id
        self.probe = probe
        self.llm_caller = llm_caller
        self.mind = mind_instance
        self.lock = InstrumentedLock("memory", probe=self.probe)

        # Core storage
        self.graph_db = GraphDB()
        self.main_vectors: Dict[str, np.ndarray] = {}
        self.main_kdtree: Optional[KDTree] = None
        self._main_storage_ids: List[str] = []
        self._main_storage_matrix = np.empty((0, EMBED_DIM), dtype=np.float32)

        # Indexing
        self.label_to_node_id: Dict[str, str] = {}
        self.autobio_index = defaultdict(list)
        self.symbol_history = defaultdict(list)

        # Core self representation
        self.core_self = np.zeros(EMBED_DIM)
        self.core_self_strength = 0.0

        # Consolidation system
        self.consolidation_buffer: List[Dict] = []
        self.consolidation_task: Optional[asyncio.Task] = None
        self.memory_consolidation_min = int(kwargs.get("memory_consolidation_min", 50))
        self.max_knn_links = int(kwargs.get("max_knn_links", 4))

        # Compression and optimization
        self._compressor = TinyCompressor(in_dim=EMBED_DIM, code_dim=8)
        self.pending_embeddings: List[np.ndarray] = []
        self._repeat_ngrams: Dict[Tuple, int] = defaultdict(int)

        # Field dynamics
        self.field: Dict[str, float] = defaultdict(float)
        self.background_temp = 0.0
        self.active_locks: Dict[Tuple[str, str], int] = {}

    def _path(self, rel: str) -> str:
        """Get path for memory files."""
        from .utils import get_path
        return get_path(rel, self.run_id)

    def apply_soft_link(self, id_a: str, id_b: str, weight_delta: float = 0.05,
                       decay: float = 0.985):
        """Apply soft link between nodes."""
        try:
            if not self.graph_db.graph.has_edge(id_a, id_b):
                self.graph_db.graph.add_edge(id_a, id_b, weight=float(weight_delta), type="proximity")
            else:
                w = float(self.graph_db.graph[id_a][id_b].get("weight", 0.0))
                self.graph_db.graph[id_a][id_b]["weight"] = w * float(decay) + float(weight_delta)

            if hasattr(self, "spike_temperature"):
                self.spike_temperature(id_a, 0.02)
                self.spike_temperature(id_b, 0.02)
        except Exception:
            pass

    def apply_gravitational_lock(self, node_id_a: str, node_id_b: str, duration_steps: int):
        """Apply gravitational lock between nodes."""
        if not self.graph_db.get_node(node_id_a) or not self.graph_db.get_node(node_id_b):
            return

        expiry_step = self.mind.step_num + duration_steps
        edge_tuple = tuple(sorted((node_id_a, node_id_b)))
        self.graph_db.add_edge(node_id_a, node_id_b, type="gravitational_lock", weight=5.0)
        self.active_locks[edge_tuple] = expiry_step

    def decay_locks(self):
        """Decay expired gravitational locks."""
        current_step = self.mind.step_num
        expired_locks = [edge for edge, expiry in self.active_locks.items() if current_step >= expiry]

        for edge in expired_locks:
            node_a, node_b = edge
            if self.graph_db.graph.has_edge(node_a, node_b):
                edge_data = self.graph_db.graph.get_edge_data(node_a, node_b)
                if edge_data and edge_data.get("type") == "gravitational_lock":
                    self.graph_db.graph.remove_edge(node_a, node_b)
            del self.active_locks[edge]

    def neighbors(self, nid: str) -> List[str]:
        """Get neighbors of a node."""
        return self.graph_db.get_neighbors(nid)

    def find_latest_node_at_blueprint(self, blueprint_index: int) -> Optional[str]:
        """Find latest node at specific blueprint location."""
        for node_id, data in reversed(list(self.graph_db.graph.nodes(data=True))):
            if data.get("blueprint_location_id") == blueprint_index:
                return node_id
        return None

    def node_vec(self, nid: str) -> Optional[np.ndarray]:
        """Get normalized vector for node."""
        v = self.main_vectors.get(nid)
        if v is None:
            return None
        n = float(np.linalg.norm(v)) + 1e-12
        return (v / n).astype(np.float32)

    def _rebuild_main_kdtree(self):
        """Rebuild main KDTree index."""
        if self.main_vectors:
            self._main_storage_ids = list(self.main_vectors.keys())
            self._main_storage_matrix = np.array([
                self.main_vectors[nid] for nid in self._main_storage_ids
            ], dtype=np.float32)
            if self._main_storage_matrix.shape[0] > 0:
                self.main_kdtree = KDTree(self._main_storage_matrix)
            else:
                self.main_kdtree = None
        else:
            self._main_storage_ids = []
            self._main_storage_matrix = np.empty((0, EMBED_DIM), dtype=np.float32)
            self.main_kdtree = None

    def find_similar_in_main_storage(self, query_vector: np.ndarray,
                                   k: int = 5) -> List[Tuple[str, float]]:
        """Find similar vectors in main storage."""
        if self.main_kdtree is None or not self._main_storage_ids:
            return []

        k = min(k, len(self._main_storage_ids))
        if k == 0:
            return []

        distances, indices = self.main_kdtree.query(query_vector, k=k)
        node_ids = self._main_storage_ids

        if isinstance(indices, (int, np.integer)):
            return [(node_ids[indices], float(distances))]

        return [(node_ids[i], d) for d, i in zip(np.atleast_1d(distances), np.atleast_1d(indices))]

    def get_average_nearest_neighbor_distance(self, sample_k: int = 256) -> float:
        """Get average nearest neighbor distance for novelty calculation."""
        if self.main_kdtree is None:
            return 1.0

        n = len(self._main_storage_ids)
        if n < 2:
            return 1.0

        rng = np.random.default_rng()
        idxs = rng.choice(n, size=min(sample_k, n), replace=False)
        pts = self._main_storage_matrix[idxs]
        dists, _ = self.main_kdtree.query(pts, k=2)
        nn_dists = dists[:, 1]  # Second column is nearest neighbor distance
        mean_dist = float(np.mean(nn_dists))
        return mean_dist if np.isfinite(mean_dist) and mean_dist > 1e-9 else 1.0

    async def consolidate_memory(self):
        """Perform memory consolidation."""
        if len(self.consolidation_buffer) < self.memory_consolidation_min:
            return

        async with self.lock:
            buffer = self.consolidation_buffer.copy()
            self.consolidation_buffer.clear()

        self.mind.console.log(f"[Memory] Consolidating {len(buffer)} memories... (Feature stub)")

    async def snapshot(self):
        """Create memory snapshot."""
        async with self.lock:
            filepath = self._path(f"snapshot_step_{self.mind.step_num:06d}.json")
            # Export graph data
            try:
                graph_data = {
                    "nodes": [{"id": nid, **data} for nid, data in self.graph_db.graph.nodes(data=True)],
                    "edges": [{"source": u, "target": v, **data} for u, v, data in self.graph_db.graph.edges(data=True)]
                }
            except Exception:
                graph_data = {"nodes": [], "edges": []}
            self._rebuild_main_kdtree()

            main_vectors_serializable = {
                nid: vec.tolist() for nid, vec in self.main_vectors.items()
            }

            snapshot_data = {
                "graph": graph_data,
                "main_vectors": main_vectors_serializable,
                "step": self.mind.step_num,
                "mood": self.mind.mood.mood_vector,
                "subconscious_narrative": self.mind.subconscious.narrative,
            }

        safe_json_write(filepath, snapshot_data)

        # Clean up old snapshots
        all_snapshots = sorted(glob.glob(self._path("snapshot_step_*.json")), key=os.path.getmtime)
        while len(all_snapshots) > 10:
            os.remove(all_snapshots.pop(0))
