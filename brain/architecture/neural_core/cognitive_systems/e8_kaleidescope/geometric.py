"""Geometric Algebra and Clifford operations for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import numpy as np
from typing import Optional, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .e8_mind_core import E8Mind

# Optional dependencies
try:
    import clifford
    from clifford.g3 import layout as g3_layout, blades as g3_blades
    CLIFFORD_AVAILABLE = True
except ImportError:
    clifford = None
    g3_layout, g3_blades = None, None
    CLIFFORD_AVAILABLE = False

from .utils import normalize_vector

class CliffordRotorGenerator:
    """
    Generates mathematically precise Geometric Algebra rotors using the Clifford library.
    Includes safety checks for collinear or zero-magnitude vectors.
    """
    def __init__(self, mind_instance: 'E8Mind', layout, blades):
        self.mind = mind_instance
        self.layout = layout
        self.blades = blades
        self.basis_vectors = [self.blades[f'e{i+1}'] for i in range(layout.dims)]

    def _random_unit_bivector(self):
        """Returns a simple, random unit bivector (e.g., e1^e2)."""
        n = len(self.basis_vectors)
        i, j = np.random.choice(np.arange(n), size=2, replace=False)
        B = self.basis_vectors[i] ^ self.basis_vectors[j]
        return B.normal()

    def _select_dynamic_pair(self, shell: 'DimensionalShell') -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Selects a pair of vectors from the shell to define the plane of rotation."""
        nodes = list(shell.vectors.keys())
        if len(nodes) < 2:
            return None

        candidates = []
        for nid in nodes:
            node_data = self.mind.memory.graph_db.get_node(nid)
            if node_data:
                vec_np = shell.get_vector(nid)
                if vec_np is not None and np.linalg.norm(vec_np) > 1e-9:
                    candidates.append({
                        'id': nid,
                        'temp': node_data.get('temperature', 0.1),
                        'vec': vec_np
                    })

        if len(candidates) < 2:
            return None

        candidates.sort(key=lambda x: x['temp'], reverse=True)
        anchor_a = candidates[0]

        best_partner = None
        max_dist = -1.0
        for partner_candidate in candidates[1:min(len(candidates), 15)]:
            dist = 1.0 - abs(np.dot(
                normalize_vector(anchor_a['vec']),
                normalize_vector(partner_candidate['vec'])
            ))
            if dist > max_dist:
                max_dist = dist
                best_partner = partner_candidate

        if best_partner is None:
            return None

        return anchor_a['vec'], best_partner['vec']

    def generate_rotor(self, shell: 'DimensionalShell', angle: float):
        """Generates a rotor that rotates by 'angle' in the plane defined by two vectors."""
        if not CLIFFORD_AVAILABLE:
            return None

        pair = self._select_dynamic_pair(shell)

        if pair is None:
            random_bivector = self._random_unit_bivector()
            return (-(random_bivector) * (angle / 2.0)).exp()

        a_vec, b_vec = pair
        a = sum(val * bv for val, bv in zip(a_vec, self.basis_vectors))
        b = sum(val * bv for val, bv in zip(b_vec, self.basis_vectors))

        a = a.normal()
        b = b.normal()

        B = (a ^ b)

        if abs(B) < 1e-9:
            return (-(self._random_unit_bivector()) * (angle / 2.0)).exp()

        B_normalized = B.normal()
        rotor = (-B_normalized * angle / 2.0).exp()
        return rotor

class DimensionalShell:
    """
    Represents a dimensional space where concepts exist as Geometric Algebra multivectors.
    """
    def __init__(self, dim: int, mind_instance: 'E8Mind'):
        if not CLIFFORD_AVAILABLE:
            raise ImportError("The 'clifford' library is required for DimensionalShell.")

        self.dim = dim
        self.mind = mind_instance
        self.layout, self.blades = clifford.Cl(dim)
        self.vectors: Dict[str, clifford.MultiVector] = {}
        self.basis_vectors = [self.blades[f'e{i+1}'] for i in range(dim)]
        self.rotor_generator = CliffordRotorGenerator(mind_instance, self.layout, self.blades)
        self.orientation = self.layout.scalar

        try:
            self._build_bivector_basis()
        except Exception:
            pass

    def _build_bivector_basis(self):
        """Build bivector basis for the shell."""
        try:
            self.bivector_basis = []
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    self.bivector_basis.append(self.basis_vectors[i] ^ self.basis_vectors[j])
        except Exception:
            self.bivector_basis = []

    def add_vector(self, node_id: str, vector: np.ndarray):
        """Converts a numpy vector to a multivector and adds it to the shell."""
        if vector.shape[0] != self.dim:
            padded_vector = np.zeros(self.dim)
            size_to_copy = min(vector.shape[0], self.dim)
            padded_vector[:size_to_copy] = vector[:size_to_copy]
            vector = padded_vector

        snapped_vector = self.mind._snap_to_lattice(vector, self.dim)
        mv = sum(val * bv for val, bv in zip(snapped_vector, self.basis_vectors))
        self.vectors[node_id] = mv

    def get_vector(self, node_id: str) -> Optional[np.ndarray]:
        """Retrieves a vector as a numpy array for external compatibility."""
        multivector = self.vectors.get(node_id)
        if multivector is None:
            return None
        return np.array([float(multivector[bv]) for bv in self.basis_vectors], dtype=np.float32)

    def spin(self, action_angle: float = 0.1):
        """Applies a concept-driven rotation to the entire shell."""
        if len(self.vectors) < 2:
            return

        incremental_rotor = self.rotor_generator.generate_rotor(self, action_angle)
        self.orientation = (incremental_rotor * self.orientation).normal()
        orientation_reverse = ~self.orientation

        for node_id, mv in self.vectors.items():
            rotated_mv = self.orientation * mv * orientation_reverse
            rotated_vec_np = np.array([float(rotated_mv[bv]) for bv in self.basis_vectors])
            snapped_vec_np = self.mind._snap_to_lattice(rotated_vec_np, self.dim)
            self.vectors[node_id] = sum(val * bv for val, bv in zip(snapped_vec_np, self.basis_vectors))

    def get_all_vectors_as_matrix(self) -> tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Returns all vectors as a single NumPy matrix."""
        if not self.vectors:
            return None, None

        node_ids = list(self.vectors.keys())
        matrix_list = []
        for mv in self.vectors.values():
            matrix_list.append([float(mv[bv]) for bv in self.basis_vectors])

        return np.array(matrix_list, dtype=np.float32), node_ids

    def spin_with_bivector(self, bivector_coeffs: List[float], angle: float):
        """Spin using specific bivector coefficients."""
        try:
            if not hasattr(self, "bivector_basis") or not self.bivector_basis:
                self._build_bivector_basis()
            if len(self.vectors) < 1:
                return

            B = 0
            k = min(len(self.bivector_basis), len(bivector_coeffs))
            for idx in range(k):
                try:
                    B = B + float(bivector_coeffs[idx]) * self.bivector_basis[idx]
                except Exception:
                    pass

            try:
                Bn = B.normal()
            except Exception:
                Bn = None

            if Bn is None:
                try:
                    self.spin(float(angle))
                except Exception:
                    pass
                return

            try:
                R = (-Bn * (float(angle) / 2.0)).exp()
                self.orientation = (R * self.orientation).normal()
                Rrev = ~self.orientation
                new_vecs = {}
                for node_id, mv in self.vectors.items():
                    mv2 = self.orientation * mv * Rrev
                    vec_np = []
                    for bv in self.basis_vectors:
                        try:
                            vec_np.append(float(mv2[bv]))
                        except Exception:
                            vec_np.append(0.0)
                    snapped = self.mind._snap_to_lattice(vec_np, self.dim) if hasattr(self.mind, "_snap_to_lattice") else vec_np
                    try:
                        new_vecs[node_id] = sum(val * bv for val, bv in zip(snapped, self.basis_vectors))
                    except Exception:
                        new_vecs[node_id] = mv
                self.vectors = new_vecs
            except Exception:
                try:
                    self.spin(float(angle))
                except Exception:
                    pass
        except Exception:
            pass
