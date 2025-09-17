"""Core E8Mind orchestration class for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import asyncio
import os
from typing import Dict, List, Optional, Any
from collections import deque

import numpy as np

from .config import (DIMENSIONAL_SHELL_SIZES, ACTION_SIZE_NO_LOCK, RUNTIME_DIR, AUTOENCODER_LAYER_SIZES)
from .geometric import DimensionalShell
from .engines import MoodEngine, SubconsciousLayer, DreamEngine
from .memory import MemoryManager
from .tasks import NoveltyScorer, InsightAgent, AutoTaskManager
from .proximity import ProximityEngine, ShellAttention, ArbiterGate
from .utils import get_path
from .async_infrastructure import Probe, set_asyncio_exception_logger

# Optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .agents import SACMPOAgent

try:
    from .profiles.loader import load_profile
except ImportError:
    def load_profile(name):
        class _FallbackSem:
            def persona_prefix(self, mood): return "You are in a balanced state."
            def pre_embed(self, t): return t
            def post_embed(self, v): return v
            def rerank(self, c): return c
        class _FallbackPrompts:
            def render(self, key, **vars):
                return vars.get("question", "")
        return _FallbackSem(), _FallbackPrompts()

# Placeholder classes for components not yet modularized
class E8Physics:
    """Placeholder for E8 physics system."""
    def __init__(self, console):
        self.console = console
        self.roots = np.random.standard_normal((240, 8)).astype(np.float32)
        self.roots_unit = self.roots / (np.linalg.norm(self.roots, axis=1, keepdims=True) + 1e-12)
        self.weights = np.random.random((240, 240)).astype(np.float32)
        self.weights = (self.weights + self.weights.T) / 2  # Make symmetric

class VariationalAutoencoder:
    """Placeholder autoencoder for dimensional projections."""
    def __init__(self, layer_sizes=None, console=None):
        self.is_trained = False
        self.console = console

    def project_to_dim(self, tensor, target_dim):
        """Placeholder projection."""
        if not TORCH_AVAILABLE:
            return None
        batch_size = tensor.shape[0]
        return torch.randn(batch_size, target_dim)

    def project_between_dim(self, tensor, source_dim, target_dim):
        """Placeholder projection between dimensions."""
        if not TORCH_AVAILABLE:
            return None
        return self.project_to_dim(tensor, target_dim)

class E8Mind:
    """Central orchestration class for E8 consciousness system."""

    def __init__(self, semantic_domain_val: str, run_id: str, llm_client_instance,
                 client_model: str, embedding_model_name: str, embed_adapter,
                 embed_in_dim: int, console):
        self.console = console
        self.console.rule(f"[bold cyan]Initializing E8 Mind | Run ID: {run_id}[/]")
        self.run_id = run_id
        os.makedirs(os.path.join(RUNTIME_DIR, self.run_id), exist_ok=True)

        # Core attributes
        self.step_num = 0
        self.semantic_domain = semantic_domain_val
        self.client_model = client_model
        self.embedding_model = embedding_model_name
        self.embed_adapter = embed_adapter
        self.embed_in_dim = embed_in_dim

        # Initialize probe and exception handling
        self.probe = Probe(run_id)
        set_asyncio_exception_logger(self.probe)

        # LLM and embedding setup
        self.llm_client = llm_client_instance
        self.local_llm_client: Optional[Any] = None
        self.local_llm_model = 'phi3:mini-4k'

        # Anti-repetition system
        self._recent_texts = deque(maxlen=500)
        self._recent_norms = deque(maxlen=500)
        self._anti_repeat_enabled = True

        # Load semantic profile
        try:
            profile_name = os.getenv("MIND_PROFILE", "default")
            self.semantics, self.prompts = load_profile(profile_name)
            self.semantic_domain = getattr(self.semantics, "base_domain", self.semantic_domain)
            self.console.log(f"[INIT] Loaded profile: {getattr(self.semantics, 'name', profile_name)}")
        except Exception as e:
            self.console.log(f"[yellow]Profile load failed: {e}. Using defaults.[/yellow]")
            self.semantics, self.prompts = load_profile("default")

        # Async locks
        self.console_lock = asyncio.Lock()
        self.insight_cycle_lock = asyncio.Lock()
        self._dream_lock = asyncio.Lock()
        self.teacher_explorer_lock = asyncio.Lock()
        self._teacher_question_context_ids: List[str] = []

        # Initialize physics and geometric foundations
        self.console.log("[INIT] Building E8 Physics and Geometric Foundations...")
        self.physics = E8Physics(self.console)

        # Initialize autoencoder if torch available
        if TORCH_AVAILABLE:
            self.autoencoder = VariationalAutoencoder(
                layer_sizes=AUTOENCODER_LAYER_SIZES, console=self.console
            )
        else:
            self.autoencoder = VariationalAutoencoder(console=self.console)
            self.console.log("[yellow]PyTorch not found. Autoencoder disabled.[/yellow]")

        # Initialize cognitive architecture
        self.console.log("[INIT] Assembling Cognitive Architecture...")
        self.mood = MoodEngine(self.console)
        self.subconscious = SubconsciousLayer(self.get_embedding, None, self.console)

        # Initialize dimensional shells
        self.dimensional_shells = {
            dim: DimensionalShell(dim, self) for dim in DIMENSIONAL_SHELL_SIZES
        }

        # Initialize processing engines
        self.proximity_engine = ProximityEngine(
            shell_dims=DIMENSIONAL_SHELL_SIZES, mind_instance=self, console=self.console
        )

        # Initialize memory and learning systems
        self.memory = MemoryManager(
            self.get_embedding, self.mood, self.subconscious,
            self.run_id, self.probe, None, self
        )
        self.novelty_scorer = NoveltyScorer(self.memory, None, self.console)
        self.insight_agent = InsightAgent(None, self.novelty_scorer, self.console)

        # Initialize attention and gating
        self.shell_attention = ShellAttention(out_dim=32, keep_k=3)
        self.arbiter_gate = ArbiterGate()
        self.curriculum = AutoTaskManager(self.console)

        # Initialize dream engine
        self.dream_engine = DreamEngine(self.memory, self)

        # Initialize RL agent if torch available
        self.state_dim = len(self.mood.mood_vector) + 4 + self.shell_attention.out_dim + 5
        self.action_dim = ACTION_SIZE_NO_LOCK
        self.max_action = 0.1

        self._bh_window = deque(maxlen=50)
        self._bh_recent = deque(maxlen=100)
        self._bh_ma50 = 0.0
        self._prev_bh = 0.0
        self._low_bh_streak = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)

        if TORCH_AVAILABLE:
            self.agent = SACMPOAgent(
                self.state_dim, self.action_dim, self.max_action,
                console=self.console, tau=0.002, use_per=True
            )
        else:
            self.agent = None
            self.console.log("[yellow]PyTorch not found. RL agent disabled.[/yellow]")

        self.console.rule("[bold green]E8 Mind Initialization Complete[/]")

    def _path(self, rel: str) -> str:
        """Get path for mind files."""
        return get_path(rel, self.run_id)

    def _snap_to_lattice(self, vector: np.ndarray, dim: int) -> np.ndarray:
        """Snap vector to E8 lattice points."""
        if dim not in self.shell_lattices:
            return np.array(vector, dtype=np.float32)

        try:
            lattice = self.shell_lattices[dim]
            kdtree = self.shell_kdtree_indices[dim]
            distances, indices = kdtree.query(np.array([vector]), k=1)
            closest_lattice_point = lattice[indices[0]]
            return closest_lattice_point.astype(np.float32)
        except Exception:
            return np.array(vector, dtype=np.float32)

    async def get_embedding(self, text: str) -> List[float]:
        """Get text embedding using configured model."""
        try:
            if hasattr(self.llm_client, 'embedding'):
                raw_embedding = await self.llm_client.embedding(text, model=self.embedding_model)
                if self.semantics and hasattr(self.semantics, 'post_embed'):
                    processed_embedding = self.semantics.post_embed(np.array(raw_embedding))
                    final_embedding = self.embed_adapter(processed_embedding)
                else:
                    final_embedding = self.embed_adapter(np.array(raw_embedding))
                return final_embedding.tolist()
            else:
                # Fallback to random embedding
                v = np.random.standard_normal(self.embed_in_dim).astype(np.float32)
                v = v / (np.linalg.norm(v) + 1e-12)
                return self.embed_adapter(v).tolist()
        except Exception as e:
            self.console.log(f"[Embedding] Error: {e}")
            v = np.random.standard_normal(self.embed_in_dim).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
            return self.embed_adapter(v).tolist()

    async def rate_concept(self, text: str) -> float:
        """Rate concept coherence using LLM."""
        try:
            prompt = (f"Rate this concept's coherence from 0.0 to 1.0: '{text}'. "
                     f"Respond with only the number.")

            # Placeholder - would use actual LLM pool
            return np.random.uniform(0.3, 0.9)
        except Exception:
            return 0.5

    def get_state_vector(self) -> np.ndarray:
        """Get current state vector for RL agent."""
        mood_vec = np.array(list(self.mood.mood_vector.values()), dtype=np.float32)

        # Add goal and dynamics components (placeholder)
        goal_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        # Add attention vector
        attention_vec = self.shell_attention.build(self)

        # Add dynamics components
        dynamics_vec = np.array([
            self._bh_ma50, self._prev_bh,
            float(len(self._bh_window)),
            float(self._low_bh_streak),
            self.mood.get_entropy_level()
        ], dtype=np.float32)

        state = np.concatenate([mood_vec, goal_vec, attention_vec, dynamics_vec])
        return state

    async def step(self) -> Dict[str, Any]:
        """Execute one cognitive step."""
        self.step_num += 1

        # Update mood
        self.mood.update()

        # Update memory
        await self.memory.consolidate_memory()

        # Dream step
        if self.step_num % 100 == 0:  # Dream every 100 steps
            dream_result = await self.dream_engine.dream_step()
            if dream_result:
                self.console.log(f"[Dream] {dream_result['content']}")

        # Return telemetry
        return {
            "step": self.step_num,
            "mood": self.mood.mood_vector.copy(),
            "memory_nodes": len(self.memory.graph_db.graph.nodes()),
            "narrative": self.subconscious.narrative
        }

    async def run_cognitive_cycle(self):
        """Main cognitive processing loop."""
        self.console.log("[bold green]Starting E8 Mind Cognitive Cycle[/bold green]")

        try:
            while True:
                telemetry = await self.step()

                # Log progress periodically
                if self.step_num % 100 == 0:
                    self.console.log(f"Step {self.step_num}: {len(self.memory.graph_db.graph.nodes())} concepts")

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)

        except Exception as e:
            self.console.log(f"[bold red]Cognitive cycle error: {e}[/bold red]")
            raise

# Placeholder classes for components not yet fully modularized
class QuantumConfig:
    """Configuration for quantum engine."""
    def __init__(self, seed: int = 1337):
        self.seed = seed

class ClassicalConfig:
    """Configuration for classical engine."""
    def __init__(self, seed: int = 1337):
        self.seed = seed

class E8BoundaryFabric:
    """Placeholder for E8 boundary fabric."""
    def __init__(self, physics):
        self.physics = physics
        self.N = physics.roots.shape[0] if hasattr(physics, 'roots') else 240

    def layout_2d(self):
        """Placeholder layout method."""
        pass

    def to_json(self) -> Dict:
        """Placeholder JSON export."""
        return {"nodes": [], "edges": []}

class GoalField:
    """Placeholder goal field system."""
    def __init__(self, embedding_fn, console):
        self.embedding_fn = embedding_fn
        self.console = console

class DriveSystem:
    """Placeholder drive system."""
    def __init__(self):
        pass

class AsyncLLMPool:
    """Placeholder LLM pool."""
    def __init__(self, mind, worker_count: int = 4):
        self.mind = mind
        self.worker_count = worker_count

    async def enqueue_and_wait(self, prompt: str, max_tokens: int = 100,
                             temperature: float = 0.7) -> str:
        """Placeholder LLM call."""
        return f"[LLM Response to: {prompt[:50]}...]"
