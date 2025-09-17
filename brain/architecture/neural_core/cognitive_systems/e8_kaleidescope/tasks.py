"""Task management and novelty scoring for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import re
import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from collections import deque

if TYPE_CHECKING:
    from .memory import MemoryManager
    from .async_infrastructure import AsyncLLMPool

@dataclass
class AutoTask:
    """Automatic task for curriculum learning."""
    id: str
    label: str
    reason: str
    novelty: float
    coherence: float
    status: str = "pending"
    created_step: int = 0

class AutoTaskManager:
    """Manages automatic task generation based on novelty and coherence."""
    def __init__(self, console):
        self.console = console
        self.queue: List[AutoTask] = []

    def maybe_spawn(self, step: int, novelty: float, coherence: float,
                    top_labels: List[str]) -> Optional[AutoTask]:
        """Spawn new task if novelty is high and coherence is low."""
        if novelty >= 1.10 and coherence <= 0.50:
            lid = f"task-{step}-{len(self.queue)+1}"
            label = (top_labels[0] if top_labels else "Consolidate new pattern")
            reason = f"Novelty {novelty:.2f} high, coherence {coherence:.2f} low. Add grounding task."

            t = AutoTask(
                id=lid, label=label, reason=reason,
                novelty=float(novelty), coherence=float(coherence),
                created_step=int(step)
            )
            self.queue.append(t)

            try:
                self.console.log(f"[Curriculum] Spawned: {t.label} · {reason}")
            except Exception:
                pass
            return t
        return None

    def complete_if_related(self, node_label: str) -> float:
        """Complete related tasks and return reward bonus."""
        for t in self.queue:
            if (t.status == "pending" and node_label and
                (node_label.lower() in t.label.lower() or t.label.lower() in node_label.lower())):
                t.status = "done"
                return float(np.clip(0.15*(t.novelty - 0.8) + 0.15*(0.6 - t.coherence), 0.0, 0.5))
        return 0.0

class NoveltyScorer:
    """
    Calculates novelty and coherence scores for new concepts.
    Uses high-dimensional memory space and adaptive normalization.
    """
    def __init__(self, memory_manager: 'MemoryManager', llm_pool: 'AsyncLLMPool', console):
        self.console = console
        self.memory_manager = memory_manager
        self.llm_pool = llm_pool

    def calculate_novelty(self, new_vector: np.ndarray) -> float:
        """Calculate novelty based on distance to nearest neighbor in memory."""
        similar_nodes = self.memory_manager.find_similar_in_main_storage(new_vector, k=1)
        if not similar_nodes:
            return 2.0

        distance_to_nearest = similar_nodes[0][1]
        avg_distance = self.memory_manager.get_average_nearest_neighbor_distance()
        if avg_distance < 1e-6:
            return 2.0

        novelty_score = distance_to_nearest / avg_distance
        return np.clip(novelty_score, 0.0, 2.0)

    async def calculate_coherence(self, new_concept_text: str) -> float:
        """Uses an LLM to rate the coherence of the new concept."""
        if not new_concept_text:
            return 0.0

        prompt = (
            f'On a scale from 0.0 to 1.0, how coherent and meaningful is the following idea? '
            f'A coherent idea is well-formed, logical, and potentially useful. '
            f'Respond with ONLY the numeric score.\n\n'
            f'Idea: "{new_concept_text}"\n\n'
            f'Coherence Score:'
        )
        try:
            response = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=10, temperature=0.1)
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                return np.clip(float(match.group()), 0.0, 1.0)
            return 0.5
        except Exception as e:
            self.console.log(f"[NoveltyScorer] Coherence check failed: {e}")
            return 0.5

class InsightAgent:
    """Agent that generates new concepts and learns from insight-driven rewards."""
    def __init__(self, llm_pool: 'AsyncLLMPool', novelty_scorer: NoveltyScorer, console):
        self.console = console
        self.llm_pool = llm_pool
        self.novelty_scorer = novelty_scorer
        self.reward_history = deque(maxlen=100)

    async def create_hybrid_concept(self, concept_a: Dict, concept_b: Dict) -> str:
        """Uses an LLM to synthesize a new, hybrid concept from two source concepts."""
        prompt = (
            f"You are a creative synthesizer of ideas. Combine the core essence of the following two concepts "
            f"into a single, novel, and coherent hybrid concept. Describe the new idea in one or two sentences.\n\n"
            f"Concept A: '{concept_a.get('metaphor', concept_a.get('label', ''))}'\n"
            f"Concept B: '{concept_b.get('metaphor', concept_b.get('label', ''))}'\n\n"
            f"Hybrid Concept:"
        )
        new_concept_text = await self.llm_pool.enqueue_and_wait(prompt, max_tokens=100, temperature=0.85)
        return new_concept_text.strip()

    def learn_from_reward(self, reward: float):
        """Stores the reward and updates learning metrics."""
        self.reward_history.append(reward)
        if len(self.reward_history) > 10:
            avg_reward = np.mean(self.reward_history)
            self.console.log(f"[InsightAgent] Average Insight Reward: {avg_reward:.3f}")

def export_graph(graph) -> Dict:
    """Export NetworkX graph to JSON-serializable format."""
    if graph is None:
        return {"nodes": [], "edges": []}

    try:
        return {
            "nodes": [{"id": nid, **data} for nid, data in graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data} for u, v, data in graph.edges(data=True)]
        }
    except Exception:
        return {"nodes": [], "edges": []}
