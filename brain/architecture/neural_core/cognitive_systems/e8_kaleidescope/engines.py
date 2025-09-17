"""Core processing engines for E8 Mind System.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.
"""
import json
import random
import zlib
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import numpy as np

from .utils import mood_get, normalize_vector, sanitize_line

class MoodEngine:
    """Affective processing engine for mood dynamics."""
    def __init__(self, console, baseline: float = 0.5, decay_rate: float = 0.995):
        self.console = console
        self.baseline = baseline
        self.decay_rate = decay_rate
        self.mood_vector = {
            "intensity": 0.5, "entropy": 0.5, "coherence": 0.5,
            "positivity": 0.5, "fluidity": 0.5, "intelligibility": 0.5
        }
        self.event_queue = deque()
        self._wx_last_code = None
        self._wx_repeat = 0
        self.console.log("🌦️  Affective WeatherEngine Initialized.")

    def _nudge(self, key: str, amount: float):
        """Nudge mood component by amount."""
        if key in self.mood_vector:
            self.mood_vector[key] = np.clip(self.mood_vector[key] + amount, 0.0, 1.0)

    def process_event(self, event_type: str, **kwargs):
        """Queue mood event for processing."""
        self.event_queue.append((event_type, kwargs))

    def update(self):
        """Update mood state based on queued events."""
        while self.event_queue:
            event_type, kwargs = self.event_queue.popleft()

            if event_type == "movement":
                mag = kwargs.get("magnitude", 0.0)
                self._nudge("intensity", 0.05 * min(mag, 5.0))
                themes = kwargs.get("themes", [])
                if any(t in themes for t in ["disorder", "burst"]):
                    self._nudge("entropy", 0.15)
                    self._nudge("coherence", -0.10)
                if any(t in themes for t in ["integration", "stasis"]):
                    self._nudge("coherence", 0.10)
                    self._nudge("entropy", -0.05)
                if "growth" in themes:
                    self._nudge("fluidity", 0.08)

            elif event_type == "new_concept":
                rating = kwargs.get("rating", 0.5)
                if rating > 0.75:
                    self._nudge("coherence", 0.05 * rating)
                    self._nudge("positivity", 0.10 * rating)
                    self._nudge("intelligibility", 0.06 * rating)
                else:
                    self._nudge("entropy", 0.05 * (1.0 - rating))

            elif event_type == "dream":
                self._nudge("entropy", 0.30)
                self._nudge("fluidity", 0.25)
                self._nudge("coherence", -0.15)
                self._nudge("intensity", 0.10)

            elif event_type == "reflection":
                self._nudge("coherence", 0.20)
                self._nudge("entropy", -0.10)
                self._nudge("positivity", 0.05)
                self._nudge("intelligibility", 0.08)

            elif event_type == "blackhole":
                m = float(kwargs.get("magnitude", 0.0))
                self._nudge("entropy", 0.25 + 0.10 * min(m, 5.0))
                self._nudge("intensity", 0.20)
                self._nudge("coherence", -0.15)

            elif event_type == "insight":
                r = float(kwargs.get("reward", 0.0))
                self._nudge("coherence", 0.12 * r)
                self._nudge("positivity", 0.05 * r)
                self._nudge("entropy", -0.04 * r)

        # Apply decay toward baseline
        for k, v in self.mood_vector.items():
            self.mood_vector[k] = v * self.decay_rate + self.baseline * (1.0 - self.decay_rate)

    def describe(self) -> str:
        """Describe current mood state."""
        high = sorted(self.mood_vector.items(), key=lambda x: -x[1])
        low = sorted(self.mood_vector.items(), key=lambda x: x[1])
        return (f"The mind feels predominantly {high[0][0]}, with undertones of {high[1][0]}. "
                f"The least active state is {low[0][0]}.")

    def get_entropy_level(self) -> float:
        """Get current entropy level."""
        return mood_get(self.mood_vector, "entropy")

    def get_llm_persona_prefix(self) -> str:
        """Get LLM persona based on current mood."""
        i = mood_get(self.mood_vector, 'intensity', 0.5)
        e = mood_get(self.mood_vector, 'entropy', 0.5)
        c = mood_get(self.mood_vector, 'coherence', 0.5)

        if e > 0.7 and i > 0.6:
            return ("You are feeling chaotic, fragmented, and electric. "
                   "Your response should be surreal and full of unexpected connections.")
        elif c > 0.75:
            return ("You are feeling exceptionally clear, logical, and focused. "
                   "Your response should be precise and structured.")
        elif i < 0.3:
            return ("You are feeling calm, quiet, and introspective. "
                   "Your response should be gentle and thoughtful.")
        else:
            return "You are in a balanced state of mind. Your response should be clear and considered."

    def get_symbolic_weather(self) -> str:
        """Get symbolic weather representation."""
        e = mood_get(self.mood_vector, "entropy")
        i = mood_get(self.mood_vector, "intensity")
        c = mood_get(self.mood_vector, "coherence")

        def bin_with_hysteresis(value, thresholds, last_bin):
            padding = 0.05
            current_bin = sum(value > t for t in thresholds)
            if last_bin is not None:
                if current_bin != last_bin:
                    if current_bin > last_bin:
                        if value < thresholds[last_bin] + padding:
                            return last_bin
                    else:
                        if value > thresholds[current_bin] - padding:
                            return last_bin
            return current_bin

        b_e = bin_with_hysteresis(e, (0.25, 0.5, 0.75), getattr(self, "_b_e", None))
        b_i = bin_with_hysteresis(i, (0.25, 0.5, 0.75), getattr(self, "_b_i", None))
        b_c = bin_with_hysteresis(c, (0.25, 0.5, 0.75), getattr(self, "_b_c", None))

        self._b_e, self._b_i, self._b_c = b_e, b_i, b_c
        code = (b_e << 4) | (b_i << 2) | b_c

        if code == self._wx_last_code:
            self._wx_repeat += 1
        else:
            self._wx_repeat, self._wx_last_code = 0, code

        variants = {
            "storm": ["Volatile, sharp swings.", "Choppy, energy spikes.", "Jittery air, quick flips."],
            "calm":  ["Calm, steady drift.", "Gentle, small ripples.", "Soft, even flow."],
            "flow":  ["In-flow, coherent.", "Rolling, smooth arcs.", "Aligned, easy motion."],
            "turbulent": ["Turbulent, scattered.", "Noisy, low signal.", "Foggy, fragmented."],
        }

        if b_i >= 2 and b_e >= 2 and b_c <= 1:
            bucket = "storm"
        elif b_c >= 2 and b_e <= 1:
            bucket = "flow"
        elif b_e <= 1 and b_i <= 1:
            bucket = "calm"
        else:
            bucket = "turbulent"

        idx = (self._wx_repeat // 8) % len(variants[bucket])
        return variants[bucket][idx]

    def get_mood_modulation_vector(self, dim: int) -> np.ndarray:
        """Get mood-modulated vector for dimensional influence."""
        seed = zlib.adler32(json.dumps(self.mood_vector, sort_keys=True).encode())
        rng = np.random.default_rng(seed)
        coherence = mood_get(self.mood_vector, 'coherence', 0.5)
        entropy = mood_get(self.mood_vector, 'entropy', 0.5)

        modulation = rng.standard_normal(dim).astype(np.float32)
        modulation *= (1.0 + 0.5 * (coherence - 0.5))
        modulation += rng.standard_normal(dim).astype(np.float32) * 0.2 * entropy
        return normalize_vector(modulation)

class SubconsciousLayer:
    """Subconscious processing layer for bias and narrative generation."""
    def __init__(self, embedding_fn, llm_caller, console,
                 decay_rate: float = 0.95, accumulation_rate: float = 0.004):
        self.embedding_fn = embedding_fn
        self.llm_caller = llm_caller
        self.console = console
        self.decay_rate = decay_rate
        self.accumulation_rate = accumulation_rate
        self.bias_vector: Optional[np.ndarray] = None
        self.narrative = "The mind is nascent, a canvas awaiting its first impression."
        self.bias_history = deque(maxlen=200)
        self.influences: List[Dict[str, Any]] = []

    async def add_influence(self, text: str, weight: float = 1.0):
        """Add textual influence to subconscious bias."""
        try:
            embedding = await self.embedding_fn(text)
            embedding_np = np.array(embedding, dtype=np.float32)
            embedding_np = normalize_vector(embedding_np)

            if self.bias_vector is None:
                self.bias_vector = embedding_np * weight * self.accumulation_rate
            else:
                self.bias_vector = (self.bias_vector * self.decay_rate +
                                  embedding_np * weight * self.accumulation_rate)
                self.bias_vector = normalize_vector(self.bias_vector)

            self.bias_history.append(embedding_np)
            self.influences.append({
                "text": text[:100],
                "weight": weight,
                "timestamp": len(self.bias_history)
            })

        except Exception as e:
            self.console.log(f"[Subconscious] Error adding influence: {e}")

    def get_bias_modulation(self, base_vector: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Apply subconscious bias to vector."""
        if self.bias_vector is None or base_vector is None:
            return base_vector

        modulated = base_vector + self.bias_vector * strength
        return normalize_vector(modulated)

    async def update_narrative(self, context: str):
        """Update internal narrative based on context."""
        try:
            prompt = (f"Given this context: '{context}', update the internal narrative. "
                     f"Current narrative: '{self.narrative}'. "
                     f"Generate a brief, evolving narrative (2-3 sentences max).")

            new_narrative = await self.llm_caller.enqueue_and_wait(
                prompt, max_tokens=80, temperature=0.7
            )
            if new_narrative and not new_narrative.startswith("[LLM"):
                self.narrative = sanitize_line(new_narrative, max_chars=200)
        except Exception as e:
            self.console.log(f"[Subconscious] Error updating narrative: {e}")

class DreamEngine:
    """Generates synthetic memories through thought experiments."""
    ALLOWED_TYPES = (
        "explorer_insight", "insight_synthesis", "meta_reflection", "phase_summary",
        "concept", "external_concept", "mutation", "synthetic_memory",
        "self_code", "self_code_section"
    )

    def __init__(self, memory, mind_instance):
        self.memory = memory
        self.mind = mind_instance
        self.console = mind_instance.console

    def _eligible_concepts(self) -> List[Tuple[str, Dict]]:
        """Get concepts eligible for dream synthesis."""
        G = self.memory.graph_db.graph
        out = []
        for nid, d in G.nodes(data=True):
            if d.get("folded"):
                continue
            if d.get("type") not in self.ALLOWED_TYPES:
                continue
            if self.memory.main_vectors.get(nid) is None:
                continue
            out.append((nid, d))
        return out

    def _pick_from_tension(self, elig: List[Tuple[str, Dict]], k: int = 1) -> List[Tuple[str, Dict]]:
        """Pick concepts based on shell tension."""
        if not elig:
            return []

        tension_candidates = sorted(elig, key=lambda item: item[1].get('shell_tension', 0.0), reverse=True)
        high_tension_seeds = [item for item in tension_candidates if item[1].get('shell_tension', 0.0) > 0.1]

        if high_tension_seeds:
            return high_tension_seeds[:k]
        else:
            return self._pick_neutral(elig, k)

    def _pick_neutral(self, elig: List[Tuple[str, Dict]], k: int = 1) -> List[Tuple[str, Dict]]:
        """Pick random concepts when no high tension found."""
        if not elig:
            return []
        return random.sample(elig, min(k, len(elig)))

    async def dream_step(self) -> Optional[Dict]:
        """Execute a single dream step."""
        if not hasattr(self.mind, 'mood') or not hasattr(self.mind, 'llm_pool'):
            return None

        elig = self._eligible_concepts()
        if len(elig) < 2:
            return None

        seeds = self._pick_from_tension(elig, k=2)
        if len(seeds) < 2:
            seeds = self._pick_neutral(elig, k=2)

        if len(seeds) < 2:
            return None

        concept_a, concept_b = seeds[0][1], seeds[1][1]

        # Generate dream synthesis
        prompt = (
            f"Synthesize a novel insight by combining these concepts in an unexpected way:\n"
            f"A: {concept_a.get('metaphor', concept_a.get('label', ''))}\n"
            f"B: {concept_b.get('metaphor', concept_b.get('label', ''))}\n"
            f"Create a brief, imaginative synthesis (1-2 sentences):"
        )

        try:
            dream_text = await self.mind.llm_pool.enqueue_and_wait(
                prompt, max_tokens=100, temperature=0.9
            )
            if dream_text and not dream_text.startswith("[LLM"):
                return {
                    "type": "dream_synthesis",
                    "content": dream_text.strip(),
                    "source_concepts": [seeds[0][0], seeds[1][0]],
                    "step": self.mind.step_num
                }
        except Exception as e:
            self.console.log(f"[Dream] Synthesis failed: {e}")

        return None

class ClassicalEngine:
    """Classical physics simulation engine."""
    def __init__(self, physics, config, console):
        self.console = console
        self.physics = physics
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.console.log("[INIT] Classical Engine online.")

    def next_index(self, prev_idx: int, sensor8: np.ndarray) -> int:
        """Calculate next index based on physics and sensor input."""
        nbrs = np.where(self.physics.weights[prev_idx] > 0)[0]
        if nbrs.size > 0:
            if np.linalg.norm(sensor8) > 0:
                scores = self.physics.roots[nbrs] @ sensor8
                p = np.exp(2.5 * scores)
                p /= np.sum(p)
                return self.rng.choice(nbrs, p=p)
            return self.rng.choice(nbrs)
        return self.rng.integers(0, 240)

class QuantumEngine:
    """Quantum processing engine for E8 dynamics."""
    def __init__(self, physics, config, console):
        self.physics = physics
        self.config = config
        self.console = console
        self.state = np.random.uniform(0, 1, size=240).astype(np.complex64)
        self.console.log("[INIT] Quantum Engine online.")

    def evolve(self, dt: float = 0.01, sensor8: Optional[np.ndarray] = None):
        """Evolve quantum state."""
        H = self.physics.weights.astype(np.complex64)
        if sensor8 is not None and np.linalg.norm(sensor8) > 0:
            # Add sensor coupling
            coupling = np.outer(sensor8[:min(8, len(sensor8))], sensor8[:min(8, len(sensor8))])
            coupling_full = np.zeros_like(H)
            coupling_full[:coupling.shape[0], :coupling.shape[1]] = coupling
            H = H + 0.1 * coupling_full

        # Simple time evolution
        self.state = self.state * np.exp(-1j * dt * np.diag(H))

        # Normalize
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm

        return self.state

    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities."""
        return np.abs(self.state) ** 2

    def measure(self) -> int:
        """Perform quantum measurement."""
        probs = self.get_probabilities()
        return np.random.choice(len(probs), p=probs)
