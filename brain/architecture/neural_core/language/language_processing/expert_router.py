#!/usr/bin/env python3
"""Expert Router Module - Semantic routing and expert selection for language processing.

Handles semantic routing of prompts to appropriate language experts and models.

Integration: Expert routing for language cortex and neural language processing.
Rationale: Specialized routing logic separate from API clients and response generation.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ExpertRouter:
    """Semantic router for selecting appropriate language experts."""

    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Define expert intents and their embeddings
        self.intents = {
            'general_conversation': [
                "Hello", "How are you?", "What's up?", "Tell me about yourself",
                "Chat with me", "Let's talk", "Conversation", "Discuss"
            ],
            'scientific_analysis': [
                "Analyze this data", "What does this research show?", "Explain the methodology",
                "Scientific conclusion", "Research findings", "Data interpretation"
            ],
            'creative_writing': [
                "Write a story", "Create a poem", "Be creative", "Imagine a scenario",
                "Creative writing", "Artistic expression", "Fiction"
            ],
            'technical_explanation': [
                "Explain how this works", "Technical details", "Implementation guide",
                "Architecture overview", "System design", "Code explanation"
            ],
            'problem_solving': [
                "Help me solve", "How do I fix", "What's the solution", "Troubleshoot",
                "Debug this", "Problem analysis", "Solution strategy"
            ],
            'learning_assistance': [
                "Teach me", "Help me understand", "Explain this concept", "Learning guide",
                "Educational content", "Tutorial", "Study help"
            ]
        }

        # Pre-compute intent embeddings
        self.intent_embeddings = {}
        for intent, examples in self.intents.items():
            embeddings = self.sentence_model.encode(examples)
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)

        print("-> Semantic router initialized with ENRICHED intents.")

    def route_to_expert(self, prompt: str) -> str:
        """Route prompt to most appropriate expert based on semantic similarity."""
        try:
            # Encode the input prompt
            prompt_embedding = self.sentence_model.encode([prompt])

            # Calculate similarities with each intent
            similarities = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                similarity = util.cos_sim(prompt_embedding, intent_embedding).item()
                similarities[intent] = similarity

            # Select the intent with highest similarity
            best_intent = max(similarities, key=similarities.get)
            best_score = similarities[best_intent]

            # If similarity is too low, default to general conversation
            if best_score < 0.3:
                return 'general_conversation'

            return best_intent

        except Exception as e:
            logger.error(f"Expert routing error: {e}")
            return 'general_conversation'  # Safe fallback

    def get_expert_preferences(self, intent: str) -> Dict[str, float]:
        """Get preferred API services for a given intent."""
        preferences = {
            'general_conversation': {'openai': 0.3, 'anthropic': 0.3, 'openrouter': 0.2, 'gemini': 0.2},
            'scientific_analysis': {'anthropic': 0.4, 'openai': 0.3, 'openrouter': 0.2, 'gemini': 0.1},
            'creative_writing': {'openai': 0.4, 'anthropic': 0.3, 'openrouter': 0.2, 'gemini': 0.1},
            'technical_explanation': {'anthropic': 0.3, 'openai': 0.3, 'openrouter': 0.2, 'gemini': 0.2},
            'problem_solving': {'openai': 0.3, 'anthropic': 0.3, 'openrouter': 0.2, 'gemini': 0.2},
            'learning_assistance': {'anthropic': 0.3, 'openai': 0.3, 'openrouter': 0.2, 'gemini': 0.2}
        }

        return preferences.get(intent, {'openai': 0.3, 'anthropic': 0.3, 'openrouter': 0.2, 'gemini': 0.2})

    def select_api_service(self, intent: str, available_services: Dict[str, bool]) -> Optional[str]:
        """Select the best available API service for the given intent."""
        preferences = self.get_expert_preferences(intent)

        # Filter to only available services
        available_preferences = {
            service: score for service, score in preferences.items()
            if available_services.get(service, False)
        }

        if not available_preferences:
            return None

        # Select service with highest preference score
        return max(available_preferences, key=available_preferences.get)
