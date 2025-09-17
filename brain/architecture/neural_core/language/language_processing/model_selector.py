#!/usr/bin/env python3
"""Model Selector Module - Intelligent model selection using GPT as meta-reasoner.

Uses GPT to dynamically select the best available language model for each prompt.

Integration: Model selection intelligence for language cortex and neural processing.
Rationale: Intelligent model routing separate from API clients and response generation.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)

class ModelSelector:
    """Intelligent model selector using GPT as meta-reasoner."""

    def __init__(self, api_clients):
        self.api_clients = api_clients
        self.selection_history = deque(maxlen=100)
        self.performance_metrics = {}

        # Rate limiting for selector calls
        self.selector_calls = deque(maxlen=10)  # 10 calls per minute

    def select_best_model(self, prompt: str, available_models: Dict[str, bool]) -> Optional[str]:
        """Use GPT as meta-reasoner to select the best model for the prompt."""

        # Check rate limiting for selector
        if not self._allow_selector_call():
            return self._fallback_selection(available_models)

        try:
            # Get all available models (APIs + local)
            all_available = self._get_comprehensive_model_list(available_models)

            if not all_available:
                return None

            # Use GPT to make intelligent selection
            selection_result = self._query_gpt_selector(prompt, all_available)

            if selection_result:
                self._record_selector_call()
                self._record_selection(prompt, selection_result)
                return selection_result["provider"]

        except Exception as e:
            logger.error(f"Model selector error: {e}")

        # Fallback to heuristic selection
        return self._fallback_selection(available_models)

    def _get_comprehensive_model_list(self, available_services: Dict[str, bool]) -> List[str]:
        """Get comprehensive list of all available models."""
        models = []

        # API services
        api_models = {
            'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
            'anthropic': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
            'gemini': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro'],
            'openrouter': [
                # OpenAI models via OpenRouter
                'openai/gpt-4o',
                'openai/gpt-4o-mini',
                'openai/gpt-3.5-turbo',
                # Anthropic models via OpenRouter
                'anthropic/claude-3-opus',
                'anthropic/claude-3-sonnet',
                'anthropic/claude-3-haiku',
                # Google models via OpenRouter
                'google/gemini-pro-1.5',
                'google/gemma-2-9b-it',
                # Meta models via OpenRouter
                'meta-llama/llama-3.1-405b-instruct',
                'meta-llama/llama-3.1-70b-instruct',
                'meta-llama/llama-3.1-8b-instruct',
                # Microsoft models via OpenRouter
                'microsoft/phi-3-medium-128k-instruct',
                'microsoft/phi-3-mini-128k-instruct',
                # Mistral models via OpenRouter
                'mistralai/mistral-7b-instruct',
                'mistralai/mixtral-8x7b-instruct',
                # Other specialized models
                'perplexity/llama-3.1-sonar-large-128k-online',
                'deepseek/deepseek-chat',
                'qwen/qwen-2-72b-instruct'
            ]
        }

        for service, service_models in api_models.items():
            if available_services.get(service, False):
                models.extend([f"{service}:{model}" for model in service_models])

        # Local models (if available)
        local_models = ['local:flan-t5-base', 'local:conversation', 'local:qa']
        for local_model in local_models:
            if available_services.get(local_model.split(':')[1], False):
                models.append(local_model)

        return models

    def _query_gpt_selector(self, prompt: str, available_models: List[str]) -> Optional[Dict[str, Any]]:
        """Query GPT to select the best model for the prompt."""

        if not self.api_clients.openai_client:
            return None

        try:
            # Create selection prompt for GPT
            system_msg = (
                "You are an intelligent model router for a language processing system. "
                "Your job is to select the BEST model for each user prompt based on the prompt characteristics. "
                "Consider: complexity, domain expertise needed, response style, computational requirements, cost efficiency. "
                "Return strict JSON with: {\"provider\": \"service:model\", \"reason\": \"explanation\", \"confidence\": 0.0-1.0}. "
                "Available models format: 'service:model' (e.g., 'openai:gpt-4o', 'openrouter:meta-llama/llama-3.1-405b-instruct'). "
                "\n"
                "MODEL CAPABILITIES:\n"
                "Direct APIs:\n"
                "- openai:gpt-4o - Best for complex reasoning, coding, analysis\n"
                "- anthropic:claude-3-opus - Best for safety-critical, nuanced writing\n"
                "- gemini:gemini-1.5-pro - Good balance, fast responses\n"
                "\n"
                "OpenRouter Models (cost-effective, diverse options):\n"
                "- openrouter:meta-llama/llama-3.1-405b-instruct - Largest model, best for complex tasks\n"
                "- openrouter:anthropic/claude-3-opus - Claude via OpenRouter\n"
                "- openrouter:openai/gpt-4o - GPT-4o via OpenRouter\n"
                "- openrouter:mistralai/mixtral-8x7b-instruct - Good for technical tasks\n"
                "- openrouter:perplexity/llama-3.1-sonar-large-128k-online - Real-time web access\n"
                "- openrouter:deepseek/deepseek-chat - Specialized for code and reasoning\n"
                "- openrouter:microsoft/phi-3-mini-128k-instruct - Fast, efficient for simple tasks\n"
                "\n"
                "SELECTION STRATEGY:\n"
                "- Prefer OpenRouter models for cost efficiency when quality is comparable\n"
                "- Use direct APIs when specific features/reliability needed\n"
                "- Match model size to task complexity\n"
                "- Consider real-time needs (Perplexity for current info)\n"
                "\n"
                "The provider MUST be exactly one from the available list."
            )

            user_msg = json.dumps({
                "available_models": available_models,
                "user_prompt": prompt,
                "selection_criteria": "best_match_for_prompt_type"
            })

            response = self.api_clients.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use efficient model for selection
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1,  # Low temperature for consistent selection
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            selection_data = json.loads(content)

            # Validate selection
            selected_provider = selection_data.get("provider", "")
            if selected_provider in available_models:
                return {
                    "provider": selected_provider,
                    "reason": selection_data.get("reason", ""),
                    "confidence": selection_data.get("confidence", 0.5),
                    "selection_method": "gpt_meta_reasoner"
                }

        except json.JSONDecodeError:
            logger.warning(f"Model selector returned invalid JSON: {content}")
        except Exception as e:
            logger.error(f"GPT model selector error: {e}")

        return None

    def _fallback_selection(self, available_services: Dict[str, bool]) -> Optional[str]:
        """Fallback heuristic model selection when GPT selector unavailable."""

        # Priority order for fallback
        priority_order = [
            'openrouter', # Cost-effective with many options
            'anthropic',  # Good general performance
            'openai',     # Strong reasoning
            'gemini',     # Fast responses
            'local'       # Always available fallback
        ]

        for service in priority_order:
            if available_services.get(service, False):
                # Return specific model for OpenRouter
                if service == 'openrouter':
                    return 'openrouter:openai/gpt-3.5-turbo'  # Good default
                return service

        return None

    def _allow_selector_call(self) -> bool:
        """Check if selector call is allowed based on rate limiting."""
        current_time = time.time()

        # Remove calls older than 1 minute
        while self.selector_calls and current_time - self.selector_calls[0] > 60:
            self.selector_calls.popleft()

        return len(self.selector_calls) < 10  # Max 10 selector calls per minute

    def _record_selector_call(self):
        """Record a selector call for rate limiting."""
        self.selector_calls.append(time.time())

    def _record_selection(self, prompt: str, selection_result: Dict[str, Any]):
        """Record selection for performance analysis."""
        self.selection_history.append({
            "timestamp": time.time(),
            "prompt_length": len(prompt),
            "selected_provider": selection_result["provider"],
            "reason": selection_result["reason"],
            "confidence": selection_result["confidence"],
            "method": selection_result["selection_method"]
        })

    def get_selection_analytics(self) -> Dict[str, Any]:
        """Get analytics on model selection patterns."""
        if not self.selection_history:
            return {"total_selections": 0}

        # Analyze selection patterns
        providers = [s["selected_provider"] for s in self.selection_history]
        provider_counts = {}
        for provider in providers:
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        avg_confidence = sum(s["confidence"] for s in self.selection_history) / len(self.selection_history)

        return {
            "total_selections": len(self.selection_history),
            "provider_distribution": provider_counts,
            "average_confidence": avg_confidence,
            "gpt_selector_usage": sum(1 for s in self.selection_history if s["method"] == "gpt_meta_reasoner"),
            "fallback_usage": sum(1 for s in self.selection_history if s["method"] == "fallback")
        }
