#!/usr/bin/env python3
"""API Clients Module - API client management and initialization for language processing.

Handles API client initialization and management for OpenAI, Anthropic, Gemini services.

Integration: API client layer for language cortex and neural language processing.
Rationale: Centralized API client management separate from routing and response logic.
"""

import openai
import anthropic
import google.generativeai as genai
import os
import logging
from typing import Optional, Dict
from ..api_loader import load_language_api_keys, validate_api_key

logger = logging.getLogger(__name__)

class LanguageAPIClients:
    """Manages API clients for language processing services."""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_model = None
        self.openrouter_client = None

        # Load and initialize all API clients
        self._initialize_api_clients()

    def _initialize_api_clients(self):
        """Initialize all API clients from secure credentials."""
        try:
            api_keys = load_language_api_keys()

            # OpenAI
            openai_key = api_keys.get('openai')
            if validate_api_key(openai_key):
                # Disable auto-retries to avoid long stalls on 429
                self.openai_client = openai.OpenAI(api_key=openai_key, max_retries=0)
                print("✅ OpenAI client initialized from secure credentials.")
            else:
                self.openai_client = None
                print("❌ OpenAI key not found in credentials directory.")

            # Anthropic
            anthropic_key = api_keys.get('anthropic')
            if validate_api_key(anthropic_key):
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print("✅ Anthropic client initialized from secure credentials.")
            else:
                self.anthropic_client = None
                print("❌ Anthropic key not found in credentials directory.")

            # Gemini
            gemini_key = api_keys.get('gemini')
            if validate_api_key(gemini_key):
                genai.configure(api_key=gemini_key)
                gemini_model_name = os.environ.get('QUARK_GEMINI_MODEL', 'gemini-1.5-flash')
                self.gemini_model = genai.GenerativeModel(gemini_model_name)
                print("✅ Gemini client initialized from secure credentials.")
            else:
                self.gemini_model = None
                print("❌ Gemini key not found in credentials directory.")

            # AlphaGenome
            alphagenome_key = api_keys.get('alphagenome')
            if validate_api_key(alphagenome_key):
                # Store for AlphaGenome modules to use
                os.environ['ALPHAGENOME_API_KEY'] = alphagenome_key
                print("✅ AlphaGenome API key configured from secure credentials.")
            else:
                print("❌ AlphaGenome API key not found in credentials directory.")

            # OpenRouter
            openrouter_key = api_keys.get('openrouter')
            if validate_api_key(openrouter_key):
                # OpenRouter uses OpenAI-compatible API
                self.openrouter_client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    max_retries=0
                )
                print("✅ OpenRouter client initialized from secure credentials.")
            else:
                self.openrouter_client = None
                print("❌ OpenRouter key not found in credentials directory.")

        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            self.openai_client = None
            self.anthropic_client = None
            self.gemini_model = None
            self.openrouter_client = None
            print("❌ Failed to load API configuration. Using local models only.")

    def query_gemini(self, prompt: str) -> Optional[str]:
        """Query the Google Gemini model."""
        if not self.gemini_model:
            return None
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini query error: {e}")
            return None

    def query_openai(self, prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
        """Query OpenAI GPT models."""
        if not self.openai_client:
            return None
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI query error: {e}")
            return None

    def query_anthropic(self, prompt: str, model: str = "claude-3-haiku-20240307") -> Optional[str]:
        """Query Anthropic Claude models."""
        if not self.anthropic_client:
            return None
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic query error: {e}")
            return None

    def query_openrouter(self, prompt: str, model: str = "openai/gpt-3.5-turbo") -> Optional[str]:
        """Query OpenRouter models (supports multiple providers)."""
        if not self.openrouter_client:
            return None
        try:
            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter query error: {e}")
            return None

    def get_available_services(self) -> Dict[str, bool]:
        """Get status of available API services."""
        return {
            'openai': self.openai_client is not None,
            'anthropic': self.anthropic_client is not None,
            'gemini': self.gemini_model is not None,
            'openrouter': self.openrouter_client is not None
        }
