#!/usr/bin/env python3
"""API Loader for Language Cortex - Load API keys from secure credentials directory.

Handles secure loading of API keys for language processing services.

Integration: API configuration for language cortex and neural language processing.
Rationale: Centralized API key management for language services.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

def load_language_api_keys() -> Dict[str, Optional[str]]:
    """Load all language-related API keys from secure credentials directory."""

    # Path to consolidated credentials
    credentials_file = Path(__file__).parent.parent.parent.parent.parent / "data" / "credentials" / "all_api_keys.json"

    api_keys = {
        'openai': None,
        'anthropic': None,
        'gemini': None,
        'alphagenome': None,
        'openrouter': None
    }

    try:
        if credentials_file.exists():
            with open(credentials_file, 'r') as f:
                credentials = json.load(f)

                # Extract keys from services section
                services = credentials.get('services', {})

                if 'openai' in services:
                    api_keys['openai'] = services['openai'].get('api_key')

                if 'claude' in services:
                    api_keys['anthropic'] = services['claude'].get('api_key')

                if 'gemini' in services:
                    api_keys['gemini'] = services['gemini'].get('api_key')

                if 'alphagenome' in services:
                    api_keys['alphagenome'] = services['alphagenome'].get('api_key')

                if 'openrouter' in services:
                    api_keys['openrouter'] = services['openrouter'].get('api_key')

                # Also check top-level keys for backward compatibility
                api_keys['openai'] = api_keys['openai'] or credentials.get('openai_api_key')
                api_keys['anthropic'] = api_keys['anthropic'] or credentials.get('anthropic_api_key')
                api_keys['gemini'] = api_keys['gemini'] or credentials.get('gemini_api_key')
                api_keys['alphagenome'] = api_keys['alphagenome'] or credentials.get('alphagenome_api_key')
                api_keys['openrouter'] = api_keys['openrouter'] or credentials.get('openrouter_api_key')

    except Exception as e:
        print(f"Warning: Could not load API keys from {credentials_file}: {e}")

    # Check environment variables as fallback
    for service in api_keys:
        if not api_keys[service]:
            env_var = f"{service.upper()}_API_KEY"
            api_keys[service] = os.getenv(env_var)

    return api_keys

def validate_api_key(api_key: Optional[str]) -> bool:
    """Validate that an API key is present and not a placeholder."""
    if not api_key:
        return False

    # Check for placeholder text
    placeholders = [
        'YOUR_', 'PLACEHOLDER', 'MOVED_TO_CREDENTIALS',
        'REPLACE_WITH', 'INSERT_YOUR', 'CHANGE_THIS'
    ]

    return not any(placeholder in api_key.upper() for placeholder in placeholders)

def get_language_api_config() -> Dict[str, Any]:
    """Get complete language API configuration."""
    api_keys = load_language_api_keys()

    config = {
        'api_keys': api_keys,
        'services_available': {},
        'fallback_mode': True
    }

    # Check which services are available
    for service, api_key in api_keys.items():
        is_valid = validate_api_key(api_key)
        config['services_available'][service] = is_valid

        if is_valid:
            config['fallback_mode'] = False  # At least one service available

    return config
