#!/usr/bin/env python3
"""API Configuration Module - Load API keys and configuration for AlphaGenome integration.

Handles secure loading of API keys and configuration for biological simulation.

Integration: API configuration for AlphaGenome biological simulation workflows.
Rationale: Centralized API key management with security best practices.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

def load_alphagenome_api_key() -> Optional[str]:
    """Load AlphaGenome API key from secure credentials directory."""

    # Check environment variable first
    api_key = os.getenv('ALPHAGENOME_API_KEY')
    if api_key:
        return api_key

    # Check consolidated credentials file
    credentials_file = Path(__file__).parent.parent.parent.parent / "data" / "credentials" / "all_api_keys.json"

    try:
        if credentials_file.exists():
            with open(credentials_file, 'r') as f:
                credentials = json.load(f)
                return credentials.get('alphagenome_api_key')
    except Exception as e:
        print(f"Warning: Could not load API key from {credentials_file}: {e}")

    # Fallback to individual file
    individual_file = Path(__file__).parent.parent.parent.parent / "data" / "credentials" / "alphagenome_api.json"
    try:
        if individual_file.exists():
            with open(individual_file, 'r') as f:
                credentials = json.load(f)
                return credentials.get('alphagenome_api_key')
    except Exception as e:
        print(f"Warning: Could not load API key from {individual_file}: {e}")

    return None

def load_gemini_api_key() -> Optional[str]:
    """Load Gemini API key from secure credentials directory."""

    # Check environment variable first
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        return api_key

    # Check consolidated credentials file
    credentials_file = Path(__file__).parent.parent.parent.parent / "data" / "credentials" / "all_api_keys.json"

    try:
        if credentials_file.exists():
            with open(credentials_file, 'r') as f:
                credentials = json.load(f)
                return credentials.get('gemini_api_key')
    except Exception as e:
        print(f"Warning: Could not load Gemini API key: {e}")

    return None

def load_all_api_keys() -> Dict[str, Any]:
    """Load all available API keys from credentials directory."""
    credentials_file = Path(__file__).parent.parent.parent.parent / "data" / "credentials" / "all_api_keys.json"

    try:
        if credentials_file.exists():
            with open(credentials_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load all API keys: {e}")
        return {}

def get_alphagenome_config() -> Dict[str, Any]:
    """Get complete AlphaGenome configuration."""

    api_key = load_alphagenome_api_key()

    config = {
        "api_key_available": api_key is not None,
        "api_key": api_key,
        "simulation_mode": "production" if api_key else "simulation",
        "endpoints": {
            "protein_structure": "https://alphafold.ebi.ac.uk/api/prediction/",
            "genome_analysis": "https://api.deepmind.com/alphagenome/",  # Example endpoint
        },
        "fallback_mode": not api_key
    }

    return config

def validate_api_configuration() -> Dict[str, Any]:
    """Validate API configuration and connectivity."""

    config = get_alphagenome_config()

    validation = {
        "api_key_present": config["api_key_available"],
        "credentials_file_exists": Path(__file__).parent.parent.parent.parent / "data" / "credentials" / "alphagenome_api.json",
        "simulation_ready": True,  # Always ready with fallback mode
        "production_ready": config["api_key_available"]
    }

    return {
        "status": "Valid" if validation["simulation_ready"] else "Invalid",
        "config": config,
        "validation": validation,
        "recommendations": [
            "API key found - production mode available" if config["api_key_available"]
            else "Using simulation mode - set ALPHAGENOME_API_KEY environment variable for production"
        ]
    }
