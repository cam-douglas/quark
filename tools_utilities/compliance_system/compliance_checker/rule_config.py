"""
Rule Configuration

Configuration for rule compliance checking including limits,
patterns, and exclusions.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict, List, Any


def get_rule_config() -> Dict[str, Any]:
    """Get rule configuration for compliance checking"""
    return {
        "file_size_limits": {
            "python_files": 300,
            "markdown_files": 1000,
            "yaml_files": 500,
            "json_files": 200
        },
        "required_patterns": {
            "python_files": [
                r'"""[\s\S]*?"""',  # Docstring
                r'from typing import',  # Type hints
            ],
            "test_files": [
                r'def test_',  # Test functions
                r'import pytest',  # Pytest import
            ]
        },
        "prohibited_patterns": {
            "all_files": [
                r'password\s*=\s*["\'][^"\']+["\']',  # Hardcoded passwords
                r'api_key\s*=\s*["\'][^"\']+["\']',  # Hardcoded API keys
                r'eval\s*\(',  # eval() usage
                r'exec\s*\(',  # exec() usage
            ],
            "brain_modules": [
                r'negative_emotion',  # Prohibited in brain modules
                r'harmful_behavior',  # Prohibited patterns
                r'toxic_pattern',  # Prohibited patterns
            ]
        },
        "directory_exclusions": [
            "archive/",
            "backup/",
            "deprecated/",
            "superseded/",
            "__pycache__/",
            ".git/",
            "node_modules/",
            "venv/",
            "env/",
        ]
    }
