"""
SmallMind Command System

A sophisticated command discovery, organization, and execution system that integrates
with the neuro agents to provide intelligent command routing and natural language processing.

Key Features:
- Hierarchical command organization with numbering
- Natural language command parsing
- Secure command execution with safety checks
- Integration with neuro agents for intelligent discovery
- Interactive CLI with help system
- Command database with usage analytics

Usage:
    # Command line interface
    python -m smallmind.commands "train a neural network"
    python -m smallmind.commands --interactive
    
    # Programmatic usage
    from smallmind.commands import CommandDatabase, CommandExecutor
    
    db = CommandDatabase()
    executor = CommandExecutor(db)
    result = executor.execute_natural_language("show brain simulation tools")
"""

from ................................................command_database import CommandDatabase, Command, Category
from ................................................natural_language_parser import NaturalLanguageParser, ParsedIntent
from ................................................command_executor import CommandExecutor, ExecutionResult, ExecutionContext
from ................................................cli import SmallMindCLI

# Exponential growth components
from ................................................curious_agents import CuriousAgent, ExponentialGrowthEngine
from ................................................adaptive_learning import AdaptiveLearningEngine, CommandDNA
from ................................................exponential_orchestrator import ExponentialOrchestrator
from ................................................neuro_integration import NeuroAgentConnector, SmartCommandDiscovery

__version__ = "2.0.0"  # Major version bump for exponential growth features
__all__ = [
    # Core system
    "CommandDatabase",
    "Command", 
    "Category",
    "NaturalLanguageParser",
    "ParsedIntent",
    "CommandExecutor",
    "ExecutionResult",
    "ExecutionContext",
    "SmallMindCLI",
    
    # Exponential growth system
    "CuriousAgent",
    "ExponentialGrowthEngine", 
    "AdaptiveLearningEngine",
    "CommandDNA",
    "ExponentialOrchestrator",
    "NeuroAgentConnector",
    "SmartCommandDiscovery"
]
