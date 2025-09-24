"""
Validation Core Modules
=======================
Core functionality for the Quark validation system.
"""

from .validator import QuarkValidator
from .scope_selector import ScopeSelector
from .prerequisite_checker import PrerequisiteChecker
from .kpi_runner import KPIRunner
from .evidence_collector import EvidenceCollector
from .rubric_manager import RubricManager
from .dashboard_generator import DashboardGenerator
from .checklist_parser import ChecklistParser
from .sprint_guide import SprintGuide
from .rules_validator import RulesValidator

__all__ = [
    'QuarkValidator',
    'ScopeSelector',
    'PrerequisiteChecker',
    'KPIRunner',
    'EvidenceCollector',
    'RubricManager',
    'DashboardGenerator',
    'ChecklistParser',
    'SprintGuide',
    'RulesValidator'
]

# Version info
__version__ = '1.0.0'
