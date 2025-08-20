#!/usr/bin/env python3
"""
Intelligent Error Learning System for Small-Mind Super Intelligence

This system ensures that for every "command not found" or failure,
the system gets smarter by analyzing why it didn't know the answer
and learning from the experience.
"""

import sys
import os
import time
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib
import re
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, asdict

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

@dataclass
class ErrorAnalysis:
    """Structured error analysis data."""
    error_id: str
    timestamp: str
    error_type: str
    error_message: str
    context: Dict[str, Any]
    root_cause: str
    learning_insights: List[str]
    improvement_actions: List[str]
    confidence_score: float
    resolved: bool = False
    resolution_method: Optional[str] = None

@dataclass
class LearningOutcome:
    """Structured learning outcome from error analysis."""
    outcome_id: str
    timestamp: str
    original_error_id: str
    knowledge_gained: str
    capability_improvement: str
    new_patterns_learned: List[str]
    confidence_boost: float
    applied_to_system: bool = False

class IntelligentErrorLearner:
    """
    System that learns from every error and failure to continuously improve.
    """
    
    def __init__(self, super_intelligence=None):
        self.super_intelligence = super_intelligence
        self.error_database = []
        self.learning_outcomes = []
        self.pattern_database = {}
        self.improvement_history = []
        
        # Initialize error analysis patterns
        self._init_error_patterns()
        
        # Initialize learning strategies
        self._init_learning_strategies()
        
        logger.info("🧠 Intelligent Error Learning System initialized")
    
    def _init_error_patterns(self):
        """Initialize patterns for different types of errors."""
        self.error_patterns = {
            "command_not_found": {
                "patterns": [
                    r"command not found",
                    r"command not found:",
                    r"zsh: command not found",
                    r"bash: command not found",
                    r"sh: command not found"
                ],
                "analysis_method": "command_analysis",
                "learning_focus": "command_knowledge",
                "priority": "high"
            },
            "import_error": {
                "patterns": [
                    r"ImportError:",
                    r"ModuleNotFoundError:",
                    r"No module named",
                    r"cannot import name"
                ],
                "analysis_method": "dependency_analysis",
                "learning_focus": "dependency_management",
                "priority": "high"
            },
            "permission_error": {
                "patterns": [
                    r"Permission denied",
                    r"PermissionError:",
                    r"Access denied",
                    r"Operation not permitted"
                ],
                "analysis_method": "permission_analysis",
                "learning_focus": "security_permissions",
                "priority": "medium"
            },
            "file_not_found": {
                "patterns": [
                    r"FileNotFoundError:",
                    r"No such file or directory",
                    r"file not found",
                    r"cannot find the file"
                ],
                "analysis_method": "file_system_analysis",
                "learning_focus": "file_management",
                "priority": "medium"
            },
            "syntax_error": {
                "patterns": [
                    r"SyntaxError:",
                    r"invalid syntax",
                    r"unexpected token",
                    r"missing colon"
                ],
                "analysis_method": "code_analysis",
                "learning_focus": "code_quality",
                "priority": "medium"
            },
            "runtime_error": {
                "patterns": [
                    r"RuntimeError:",
                    r"TypeError:",
                    r"ValueError:",
                    r"AttributeError:"
                ],
                "analysis_method": "runtime_analysis",
                "learning_focus": "error_handling",
                "priority": "medium"
            },
            "network_error": {
                "patterns": [
                    r"Connection refused",
                    r"Network unreachable",
                    r"timeout",
                    r"connection failed"
                ],
                "analysis_method": "network_analysis",
                "learning_focus": "network_management",
                "priority": "low"
            }
        }
    
    def _init_learning_strategies(self):
        """Initialize learning strategies for different error types."""
        self.learning_strategies = {
            "command_knowledge": {
                "methods": ["command_research", "alternative_discovery", "context_learning"],
                "improvement_type": "capability_expansion",
                "confidence_boost": 0.3
            },
            "dependency_management": {
                "methods": ["package_research", "version_analysis", "compatibility_check"],
                "improvement_type": "system_knowledge",
                "confidence_boost": 0.4
            },
            "security_permissions": {
                "methods": ["permission_analysis", "security_learning", "best_practices"],
                "improvement_type": "security_awareness",
                "confidence_boost": 0.2
            },
            "file_management": {
                "methods": ["path_analysis", "file_system_learning", "organization_patterns"],
                "improvement_type": "system_understanding",
                "confidence_boost": 0.3
            },
            "code_quality": {
                "methods": ["syntax_learning", "best_practices", "pattern_recognition"],
                "improvement_type": "code_improvement",
                "confidence_boost": 0.4
            },
            "error_handling": {
                "methods": ["error_pattern_learning", "robustness_improvement", "fallback_strategies"],
                "improvement_type": "system_reliability",
                "confidence_boost": 0.3
            },
            "network_management": {
                "methods": ["network_analysis", "connectivity_learning", "troubleshooting"],
                "improvement_type": "network_understanding",
                "confidence_boost": 0.2
            }
        }
    
    def analyze_error(self, error_message: str, context: Dict[str, Any]) -> ErrorAnalysis:
        """
        Analyze an error and determine why it occurred.
        
        Args:
            error_message: The error message or output
            context: Context information about the error
            
        Returns:
            Structured error analysis with learning insights
        """
        # Determine error type
        error_type = self._classify_error(error_message)
        
        # Generate unique error ID
        error_id = self._generate_error_id(error_message, context)
        
        # Analyze root cause
        root_cause = self._analyze_root_cause(error_type, error_message, context)
        
        # Generate learning insights
        learning_insights = self._generate_learning_insights(error_type, root_cause, context)
        
        # Generate improvement actions
        improvement_actions = self._generate_improvement_actions(error_type, learning_insights)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(error_type, root_cause, context)
        
        # Create error analysis
        error_analysis = ErrorAnalysis(
            error_id=error_id,
            timestamp=datetime.utcnow().isoformat(),
            error_type=error_type,
            error_message=error_message,
            context=context,
            root_cause=root_cause,
            learning_insights=learning_insights,
            improvement_actions=improvement_actions,
            confidence_score=confidence_score
        )
        
        # Store in database
        self.error_database.append(error_analysis)
        
        # Trigger learning process
        self._trigger_learning_process(error_analysis)
        
        logger.info(f"🔍 Error analyzed: {error_type} - {root_cause}")
        
        return error_analysis
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error based on patterns."""
        error_message_lower = error_message.lower()
        
        for error_type, pattern_info in self.error_patterns.items():
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, error_message_lower, re.IGNORECASE):
                    return error_type
        
        # Default to unknown error type
        return "unknown_error"
    
    def _generate_error_id(self, error_message: str, context: Dict[str, Any]) -> str:
        """Generate a unique ID for the error."""
        # Create a hash from error message and context
        error_data = f"{error_message}{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(error_data.encode()).hexdigest()[:12]
    
    def _analyze_root_cause(self, error_type: str, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze the root cause of the error."""
        if error_type == "command_not_found":
            return self._analyze_command_not_found(error_message, context)
        elif error_type == "import_error":
            return self._analyze_import_error(error_message, context)
        elif error_type == "permission_error":
            return self._analyze_permission_error(error_message, context)
        elif error_type == "file_not_found":
            return self._analyze_file_not_found(error_message, context)
        elif error_type == "syntax_error":
            return self._analyze_syntax_error(error_message, context)
        elif error_type == "runtime_error":
            return self._analyze_runtime_error(error_message, context)
        elif error_type == "network_error":
            return self._analyze_network_error(error_message, context)
        else:
            return "Unknown error type - requires deeper analysis"
    
    def _analyze_command_not_found(self, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze why a command was not found."""
        # Extract the command that wasn't found
        command_match = re.search(r"command not found:?\s*(\S+)", error_message, re.IGNORECASE)
        if command_match:
            command = command_match.group(1)
            
            # Check if it's a common command that should be available
            common_commands = ["python", "pip", "git", "node", "npm", "docker", "kubectl"]
            if command in common_commands:
                return f"Common command '{command}' not available - system may need setup or PATH configuration"
            
            # Check if it's a custom script or tool
            if command.endswith('.py') or command.endswith('.sh'):
                return f"Script '{command}' not found - file may not exist or be in wrong location"
            
            return f"Command '{command}' not found - may need installation or different approach"
        
        return "Command not found - unable to determine specific command"
    
    def _analyze_import_error(self, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze why an import failed."""
        # Extract the module that couldn't be imported
        module_match = re.search(r"No module named ['\"]?(\w+)['\"]?", error_message)
        if module_match:
            module = module_match.group(1)
            return f"Module '{module}' not installed - requires pip install or alternative approach"
        
        # Check for import name errors
        name_match = re.search(r"cannot import name ['\"]?(\w+)['\"]?", error_message)
        if name_match:
            name = name_match.group(1)
            return f"Cannot import '{name}' - may be version compatibility issue or missing dependency"
        
        return "Import failed - unable to determine specific cause"
    
    def _analyze_permission_error(self, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze why a permission error occurred."""
        if "Permission denied" in error_message:
            return "Insufficient permissions - may need sudo or proper user rights"
        elif "Access denied" in error_message:
            return "Access denied - file or resource may be protected"
        elif "Operation not permitted" in error_message:
            return "Operation not permitted - may be system restriction or security policy"
        
        return "Permission error - unable to determine specific cause"
    
    def _analyze_file_not_found(self, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze why a file was not found."""
        # Extract the file path
        file_match = re.search(r"['\"]?([^'\"]+)['\"]?", error_message)
        if file_match:
            file_path = file_match.group(1)
            return f"File '{file_path}' not found - may be wrong path, missing file, or typo"
        
        return "File not found - unable to determine specific file"
    
    def _analyze_syntax_error(self, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze why a syntax error occurred."""
        if "invalid syntax" in error_message:
            return "Invalid syntax - code structure or formatting issue"
        elif "unexpected token" in error_message:
            return "Unexpected token - syntax or punctuation error"
        elif "missing colon" in error_message:
            return "Missing colon - likely in function definition or conditional statement"
        
        return "Syntax error - unable to determine specific cause"
    
    def _analyze_runtime_error(self, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze why a runtime error occurred."""
        if "TypeError" in error_message:
            return "Type error - wrong data type used in operation"
        elif "ValueError" in error_message:
            return "Value error - invalid value or parameter"
        elif "AttributeError" in error_message:
            return "Attribute error - object doesn't have expected attribute or method"
        
        return "Runtime error - unable to determine specific cause"
    
    def _analyze_network_error(self, error_message: str, context: Dict[str, Any]) -> str:
        """Analyze why a network error occurred."""
        if "Connection refused" in error_message:
            return "Connection refused - service may not be running or port blocked"
        elif "Network unreachable" in error_message:
            return "Network unreachable - connectivity or routing issue"
        elif "timeout" in error_message:
            return "Connection timeout - service may be slow or network congested"
        
        return "Network error - unable to determine specific cause"
    
    def _generate_learning_insights(self, error_type: str, root_cause: str, context: Dict[str, Any]) -> List[str]:
        """Generate learning insights from the error."""
        insights = []
        
        # Get learning strategy for this error type
        if error_type in self.error_patterns:
            learning_focus = self.error_patterns[error_type]["learning_focus"]
            strategy = self.learning_strategies.get(learning_focus, {})
            
            # Generate insights based on learning focus
            if learning_focus == "command_knowledge":
                insights.extend([
                    "Need to expand command knowledge base",
                    "Should learn alternative approaches for common tasks",
                    "Must understand system PATH and command availability"
                ])
            elif learning_focus == "dependency_management":
                insights.extend([
                    "Need to understand package dependencies better",
                    "Should learn version compatibility requirements",
                    "Must improve dependency resolution strategies"
                ])
            elif learning_focus == "security_permissions":
                insights.extend([
                    "Need to understand permission systems better",
                    "Should learn security best practices",
                    "Must improve permission handling strategies"
                ])
            elif learning_focus == "file_management":
                insights.extend([
                    "Need to understand file system structure better",
                    "Should learn path resolution and file organization",
                    "Must improve file handling strategies"
                ])
            elif learning_focus == "code_quality":
                insights.extend([
                    "Need to improve code syntax understanding",
                    "Should learn programming best practices",
                    "Must improve error prevention strategies"
                ])
            elif learning_focus == "error_handling":
                insights.extend([
                    "Need to improve error handling strategies",
                    "Should learn robust programming practices",
                    "Must improve fallback mechanisms"
                ])
            elif learning_focus == "network_management":
                insights.extend([
                    "Need to understand network connectivity better",
                    "Should learn troubleshooting strategies",
                    "Must improve network error handling"
                ])
        
        # Add general insights
        insights.extend([
            f"Error type '{error_type}' requires attention",
            f"Root cause: {root_cause}",
            "Should implement preventive measures for similar errors"
        ])
        
        return insights
    
    def _generate_improvement_actions(self, error_type: str, learning_insights: List[str]) -> List[str]:
        """Generate specific actions to improve based on the error."""
        actions = []
        
        if error_type == "command_not_found":
            actions.extend([
                "Research alternative commands for the same task",
                "Learn system PATH configuration",
                "Build command knowledge database",
                "Implement command suggestion system"
            ])
        elif error_type == "import_error":
            actions.extend([
                "Research package dependencies",
                "Learn version compatibility requirements",
                "Build dependency resolution system",
                "Implement automatic package installation"
            ])
        elif error_type == "permission_error":
            actions.extend([
                "Learn permission systems and security",
                "Research best practices for user management",
                "Build permission checking system",
                "Implement secure operation strategies"
            ])
        elif error_type == "file_not_found":
            actions.extend([
                "Learn file system organization",
                "Research path resolution strategies",
                "Build file discovery system",
                "Implement intelligent path suggestions"
            ])
        elif error_type == "syntax_error":
            actions.extend([
                "Improve code syntax understanding",
                "Learn programming best practices",
                "Build syntax validation system",
                "Implement code quality checks"
            ])
        elif error_type == "runtime_error":
            actions.extend([
                "Improve error handling strategies",
                "Learn robust programming practices",
                "Build error prevention system",
                "Implement comprehensive testing"
            ])
        elif error_type == "network_error":
            actions.extend([
                "Learn network troubleshooting",
                "Research connectivity strategies",
                "Build network monitoring system",
                "Implement connection fallbacks"
            ])
        
        # Add general improvement actions
        actions.extend([
            "Document this error type for future reference",
            "Implement automatic error recovery strategies",
            "Build error pattern recognition system",
            "Create knowledge base for common solutions"
        ])
        
        return actions
    
    def _calculate_confidence_score(self, error_type: str, root_cause: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for the error analysis."""
        base_score = 0.5
        
        # Boost score for well-understood error types
        if error_type in self.error_patterns:
            base_score += 0.2
        
        # Boost score for clear root causes
        if len(root_cause) > 20 and "unable to determine" not in root_cause:
            base_score += 0.2
        
        # Boost score for rich context
        if context and len(context) > 2:
            base_score += 0.1
        
        # Cap at 1.0
        return min(base_score, 1.0)
    
    def _trigger_learning_process(self, error_analysis: ErrorAnalysis):
        """Trigger the learning process for this error."""
        try:
            # Create learning outcome
            learning_outcome = self._create_learning_outcome(error_analysis)
            
            # Apply learning to system
            self._apply_learning_to_system(learning_outcome)
            
            # Update improvement history
            self.improvement_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": error_analysis.error_type,
                "improvements_applied": len(learning_outcome.new_patterns_learned),
                "confidence_boost": learning_outcome.confidence_boost
            })
            
            logger.info(f"🎓 Learning applied: {len(learning_outcome.new_patterns_learned)} new patterns")
            
        except Exception as e:
            logger.error(f"Error in learning process: {e}")
    
    def _create_learning_outcome(self, error_analysis: ErrorAnalysis) -> LearningOutcome:
        """Create a learning outcome from error analysis."""
        # Generate new patterns based on the error
        new_patterns = self._extract_new_patterns(error_analysis)
        
        # Calculate confidence boost
        confidence_boost = self._calculate_confidence_boost(error_analysis)
        
        learning_outcome = LearningOutcome(
            outcome_id=f"learn_{error_analysis.error_id}",
            timestamp=datetime.utcnow().isoformat(),
            original_error_id=error_analysis.error_id,
            knowledge_gained=error_analysis.root_cause,
            capability_improvement=error_analysis.error_type,
            new_patterns_learned=new_patterns,
            confidence_boost=confidence_boost
        )
        
        self.learning_outcomes.append(learning_outcome)
        return learning_outcome
    
    def _extract_new_patterns(self, error_analysis: ErrorAnalysis) -> List[str]:
        """Extract new patterns to learn from the error."""
        patterns = []
        
        # Extract command patterns
        if error_analysis.error_type == "command_not_found":
            command_match = re.search(r"command not found:?\s*(\S+)", error_analysis.error_message, re.IGNORECASE)
            if command_match:
                command = command_match.group(1)
                patterns.append(f"command_pattern:{command}")
                patterns.append(f"alternative_approach:{command}")
        
        # Extract module patterns
        elif error_analysis.error_type == "import_error":
            module_match = re.search(r"No module named ['\"]?(\w+)['\"]?", error_analysis.error_message)
            if module_match:
                module = module_match.group(1)
                patterns.append(f"module_pattern:{module}")
                patterns.append(f"dependency_pattern:{module}")
        
        # Extract file patterns
        elif error_analysis.error_type == "file_not_found":
            file_match = re.search(r"['\"]?([^'\"]+)['\"]?", error_analysis.error_message)
            if file_match:
                file_path = file_match.group(1)
                patterns.append(f"file_pattern:{file_path}")
                patterns.append(f"path_pattern:{os.path.dirname(file_path)}")
        
        # Add general error patterns
        patterns.append(f"error_type:{error_analysis.error_type}")
        patterns.append(f"root_cause:{error_analysis.root_cause[:50]}")
        
        return patterns
    
    def _calculate_confidence_boost(self, error_analysis: ErrorAnalysis) -> float:
        """Calculate confidence boost from learning this error."""
        base_boost = 0.1
        
        # Boost based on error type priority
        if error_analysis.error_type in self.error_patterns:
            priority = self.error_patterns[error_analysis.error_type]["priority"]
            if priority == "high":
                base_boost += 0.2
            elif priority == "medium":
                base_boost += 0.1
        
        # Boost based on confidence score
        base_boost += error_analysis.confidence_score * 0.1
        
        return min(base_boost, 0.5)  # Cap at 0.5
    
    def _apply_learning_to_system(self, learning_outcome: LearningOutcome):
        """Apply the learning to the system."""
        try:
            # Update pattern database
            for pattern in learning_outcome.new_patterns_learned:
                if pattern not in self.pattern_database:
                    self.pattern_database[pattern] = {
                        "first_seen": learning_outcome.timestamp,
                        "occurrence_count": 1,
                        "confidence": learning_outcome.confidence_boost,
                        "applied": False
                    }
                else:
                    self.pattern_database[pattern]["occurrence_count"] += 1
                    self.pattern_database[pattern]["confidence"] += learning_outcome.confidence_boost
            
            # Mark as applied
            learning_outcome.applied_to_system = True
            
            # If we have a super intelligence reference, update it
            if self.super_intelligence:
                self._update_super_intelligence(learning_outcome)
            
        except Exception as e:
            logger.error(f"Error applying learning to system: {e}")
    
    def _update_super_intelligence(self, learning_outcome: LearningOutcome):
        """Update the super intelligence with new learning."""
        try:
            # This would integrate with the super intelligence system
            # to apply the learned patterns and improve capabilities
            pass
        except Exception as e:
            logger.error(f"Error updating super intelligence: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of all learning outcomes."""
        return {
            "total_errors_analyzed": len(self.error_database),
            "total_learning_outcomes": len(self.learning_outcomes),
            "patterns_learned": len(self.pattern_database),
            "improvements_applied": len([lo for lo in self.learning_outcomes if lo.applied_to_system]),
            "confidence_improvements": sum(lo.confidence_boost for lo in self.learning_outcomes),
            "error_type_distribution": self._get_error_type_distribution(),
            "learning_focus_areas": self._get_learning_focus_areas(),
            "recent_improvements": self.improvement_history[-10:] if self.improvement_history else []
        }
    
    def _get_error_type_distribution(self) -> Dict[str, int]:
        """Get distribution of error types."""
        distribution = {}
        for error in self.error_database:
            error_type = error.error_type
            distribution[error_type] = distribution.get(error_type, 0) + 1
        return distribution
    
    def _get_learning_focus_areas(self) -> Dict[str, int]:
        """Get distribution of learning focus areas."""
        focus_areas = {}
        for error in self.error_database:
            if error.error_type in self.error_patterns:
                learning_focus = self.error_patterns[error.error_type]["learning_focus"]
                focus_areas[learning_focus] = focus_areas.get(learning_focus, 0) + 1
        return focus_areas
    
    def save_state(self, filepath: str = "error_learning_state.json"):
        """Save the current state to a file."""
        state = {
            "error_database": [asdict(error) for error in self.error_database],
            "learning_outcomes": [asdict(outcome) for outcome in self.learning_outcomes],
            "pattern_database": self.pattern_database,
            "improvement_history": self.improvement_history,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            with open(filepath, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"💾 Error learning state saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save error learning state: {e}")
    
    def load_state(self, filepath: str = "error_learning_state.json"):
        """Load state from a file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    state = json.load(f)
                
                # Restore error database
                self.error_database = [ErrorAnalysis(**error) for error in state.get("error_database", [])]
                
                # Restore learning outcomes
                self.learning_outcomes = [LearningOutcome(**outcome) for outcome in state.get("learning_outcomes", [])]
                
                # Restore other data
                self.pattern_database = state.get("pattern_database", {})
                self.improvement_history = state.get("improvement_history", [])
                
                logger.info(f"📂 Error learning state loaded from {filepath}")
            else:
                logger.info(f"📂 No existing state file found at {filepath}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load error learning state: {e}")

# Factory function
def create_error_learner(super_intelligence=None) -> IntelligentErrorLearner:
    """Create an intelligent error learner instance."""
    return IntelligentErrorLearner(super_intelligence)

# Example usage
if __name__ == "__main__":
    # Create error learner
    error_learner = create_error_learner()
    
    # Example error analysis
    error_message = "zsh: command not found: python"
    context = {
        "current_directory": "ROOT",
        "user": "camdouglas",
        "system": "darwin",
        "command_attempted": "python"
    }
    
    # Analyze the error
    analysis = error_learner.analyze_error(error_message, context)
    
    print(f"🔍 Error Analysis:")
    print(f"   Type: {analysis.error_type}")
    print(f"   Root Cause: {analysis.root_cause}")
    print(f"   Learning Insights: {len(analysis.learning_insights)}")
    print(f"   Improvement Actions: {len(analysis.improvement_actions)}")
    print(f"   Confidence: {analysis.confidence_score:.2f}")
    
    # Get learning summary
    summary = error_learner.get_learning_summary()
    print(f"\n📊 Learning Summary:")
    print(f"   Total Errors: {summary['total_errors_analyzed']}")
    print(f"   Patterns Learned: {summary['patterns_learned']}")
    print(f"   Confidence Boost: {summary['confidence_improvements']:.2f}")
    
    # Save state
    error_learner.save_state()
