"""
Intelligent Feedback Collection System

Automatically collects feedback and metrics to improve the agent hub's intelligence
through continuous learning and optimization.
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class IntelligentFeedbackCollector:
    """Intelligent feedback collection for continuous improvement."""
    
    def __init__(self, training_manager=None):
        self.training_manager = training_manager
        self.feedback_cache = {}
        self.quality_metrics = {}
        
        # Initialize quality assessment models
        self._init_quality_models()
    
    def _init_quality_models(self):
        """Initialize quality assessment models."""
        # Simple heuristics for quality assessment
        self.quality_patterns = {
            "code_quality": [
                r"def\s+\w+\s*\([^)]*\)\s*:",  # Function definitions
                r"class\s+\w+\s*:",  # Class definitions
                r"import\s+\w+",  # Imports
                r"try:\s*\n\s*.*\nexcept",  # Error handling
                r"if\s+__name__\s*==\s*['\"]__main__['\"]",  # Main guard
            ],
            "explanation_quality": [
                r"\d+\.\s+\w+",  # Numbered lists
                r"first|second|third|finally",  # Sequential indicators
                r"because|therefore|however|although",  # Logical connectors
                r"example|instance|case",  # Examples
                r"step|phase|stage",  # Process indicators
            ],
            "actionability": [
                r"run|execute|install|configure|setup",  # Action verbs
                r"command|script|code|example",  # Executable content
                r"here's how|follow these steps|do this",  # Instructions
                r"copy|paste|type|enter",  # Direct actions
            ]
        }
    
    def collect_execution_feedback(self, 
                                  run_result: Dict[str, Any],
                                  user_prompt: str,
                                  model_id: str,
                                  execution_metrics: Dict[str, Any]) -> str:
        """
        Collect feedback from execution results automatically.
        
        Args:
            run_result: Result from model execution
            user_prompt: Original user prompt
            model_id: ID of the model used
            execution_metrics: Performance metrics
            
        Returns:
            Feedback ID
        """
        # Analyze response quality automatically
        response_quality = self._analyze_response_quality(run_result)
        
        # Estimate user satisfaction based on quality metrics
        estimated_rating = self._estimate_user_satisfaction(response_quality, execution_metrics)
        
        # Generate automatic feedback
        auto_feedback = self._generate_auto_feedback(response_quality, execution_metrics)
        
        # Collect feedback if training manager is available
        if self.training_manager:
            feedback_id = self.training_manager.collect_feedback(
                prompt=user_prompt,
                response=run_result.get("result", {}).get("stdout", ""),
                model_id=model_id,
                user_rating=estimated_rating,
                user_feedback=auto_feedback,
                execution_metrics=execution_metrics
            )
            
            # Also collect execution metrics
            self.training_manager.collect_execution_metrics(
                run_id=run_result.get("run_dir", "unknown"),
                model_id=model_id,
                prompt=user_prompt,
                execution_time=execution_metrics.get("execution_time", 0),
                resource_usage=execution_metrics.get("resource_usage", {}),
                success=run_result.get("result", {}).get("rc", 1) == 0,
                error_message=run_result.get("result", {}).get("stderr", "")
            )
            
            return feedback_id
        
        return "no_training_manager"
    
    def _analyze_response_quality(self, run_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of the model response."""
        result = run_result.get("result", {})
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        return_code = result.get("rc", 1)
        
        quality_metrics = {
            "success": return_code == 0,
            "has_output": bool(stdout.strip()),
            "has_errors": bool(stderr.strip()),
            "output_length": len(stdout),
            "error_count": len(stderr.split('\n')) if stderr else 0,
            "code_quality": self._assess_code_quality(stdout),
            "explanation_quality": self._assess_explanation_quality(stdout),
            "actionability": self._assess_actionability(stdout),
            "completeness": self._assess_completeness(stdout, stderr),
            "clarity": self._assess_clarity(stdout)
        }
        
        return quality_metrics
    
    def _assess_code_quality(self, text: str) -> float:
        """Assess the quality of code in the response."""
        import re
        
        score = 0.0
        total_patterns = len(self.quality_patterns["code_quality"])
        
        for pattern in self.quality_patterns["code_quality"]:
            if re.search(pattern, text, re.MULTILINE):
                score += 1.0
        
        # Bonus for proper formatting
        if "    " in text or "\t" in text:  # Indentation
            score += 0.5
        
        if "def " in text and "class " in text:  # Both functions and classes
            score += 0.5
        
        return min(score / total_patterns, 1.0)
    
    def _assess_explanation_quality(self, text: str) -> float:
        """Assess the quality of explanations in the response."""
        import re
        
        score = 0.0
        total_patterns = len(self.quality_patterns["explanation_quality"])
        
        for pattern in self.quality_patterns["explanation_quality"]:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1.0
        
        # Bonus for structured explanations
        if any(indicator in text.lower() for indicator in ["1.", "2.", "3.", "step", "phase"]):
            score += 0.5
        
        if any(connector in text.lower() for connector in ["because", "therefore", "however"]):
            score += 0.5
        
        return min(score / total_patterns, 1.0)
    
    def _assess_actionability(self, text: str) -> float:
        """Assess how actionable the response is."""
        import re
        
        score = 0.0
        total_patterns = len(self.quality_patterns["actionability"])
        
        for pattern in self.quality_patterns["actionability"]:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1.0
        
        # Bonus for specific instructions
        if any(word in text.lower() for word in ["copy", "paste", "type", "enter"]):
            score += 0.5
        
        if any(word in text.lower() for word in ["here's how", "follow these steps"]):
            score += 0.5
        
        return min(score / total_patterns, 1.0)
    
    def _assess_completeness(self, stdout: str, stderr: str) -> float:
        """Assess the completeness of the response."""
        score = 0.0
        
        # Has substantial output
        if len(stdout) > 100:
            score += 0.3
        elif len(stdout) > 50:
            score += 0.2
        elif len(stdout) > 10:
            score += 0.1
        
        # No errors or errors are informative
        if not stderr:
            score += 0.3
        elif "error" in stderr.lower() and len(stderr) > 20:
            score += 0.2  # Informative error
        
        # Has multiple components (code + explanation)
        has_code = any(keyword in stdout.lower() for keyword in ["def ", "class ", "import "])
        has_explanation = len(stdout.split()) > 20
        
        if has_code and has_explanation:
            score += 0.4
        elif has_code or has_explanation:
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_clarity(self, text: str) -> float:
        """Assess the clarity of the response."""
        score = 0.0
        
        # Sentence structure
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
        
        if avg_sentence_length < 20:
            score += 0.3
        elif avg_sentence_length < 30:
            score += 0.2
        else:
            score += 0.1
        
        # Paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2
        
        # Use of examples
        if any(word in text.lower() for word in ["example", "instance", "case"]):
            score += 0.2
        
        # Use of formatting
        if any(char in text for char in ["*", "-", "•", "1.", "2."]):
            score += 0.3
        
        return min(score, 1.0)
    
    def _estimate_user_satisfaction(self, quality_metrics: Dict[str, Any], 
                                  execution_metrics: Dict[str, Any]) -> int:
        """Estimate user satisfaction based on quality metrics."""
        base_score = 3  # Neutral starting point
        
        # Quality adjustments
        if quality_metrics["success"]:
            base_score += 1
        
        if quality_metrics["has_output"] and not quality_metrics["has_errors"]:
            base_score += 1
        
        # Code quality bonus
        if quality_metrics["code_quality"] > 0.7:
            base_score += 0.5
        
        # Explanation quality bonus
        if quality_metrics["explanation_quality"] > 0.7:
            base_score += 0.5
        
        # Actionability bonus
        if quality_metrics["actionability"] > 0.7:
            base_score += 0.5
        
        # Performance adjustments
        execution_time = execution_metrics.get("execution_time", 0)
        if execution_time < 5:  # Fast response
            base_score += 0.5
        elif execution_time > 30:  # Slow response
            base_score -= 0.5
        
        # Resource usage adjustments
        resource_usage = execution_metrics.get("resource_usage", {})
        memory_usage = resource_usage.get("memory_mb", 0)
        if memory_usage > 1000:  # High memory usage
            base_score -= 0.5
        
        # Clamp to 1-5 range
        return max(1, min(5, int(round(base_score))))
    
    def _generate_auto_feedback(self, quality_metrics: Dict[str, Any], 
                               execution_metrics: Dict[str, Any]) -> str:
        """Generate automatic feedback based on quality metrics."""
        feedback_parts = []
        
        # Success feedback
        if quality_metrics["success"]:
            feedback_parts.append("Execution successful")
        else:
            feedback_parts.append("Execution failed")
        
        # Quality feedback
        if quality_metrics["code_quality"] > 0.8:
            feedback_parts.append("High code quality")
        elif quality_metrics["code_quality"] < 0.3:
            feedback_parts.append("Low code quality")
        
        if quality_metrics["explanation_quality"] > 0.8:
            feedback_parts.append("Clear explanations")
        elif quality_metrics["explanation_quality"] < 0.3:
            feedback_parts.append("Unclear explanations")
        
        # Performance feedback
        execution_time = execution_metrics.get("execution_time", 0)
        if execution_time < 5:
            feedback_parts.append("Fast execution")
        elif execution_time > 30:
            feedback_parts.append("Slow execution")
        
        # Resource feedback
        resource_usage = execution_metrics.get("resource_usage", {})
        memory_usage = resource_usage.get("memory_mb", 0)
        if memory_usage > 1000:
            feedback_parts.append("High memory usage")
        
        return "; ".join(feedback_parts) if feedback_parts else "Standard execution"
    
    def get_quality_report(self, run_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        quality_metrics = self._analyze_response_quality(run_result)
        
        # Calculate overall quality score
        quality_scores = [
            quality_metrics["code_quality"],
            quality_metrics["explanation_quality"],
            quality_metrics["actionability"],
            quality_metrics["completeness"],
            quality_metrics["clarity"]
        ]
        
        overall_score = sum(quality_scores) / len(quality_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_metrics)
        
        return {
            "overall_score": overall_score,
            "quality_metrics": quality_metrics,
            "recommendations": recommendations,
            "estimated_rating": self._estimate_user_satisfaction(quality_metrics, {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on quality metrics."""
        recommendations = []
        
        if quality_metrics["code_quality"] < 0.5:
            recommendations.append("Include more structured code examples")
        
        if quality_metrics["explanation_quality"] < 0.5:
            recommendations.append("Provide clearer step-by-step explanations")
        
        if quality_metrics["actionability"] < 0.5:
            recommendations.append("Make responses more actionable with specific steps")
        
        if quality_metrics["completeness"] < 0.5:
            recommendations.append("Ensure responses cover all aspects of the request")
        
        if quality_metrics["clarity"] < 0.5:
            recommendations.append("Improve response structure and formatting")
        
        if not recommendations:
            recommendations.append("Response quality is good, maintain current standards")
        
        return recommendations

# Factory function
def create_feedback_collector(training_manager=None) -> IntelligentFeedbackCollector:
    """Create an intelligent feedback collector instance."""
    return IntelligentFeedbackCollector(training_manager)
