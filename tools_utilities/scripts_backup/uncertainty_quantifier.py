"""
Uncertainty Quantifier - Quantifies and manages uncertainty in learning
"""

from typing import Dict, List, Any, Optional, Tuple
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass
from enum import Enum
import math

class UncertaintyType(Enum):
    """Types of uncertainty in learning systems"""
    EPISTEMIC = "epistemic"  # Uncertainty due to lack of knowledge
    ALEATORIC = "aleatoric"  # Uncertainty due to randomness/noise
    MODEL = "model"          # Uncertainty in model predictions
    DATA = "data"            # Uncertainty in data quality/completeness
    CAUSAL = "causal"        # Uncertainty in causal relationships

@dataclass
class UncertaintyEstimate:
    """Represents an uncertainty estimate"""
    value: float  # Uncertainty magnitude [0, 1]
    confidence: float  # Confidence in the uncertainty estimate
    uncertainty_type: UncertaintyType
    evidence: Dict[str, Any]
    
class UncertaintyQuantifier:
    """
    Quantifies and manages different types of uncertainty in learning.
    
    Helps the curiosity system understand what we don't know and how
    confident we should be in our assessments.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Uncertainty tracking
        self.uncertainty_history: List[UncertaintyEstimate] = []
        self.knowledge_confidence: Dict[str, float] = {}
        self.prediction_errors: List[float] = []
        
        # Thresholds
        self.high_uncertainty_threshold = self.config.get('high_uncertainty_threshold', 0.7)
        self.low_uncertainty_threshold = self.config.get('low_uncertainty_threshold', 0.3)
        
    def quantify_uncertainty(self, 
                           observation: Any,
                           context: Optional[Dict] = None,
                           prediction: Optional[Any] = None) -> UncertaintyEstimate:
        """
        Quantify uncertainty about an observation or prediction.
        
        Args:
            observation: The thing we're uncertain about
            context: Additional context
            prediction: Optional prediction to evaluate uncertainty of
            
        Returns:
            UncertaintyEstimate with uncertainty level and type
        """
        context = context or {}
        
        # Determine primary uncertainty type
        uncertainty_type = self._determine_uncertainty_type(observation, context, prediction)
        
        # Calculate uncertainty based on type
        if uncertainty_type == UncertaintyType.EPISTEMIC:
            uncertainty = self._calculate_epistemic_uncertainty(observation, context)
        elif uncertainty_type == UncertaintyType.ALEATORIC:
            uncertainty = self._calculate_aleatoric_uncertainty(observation, context)
        elif uncertainty_type == UncertaintyType.MODEL:
            uncertainty = self._calculate_model_uncertainty(observation, prediction, context)
        elif uncertainty_type == UncertaintyType.DATA:
            uncertainty = self._calculate_data_uncertainty(observation, context)
        elif uncertainty_type == UncertaintyType.CAUSAL:
            uncertainty = self._calculate_causal_uncertainty(observation, context)
        else:
            uncertainty = 0.5  # Default moderate uncertainty
            
        # Calculate confidence in this uncertainty estimate
        confidence = self._calculate_confidence(observation, uncertainty_type, context)
        
        # Gather evidence for the uncertainty estimate
        evidence = self._gather_evidence(observation, uncertainty_type, context)
        
        estimate = UncertaintyEstimate(
            value=uncertainty,
            confidence=confidence,
            uncertainty_type=uncertainty_type,
            evidence=evidence
        )
        
        # Record the estimate
        self.uncertainty_history.append(estimate)
        
        # Maintain history size
        max_history = self.config.get('max_history', 1000)
        if len(self.uncertainty_history) > max_history:
            self.uncertainty_history = self.uncertainty_history[-max_history//2:]
            
        return estimate
        
    def update_with_outcome(self, 
                           original_estimate: UncertaintyEstimate,
                           actual_outcome: Any,
                           predicted_outcome: Optional[Any] = None):
        """
        Update uncertainty models based on actual outcomes.
        
        This helps improve future uncertainty estimates through learning.
        """
        # Calculate prediction error if we had a prediction
        if predicted_outcome is not None:
            error = self._calculate_prediction_error(predicted_outcome, actual_outcome)
            self.prediction_errors.append(error)
            
            # Update confidence for this type of uncertainty
            observation_key = str(original_estimate.uncertainty_type.value)
            current_confidence = self.knowledge_confidence.get(observation_key, 0.5)
            
            # If our uncertainty was well-calibrated, increase confidence
            expected_error = original_estimate.value
            actual_error = error
            
            calibration_quality = 1.0 - abs(expected_error - actual_error)
            new_confidence = 0.7 * current_confidence + 0.3 * calibration_quality
            
            self.knowledge_confidence[observation_key] = new_confidence
            
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of current uncertainty state."""
        if not self.uncertainty_history:
            return {'status': 'no_data'}
            
        recent_estimates = self.uncertainty_history[-20:]
        
        # Calculate averages by type
        type_uncertainties = {}
        type_counts = {}
        
        for estimate in recent_estimates:
            uncertainty_type = estimate.uncertainty_type.value
            type_uncertainties[uncertainty_type] = (
                type_uncertainties.get(uncertainty_type, 0) + estimate.value
            )
            type_counts[uncertainty_type] = type_counts.get(uncertainty_type, 0) + 1
            
        avg_uncertainties = {
            utype: total / type_counts[utype]
            for utype, total in type_uncertainties.items()
        }
        
        overall_uncertainty = sum(est.value for est in recent_estimates) / len(recent_estimates)
        overall_confidence = sum(est.confidence for est in recent_estimates) / len(recent_estimates)
        
        return {
            'overall_uncertainty': overall_uncertainty,
            'overall_confidence': overall_confidence,
            'uncertainty_by_type': avg_uncertainties,
            'high_uncertainty_areas': [
                utype for utype, uncertainty in avg_uncertainties.items()
                if uncertainty > self.high_uncertainty_threshold
            ],
            'low_uncertainty_areas': [
                utype for utype, uncertainty in avg_uncertainties.items()
                if uncertainty < self.low_uncertainty_threshold
            ],
            'prediction_accuracy': self._calculate_prediction_accuracy(),
            'total_estimates': len(self.uncertainty_history)
        }
        
    def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify areas with high uncertainty that represent knowledge gaps."""
        gaps = []
        
        if len(self.uncertainty_history) < 5:
            return gaps
            
        # Group by uncertainty type and find high-uncertainty areas
        type_groups = {}
        for estimate in self.uncertainty_history[-50:]:  # Recent estimates
            utype = estimate.uncertainty_type.value
            if utype not in type_groups:
                type_groups[utype] = []
            type_groups[utype].append(estimate)
            
        for uncertainty_type, estimates in type_groups.items():
            avg_uncertainty = sum(est.value for est in estimates) / len(estimates)
            avg_confidence = sum(est.confidence for est in estimates) / len(estimates)
            
            if avg_uncertainty > self.high_uncertainty_threshold:
                gap = {
                    'type': uncertainty_type,
                    'uncertainty_level': avg_uncertainty,
                    'confidence': avg_confidence,
                    'frequency': len(estimates),
                    'priority': avg_uncertainty * avg_confidence,  # High uncertainty + high confidence = priority
                    'examples': [est.evidence for est in estimates[-3:]]  # Recent examples
                }
                gaps.append(gap)
                
        # Sort by priority (high uncertainty + high confidence)
        gaps.sort(key=lambda x: x['priority'], reverse=True)
        
        return gaps
        
    def _determine_uncertainty_type(self, 
                                   observation: Any,
                                   context: Dict,
                                   prediction: Optional[Any]) -> UncertaintyType:
        """Determine the primary type of uncertainty for this situation."""
        
        # Check context for explicit type hints
        if 'uncertainty_type' in context:
            try:
                return UncertaintyType(context['uncertainty_type'])
            except ValueError:
                pass
                
        # Infer from observation characteristics
        obs_str = str(observation).lower()
        
        # Model uncertainty - when we have predictions
        if prediction is not None:
            return UncertaintyType.MODEL
            
        # Data uncertainty - when dealing with data quality issues
        if any(keyword in obs_str for keyword in ['missing', 'incomplete', 'noisy', 'corrupted']):
            return UncertaintyType.DATA
            
        # Causal uncertainty - when dealing with causal relationships
        if any(keyword in obs_str for keyword in ['causes', 'effect', 'because', 'leads to', 'results in']):
            return UncertaintyType.CAUSAL
            
        # Aleatoric uncertainty - when dealing with randomness
        if any(keyword in obs_str for keyword in ['random', 'stochastic', 'probability', 'chance']):
            return UncertaintyType.ALEATORIC
            
        # Default to epistemic uncertainty (lack of knowledge)
        return UncertaintyType.EPISTEMIC
        
    def _calculate_epistemic_uncertainty(self, observation: Any, context: Dict) -> float:
        """Calculate uncertainty due to lack of knowledge."""
        obs_str = str(observation)
        
        # Factors that increase epistemic uncertainty:
        # 1. Novel concepts not seen before
        # 2. Complex observations
        # 3. Lack of similar examples in history
        
        # Novelty factor
        novelty = self._calculate_novelty_factor(obs_str)
        
        # Complexity factor
        complexity = self._calculate_complexity_factor(obs_str)
        
        # Historical familiarity factor
        familiarity = self._calculate_familiarity_factor(obs_str)
        
        # Combine factors
        epistemic_uncertainty = (
            0.4 * novelty +
            0.3 * complexity +
            0.3 * (1.0 - familiarity)  # Less familiarity = more uncertainty
        )
        
        return max(0.0, min(1.0, epistemic_uncertainty))
        
    def _calculate_aleatoric_uncertainty(self, observation: Any, context: Dict) -> float:
        """Calculate uncertainty due to inherent randomness."""
        # For aleatoric uncertainty, look for indicators of randomness or noise
        
        obs_str = str(observation).lower()
        
        # Keyword-based detection of randomness
        randomness_keywords = ['random', 'stochastic', 'noise', 'variable', 'uncertain']
        randomness_score = sum(1 for keyword in randomness_keywords if keyword in obs_str)
        randomness_factor = min(randomness_score / len(randomness_keywords), 1.0)
        
        # Structural indicators (high variability, inconsistency)
        if hasattr(observation, '__iter__') and not isinstance(observation, str):
            try:
                # For sequences, calculate variability
                values = list(observation)
                if len(values) > 1 and all(isinstance(v, (int, float)) for v in values):
                    mean_val = sum(values) / len(values)
                    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                    cv = math.sqrt(variance) / (abs(mean_val) + 1e-8)  # Coefficient of variation
                    variability_factor = min(cv, 1.0)
                else:
                    variability_factor = 0.3  # Default for non-numeric sequences
            except:
                variability_factor = 0.3
        else:
            variability_factor = 0.3
            
        aleatoric_uncertainty = (randomness_factor + variability_factor) / 2.0
        
        return max(0.0, min(1.0, aleatoric_uncertainty))
        
    def _calculate_model_uncertainty(self, 
                                   observation: Any,
                                   prediction: Any,
                                   context: Dict) -> float:
        """Calculate uncertainty in model predictions."""
        if prediction is None:
            return 0.8  # High uncertainty when no prediction available
            
        # Factors affecting model uncertainty:
        # 1. Distance from ml_architecture.training_pipelines data
        # 2. Model confidence (if available)
        # 3. Prediction consistency
        
        # Simple distance-based uncertainty
        obs_str = str(observation)
        pred_str = str(prediction)
        
        # Calculate similarity between observation and prediction
        similarity = self._calculate_text_similarity(obs_str, pred_str)
        
        # Lower similarity might indicate model uncertainty
        uncertainty_from_similarity = 1.0 - similarity
        
        # Check for explicit confidence in context
        model_confidence = context.get('model_confidence', 0.5)
        uncertainty_from_confidence = 1.0 - model_confidence
        
        # Combine uncertainty sources
        model_uncertainty = (uncertainty_from_similarity + uncertainty_from_confidence) / 2.0
        
        return max(0.0, min(1.0, model_uncertainty))
        
    def _calculate_data_uncertainty(self, observation: Any, context: Dict) -> float:
        """Calculate uncertainty due to data quality issues."""
        obs_str = str(observation).lower()
        
        # Data quality indicators
        quality_issues = [
            'missing', 'null', 'none', 'unknown', 'incomplete',
            'corrupted', 'error', 'invalid', 'nan', 'empty'
        ]
        
        issue_count = sum(1 for issue in quality_issues if issue in obs_str)
        quality_uncertainty = min(issue_count / len(quality_issues), 1.0)
        
        # Completeness check
        completeness = context.get('data_completeness', 1.0)
        completeness_uncertainty = 1.0 - completeness
        
        # Consistency check
        consistency = context.get('data_consistency', 1.0)
        consistency_uncertainty = 1.0 - consistency
        
        data_uncertainty = (
            0.4 * quality_uncertainty +
            0.3 * completeness_uncertainty +
            0.3 * consistency_uncertainty
        )
        
        return max(0.0, min(1.0, data_uncertainty))
        
    def _calculate_causal_uncertainty(self, observation: Any, context: Dict) -> float:
        """Calculate uncertainty in causal relationships."""
        obs_str = str(observation).lower()
        
        # Causal complexity indicators
        causal_keywords = ['causes', 'because', 'leads to', 'results in', 'due to', 'affects']
        causal_mentions = sum(1 for keyword in causal_keywords if keyword in obs_str)
        
        # More causal relationships mentioned = higher complexity = higher uncertainty
        complexity_uncertainty = min(causal_mentions / 10.0, 1.0)
        
        # Confounding factors
        confounding_keywords = ['however', 'but', 'although', 'despite', 'except']
        confounding_mentions = sum(1 for keyword in confounding_keywords if keyword in obs_str)
        confounding_uncertainty = min(confounding_mentions / 5.0, 1.0)
        
        # Temporal factors
        temporal_keywords = ['before', 'after', 'during', 'while', 'when']
        temporal_mentions = sum(1 for keyword in temporal_keywords if keyword in obs_str)
        temporal_complexity = min(temporal_mentions / 5.0, 1.0)
        
        causal_uncertainty = (
            0.4 * complexity_uncertainty +
            0.3 * confounding_uncertainty +
            0.3 * temporal_complexity
        )
        
        return max(0.0, min(1.0, causal_uncertainty))
        
    def _calculate_confidence(self, 
                            observation: Any,
                            uncertainty_type: UncertaintyType,
                            context: Dict) -> float:
        """Calculate confidence in the uncertainty estimate."""
        
        # Base confidence from historical performance
        type_key = uncertainty_type.value
        base_confidence = self.knowledge_confidence.get(type_key, 0.5)
        
        # Adjust based on observation characteristics
        obs_str = str(observation)
        
        # More detailed observations -> higher confidence
        detail_factor = min(len(obs_str) / 100.0, 1.0)
        
        # Context information -> higher confidence
        context_factor = min(len(context) / 10.0, 1.0)
        
        # Historical experience with this uncertainty type
        type_experience = sum(
            1 for est in self.uncertainty_history
            if est.uncertainty_type == uncertainty_type
        )
        experience_factor = min(type_experience / 20.0, 1.0)
        
        confidence = (
            0.4 * base_confidence +
            0.2 * detail_factor +
            0.2 * context_factor +
            0.2 * experience_factor
        )
        
        return max(0.1, min(1.0, confidence))
        
    def _gather_evidence(self, 
                        observation: Any,
                        uncertainty_type: UncertaintyType,
                        context: Dict) -> Dict[str, Any]:
        """Gather evidence supporting the uncertainty estimate."""
        
        evidence = {
            'observation_length': len(str(observation)),
            'context_keys': list(context.keys()),
            'uncertainty_type': uncertainty_type.value,
            'timestamp': len(self.uncertainty_history)
        }
        
        # Type-specific evidence
        if uncertainty_type == UncertaintyType.EPISTEMIC:
            evidence['novelty_indicators'] = self._find_novelty_indicators(observation)
        elif uncertainty_type == UncertaintyType.DATA:
            evidence['quality_indicators'] = self._find_quality_indicators(observation)
        elif uncertainty_type == UncertaintyType.CAUSAL:
            evidence['causal_indicators'] = self._find_causal_indicators(observation)
            
        return evidence
        
    def _calculate_novelty_factor(self, obs_str: str) -> float:
        """Calculate novelty factor for epistemic uncertainty."""
        if not self.uncertainty_history:
            return 1.0
            
        # Compare with historical observations
        similarities = []
        for past_estimate in self.uncertainty_history[-20:]:
            past_obs = str(past_estimate.evidence.get('observation', ''))
            similarity = self._calculate_text_similarity(obs_str, past_obs)
            similarities.append(similarity)
            
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        return max(0.0, min(1.0, novelty))
        
    def _calculate_complexity_factor(self, obs_str: str) -> float:
        """Calculate complexity factor for epistemic uncertainty."""
        # Length-based complexity
        length_complexity = min(len(obs_str) / 500.0, 1.0)
        
        # Vocabulary diversity
        words = obs_str.split()
        unique_words = len(set(words))
        vocab_complexity = min(unique_words / max(len(words), 1), 1.0) if words else 0.0
        
        complexity = (length_complexity + vocab_complexity) / 2.0
        return max(0.0, min(1.0, complexity))
        
    def _calculate_familiarity_factor(self, obs_str: str) -> float:
        """Calculate familiarity factor based on historical observations."""
        if not self.uncertainty_history:
            return 0.0
            
        # Calculate average similarity with historical observations
        similarities = []
        for past_estimate in self.uncertainty_history[-10:]:
            past_obs = str(past_estimate.evidence.get('observation', ''))
            similarity = self._calculate_text_similarity(obs_str, past_obs)
            similarities.append(similarity)
            
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return max(0.0, min(1.0, avg_similarity))
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
            
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_prediction_error(self, predicted: Any, actual: Any) -> float:
        """Calculate error between prediction and actual outcome."""
        # Simple string-based error calculation
        pred_str = str(predicted)
        actual_str = str(actual)
        
        similarity = self._calculate_text_similarity(pred_str, actual_str)
        error = 1.0 - similarity
        
        return max(0.0, min(1.0, error))
        
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate overall prediction accuracy from error history."""
        if not self.prediction_errors:
            return 0.5  # Unknown accuracy
            
        avg_error = sum(self.prediction_errors) / len(self.prediction_errors)
        accuracy = 1.0 - avg_error
        
        return max(0.0, min(1.0, accuracy))
        
    def _find_novelty_indicators(self, observation: Any) -> List[str]:
        """Find indicators of novelty in observation."""
        obs_str = str(observation).lower()
        novelty_keywords = ['new', 'novel', 'unprecedented', 'unique', 'unusual', 'unexpected']
        
        return [keyword for keyword in novelty_keywords if keyword in obs_str]
        
    def _find_quality_indicators(self, observation: Any) -> List[str]:
        """Find data quality indicators in observation."""
        obs_str = str(observation).lower()
        quality_keywords = ['missing', 'incomplete', 'corrupted', 'error', 'invalid', 'null']
        
        return [keyword for keyword in quality_keywords if keyword in obs_str]
        
    def _find_causal_indicators(self, observation: Any) -> List[str]:
        """Find causal relationship indicators in observation."""
        obs_str = str(observation).lower()
        causal_keywords = ['causes', 'because', 'leads to', 'results in', 'due to', 'affects']
        
        return [keyword for keyword in causal_keywords if keyword in obs_str]
