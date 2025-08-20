"""
Insight Generator - Generates insights from knowledge and patterns
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import random

@dataclass
class GeneratedInsight:
    """Represents a generated insight."""
    insight_id: str
    content: str
    confidence: float
    novelty_score: float
    supporting_evidence: List[str]
    insight_type: str
    implications: List[str]

class InsightGenerator:
    """Generates insights from integrated knowledge and recognized patterns."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.generated_insights: List[GeneratedInsight] = []
        
        # Insight generation templates
        self.insight_templates = {
            'pattern_connection': [
                "The pattern {} suggests a connection to {}",
                "There appears to be a relationship between {} and {}",
                "{} consistently occurs alongside {}, indicating possible causation"
            ],
            'trend_analysis': [
                "The data shows a {} trend in {}",
                "Over time, {} has been {} consistently",
                "The frequency of {} indicates {}"
            ],
            'contradiction_resolution': [
                "While {} seems to contradict {}, this may be explained by {}",
                "The apparent conflict between {} and {} could be resolved through {}",
                "Different perspectives on {} may explain why {} and {} both seem valid"
            ],
            'synthesis': [
                "Combining insights from {} and {} reveals {}",
                "The integration of {} with {} suggests {}",
                "When we consider {} together with {}, we can conclude {}"
            ],
            'extrapolation': [
                "If {} continues, we might expect {}",
                "Based on {}, the logical next step would be {}",
                "The pattern {} implies that {} is likely"
            ]
        }
        
    def generate_insights(self,
                         knowledge: Dict[str, Any],
                         patterns: Optional[List[Any]] = None,
                         max_insights: int = 5) -> List[GeneratedInsight]:
        """Generate insights from knowledge and patterns."""
        
        patterns = patterns or []
        insights = []
        
        # Generate different types of insights
        insights.extend(self._generate_pattern_insights(knowledge, patterns))
        insights.extend(self._generate_knowledge_synthesis_insights(knowledge))
        insights.extend(self._generate_trend_insights(knowledge, patterns))
        insights.extend(self._generate_contradiction_insights(knowledge))
        insights.extend(self._generate_extrapolation_insights(knowledge, patterns))
        
        # Rank and filter insights
        ranked_insights = self._rank_insights(insights)
        
        # Store generated insights
        selected_insights = ranked_insights[:max_insights]
        self.generated_insights.extend(selected_insights)
        
        return selected_insights
        
    def _generate_pattern_insights(self,
                                  knowledge: Dict[str, Any],
                                  patterns: List[Any]) -> List[GeneratedInsight]:
        """Generate insights from recognized patterns."""
        
        insights = []
        
        for i, pattern in enumerate(patterns):
            if hasattr(pattern, 'pattern_type') and hasattr(pattern, 'elements'):
                # Generate insight about the pattern
                template = random.choice(self.insight_templates['pattern_connection'])
                
                pattern_desc = f"{pattern.pattern_type} pattern"
                elements_desc = ', '.join(str(e) for e in pattern.elements[:3])
                
                content = template.format(pattern_desc, elements_desc)
                
                insight = GeneratedInsight(
                    insight_id=f"pattern_insight_{len(self.generated_insights) + len(insights)}",
                    content=content,
                    confidence=getattr(pattern, 'strength', 0.5),
                    novelty_score=0.6,
                    supporting_evidence=[f"Pattern: {pattern.pattern_type}"],
                    insight_type='pattern_analysis',
                    implications=[f"This pattern may indicate underlying structure in {elements_desc}"]
                )
                
                insights.append(insight)
                
        return insights
        
    def _generate_knowledge_synthesis_insights(self,
                                             knowledge: Dict[str, Any]) -> List[GeneratedInsight]:
        """Generate insights by synthesizing knowledge from different sources."""
        
        insights = []
        
        # Look for knowledge that can be combined
        knowledge_items = list(knowledge.items())
        
        for i in range(min(3, len(knowledge_items))):
            for j in range(i + 1, min(i + 4, len(knowledge_items))):
                key1, value1 = knowledge_items[i]
                key2, value2 = knowledge_items[j]
                
                # Check if these knowledge items can be meaningfully combined
                if self._can_synthesize(key1, key2, value1, value2):
                    template = random.choice(self.insight_templates['synthesis'])
                    
                    content = template.format(
                        self._simplify_key(key1),
                        self._simplify_key(key2),
                        "new understanding about their relationship"
                    )
                    
                    insight = GeneratedInsight(
                        insight_id=f"synthesis_insight_{len(self.generated_insights) + len(insights)}",
                        content=content,
                        confidence=0.7,
                        novelty_score=0.8,
                        supporting_evidence=[f"Knowledge: {key1}", f"Knowledge: {key2}"],
                        insight_type='knowledge_synthesis',
                        implications=["This synthesis reveals new connections in the knowledge base"]
                    )
                    
                    insights.append(insight)
                    
        return insights
        
    def _generate_trend_insights(self,
                                knowledge: Dict[str, Any],
                                patterns: List[Any]) -> List[GeneratedInsight]:
        """Generate insights about trends in the data."""
        
        insights = []
        
        # Look for frequency or sequence patterns that indicate trends
        for pattern in patterns:
            if hasattr(pattern, 'pattern_type') and pattern.pattern_type in ['frequency', 'sequence']:
                template = random.choice(self.insight_templates['trend_analysis'])
                
                if pattern.pattern_type == 'frequency':
                    trend_type = "increasing frequency"
                    subject = ', '.join(str(e) for e in pattern.elements[:2])
                else:
                    trend_type = "sequential"
                    subject = "observed patterns"
                    
                content = template.format(trend_type, subject)
                
                insight = GeneratedInsight(
                    insight_id=f"trend_insight_{len(self.generated_insights) + len(insights)}",
                    content=content,
                    confidence=getattr(pattern, 'strength', 0.5),
                    novelty_score=0.5,
                    supporting_evidence=[f"Pattern frequency: {getattr(pattern, 'frequency', 'unknown')}"],
                    insight_type='trend_analysis',
                    implications=[f"This trend may continue or evolve in the future"]
                )
                
                insights.append(insight)
                
        return insights
        
    def _generate_contradiction_insights(self,
                                       knowledge: Dict[str, Any]) -> List[GeneratedInsight]:
        """Generate insights about contradictions and how to resolve them."""
        
        insights = []
        
        # Look for potential contradictions in knowledge
        knowledge_items = list(knowledge.items())
        
        for i in range(len(knowledge_items)):
            for j in range(i + 1, len(knowledge_items)):
                key1, value1 = knowledge_items[i]
                key2, value2 = knowledge_items[j]
                
                # Simple contradiction detection
                if self._might_be_contradictory(key1, key2, value1, value2):
                    template = random.choice(self.insight_templates['contradiction_resolution'])
                    
                    content = template.format(
                        self._simplify_key(key1),
                        self._simplify_key(key2),
                        "considering different contexts or perspectives"
                    )
                    
                    insight = GeneratedInsight(
                        insight_id=f"contradiction_insight_{len(self.generated_insights) + len(insights)}",
                        content=content,
                        confidence=0.6,
                        novelty_score=0.7,
                        supporting_evidence=[f"Potential contradiction between {key1} and {key2}"],
                        insight_type='contradiction_resolution',
                        implications=["Resolving this contradiction could lead to deeper understanding"]
                    )
                    
                    insights.append(insight)
                    
        return insights
        
    def _generate_extrapolation_insights(self,
                                        knowledge: Dict[str, Any],
                                        patterns: List[Any]) -> List[GeneratedInsight]:
        """Generate insights by extrapolating from current knowledge and patterns."""
        
        insights = []
        
        # Use patterns to make predictions
        for pattern in patterns[:2]:  # Limit to avoid too many insights
            if hasattr(pattern, 'elements') and hasattr(pattern, 'strength'):
                if pattern.strength > 0.6:  # Only extrapolate from strong patterns
                    template = random.choice(self.insight_templates['extrapolation'])
                    
                    pattern_desc = f"the {getattr(pattern, 'pattern_type', 'observed')} pattern"
                    prediction = "similar patterns will emerge in related areas"
                    
                    content = template.format(pattern_desc, prediction)
                    
                    insight = GeneratedInsight(
                        insight_id=f"extrapolation_insight_{len(self.generated_insights) + len(insights)}",
                        content=content,
                        confidence=pattern.strength * 0.8,  # Slightly lower confidence for predictions
                        novelty_score=0.9,  # High novelty for predictions
                        supporting_evidence=[f"Strong pattern: {pattern.pattern_type}"],
                        insight_type='extrapolation',
                        implications=["This prediction could guide future exploration and learning"]
                    )
                    
                    insights.append(insight)
                    
        return insights
        
    def _can_synthesize(self, key1: str, key2: str, value1: Any, value2: Any) -> bool:
        """Check if two knowledge items can be meaningfully synthesized."""
        
        # Simple heuristic: look for shared keywords
        key1_words = set(str(key1).lower().split())
        key2_words = set(str(key2).lower().split())
        
        # Some overlap but not identical
        overlap = len(key1_words.intersection(key2_words))
        total_unique = len(key1_words.union(key2_words))
        
        overlap_ratio = overlap / total_unique if total_unique > 0 else 0
        
        return 0.2 <= overlap_ratio <= 0.7  # Some connection but not too similar
        
    def _might_be_contradictory(self, key1: str, key2: str, value1: Any, value2: Any) -> bool:
        """Check if two knowledge items might be contradictory."""
        
        # Simple contradiction detection
        key1_str = str(key1).lower()
        key2_str = str(key2).lower()
        value1_str = str(value1).lower()
        value2_str = str(value2).lower()
        
        # Look for negation words or opposite terms
        negation_indicators = [
            ('not', ''), ('false', 'true'), ('never', 'always'),
            ('impossible', 'possible'), ('incorrect', 'correct')
        ]
        
        for neg, pos in negation_indicators:
            if ((neg in key1_str or neg in value1_str) and 
                (pos in key2_str or pos in value2_str)):
                return True
            if ((pos in key1_str or pos in value1_str) and 
                (neg in key2_str or neg in value2_str)):
                return True
                
        return False
        
    def _simplify_key(self, key: str) -> str:
        """Simplify a knowledge key for use in insight templates."""
        # Remove common prefixes and simplify
        simplified = str(key).replace('_', ' ').strip()
        
        # Limit length
        if len(simplified) > 30:
            simplified = simplified[:27] + "..."
            
        return simplified
        
    def _rank_insights(self, insights: List[GeneratedInsight]) -> List[GeneratedInsight]:
        """Rank insights by importance and quality."""
        
        def insight_score(insight: GeneratedInsight) -> float:
            score = (
                insight.confidence * 0.3 +
                insight.novelty_score * 0.4 +
                len(insight.supporting_evidence) * 0.1 +
                len(insight.implications) * 0.2
            )
            return score
            
        return sorted(insights, key=insight_score, reverse=True)
        
    def get_insight_summary(self) -> Dict[str, Any]:
        """Get summary of generated insights."""
        
        if not self.generated_insights:
            return {'total_insights': 0}
            
        by_type = {}
        total_confidence = 0.0
        total_novelty = 0.0
        
        for insight in self.generated_insights:
            insight_type = insight.insight_type
            by_type[insight_type] = by_type.get(insight_type, 0) + 1
            total_confidence += insight.confidence
            total_novelty += insight.novelty_score
            
        count = len(self.generated_insights)
        
        return {
            'total_insights': count,
            'insights_by_type': by_type,
            'average_confidence': total_confidence / count,
            'average_novelty': total_novelty / count,
            'most_confident_insight': max(self.generated_insights, key=lambda i: i.confidence).content
        }
