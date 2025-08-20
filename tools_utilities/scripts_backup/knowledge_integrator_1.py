"""
Knowledge Integrator - Integrates knowledge from multiple sources
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, field
from datetime import datetime

@dataclass
class KnowledgeSource:
    """Represents a source of knowledge."""
    source_id: str
    source_type: str
    reliability: float
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class IntegrationResult:
    """Results of knowledge integration."""
    integrated_knowledge: Dict[str, Any]
    conflicts: List[Dict[str, Any]]
    confidence: float
    sources_used: List[str]

class KnowledgeIntegrator:
    """Integrates knowledge from multiple sources and resolves conflicts."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.knowledge_sources: List[KnowledgeSource] = []
        self.integration_history: List[IntegrationResult] = []
        
        # Integration parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.max_sources = self.config.get('max_sources', 10)
        
    def add_knowledge_source(self, 
                           source_id: str,
                           source_type: str,
                           content: Any,
                           reliability: float = 0.5,
                           metadata: Optional[Dict] = None) -> str:
        """Add a new knowledge source."""
        
        source = KnowledgeSource(
            source_id=source_id,
            source_type=source_type,
            reliability=max(0.0, min(1.0, reliability)),
            content=content,
            metadata=metadata or {}
        )
        
        self.knowledge_sources.append(source)
        
        # Keep sources list manageable
        if len(self.knowledge_sources) > self.max_sources * 2:
            self._prune_sources()
            
        return source_id
        
    def integrate_knowledge(self, 
                          topic: str,
                          source_types: Optional[List[str]] = None) -> IntegrationResult:
        """Integrate knowledge about a specific topic."""
        
        # Find relevant sources
        relevant_sources = self._find_relevant_sources(topic, source_types)
        
        if not relevant_sources:
            return IntegrationResult(
                integrated_knowledge={},
                conflicts=[],
                confidence=0.0,
                sources_used=[]
            )
            
        # Extract facts from sources
        facts = []
        for source in relevant_sources:
            source_facts = self._extract_facts(source, topic)
            facts.extend(source_facts)
            
        # Resolve conflicts and integrate
        integrated, conflicts = self._resolve_conflicts(facts)
        
        # Calculate confidence
        confidence = self._calculate_integration_confidence(integrated, relevant_sources)
        
        result = IntegrationResult(
            integrated_knowledge=integrated,
            conflicts=conflicts,
            confidence=confidence,
            sources_used=[s.source_id for s in relevant_sources]
        )
        
        self.integration_history.append(result)
        
        return result
        
    def _find_relevant_sources(self, 
                              topic: str,
                              source_types: Optional[List[str]] = None) -> List[KnowledgeSource]:
        """Find sources relevant to a topic."""
        
        relevant = []
        topic_lower = topic.lower()
        
        for source in self.knowledge_sources:
            # Filter by source type if specified
            if source_types and source.source_type not in source_types:
                continue
                
            # Check if source is relevant to topic
            relevance = self._calculate_relevance(source, topic_lower)
            
            if relevance > 0.3:  # Relevance threshold
                relevant.append(source)
                
        # Sort by reliability and relevance
        relevant.sort(key=lambda s: s.reliability, reverse=True)
        
        return relevant[:self.max_sources]  # Limit number of sources
        
    def _calculate_relevance(self, source: KnowledgeSource, topic: str) -> float:
        """Calculate how relevant a source is to a topic."""
        
        content_str = str(source.content).lower()
        
        # Simple keyword matching
        topic_words = set(topic.split())
        content_words = set(content_str.split())
        
        if not topic_words:
            return 0.0
            
        # Calculate word overlap
        overlap = len(topic_words.intersection(content_words))
        relevance = overlap / len(topic_words)
        
        # Boost relevance for certain source types
        if source.source_type in ['expert', 'primary', 'verified']:
            relevance *= 1.2
            
        return min(1.0, relevance)
        
    def _extract_facts(self, source: KnowledgeSource, topic: str) -> List[Dict[str, Any]]:
        """Extract facts from a knowledge source."""
        
        facts = []
        content = source.content
        
        if isinstance(content, dict):
            # Extract key-value facts from dictionary
            for key, value in content.items():
                if topic.lower() in key.lower() or topic.lower() in str(value).lower():
                    fact = {
                        'statement': f"{key}: {value}",
                        'confidence': source.reliability,
                        'source_id': source.source_id,
                        'type': 'key_value'
                    }
                    facts.append(fact)
                    
        elif isinstance(content, list):
            # Extract facts from list items
            for item in content:
                if topic.lower() in str(item).lower():
                    fact = {
                        'statement': str(item),
                        'confidence': source.reliability,
                        'source_id': source.source_id,
                        'type': 'list_item'
                    }
                    facts.append(fact)
                    
        else:
            # Extract facts from string content
            content_str = str(content)
            sentences = content_str.split('.')
            
            for sentence in sentences:
                if topic.lower() in sentence.lower():
                    fact = {
                        'statement': sentence.strip(),
                        'confidence': source.reliability,
                        'source_id': source.source_id,
                        'type': 'sentence'
                    }
                    facts.append(fact)
                    
        return facts
        
    def _resolve_conflicts(self, facts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Resolve conflicts between facts and integrate knowledge."""
        
        integrated = {}
        conflicts = []
        
        # Group facts by similar statements
        fact_groups = self._group_similar_facts(facts)
        
        for group_key, group_facts in fact_groups.items():
            if len(group_facts) == 1:
                # No conflict, accept the fact
                fact = group_facts[0]
                integrated[group_key] = {
                    'statement': fact['statement'],
                    'confidence': fact['confidence'],
                    'sources': [fact['source_id']]
                }
            else:
                # Multiple facts, resolve conflict
                resolved_fact, conflict_info = self._resolve_fact_conflict(group_facts)
                
                integrated[group_key] = resolved_fact
                
                if conflict_info:
                    conflicts.append(conflict_info)
                    
        return integrated, conflicts
        
    def _group_similar_facts(self, facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group similar facts together."""
        
        groups = {}
        
        for fact in facts:
            statement = fact['statement']
            
            # Find existing group with similar statement
            found_group = None
            for group_key in groups.keys():
                if self._statements_similar(statement, group_key):
                    found_group = group_key
                    break
                    
            if found_group:
                groups[found_group].append(fact)
            else:
                # Create new group
                groups[statement] = [fact]
                
        return groups
        
    def _statements_similar(self, stmt1: str, stmt2: str, threshold: float = 0.7) -> bool:
        """Check if two statements are similar."""
        
        words1 = set(stmt1.lower().split())
        words2 = set(stmt2.lower().split())
        
        if not words1 and not words2:
            return True
        if not words1 or not words2:
            return False
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        return similarity >= threshold
        
    def _resolve_fact_conflict(self, conflicting_facts: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Resolve conflicts between multiple facts."""
        
        # Sort by confidence (reliability)
        sorted_facts = sorted(conflicting_facts, key=lambda f: f['confidence'], reverse=True)
        
        # Use highest confidence fact as base
        best_fact = sorted_facts[0]
        
        # Calculate combined confidence
        confidences = [f['confidence'] for f in conflicting_facts]
        combined_confidence = sum(confidences) / len(confidences)
        
        # Check if facts are actually conflicting or just different perspectives
        statements = [f['statement'] for f in conflicting_facts]
        if self._are_statements_conflicting(statements):
            # True conflict - use weighted average or highest confidence
            resolved = {
                'statement': best_fact['statement'],
                'confidence': combined_confidence * 0.8,  # Reduce confidence due to conflict
                'sources': [f['source_id'] for f in conflicting_facts],
                'resolution_method': 'highest_confidence'
            }
            
            conflict_info = {
                'type': 'conflicting_statements',
                'statements': statements,
                'sources': [f['source_id'] for f in conflicting_facts],
                'resolution': 'used_highest_confidence'
            }
            
        else:
            # Similar facts - combine them
            resolved = {
                'statement': best_fact['statement'],
                'confidence': combined_confidence,
                'sources': [f['source_id'] for f in conflicting_facts],
                'resolution_method': 'combined_similar'
            }
            
            conflict_info = None  # No real conflict
            
        return resolved, conflict_info
        
    def _are_statements_conflicting(self, statements: List[str]) -> bool:
        """Check if statements are actually conflicting."""
        
        # Simple conflict detection based on negation words
        negation_words = ['not', 'no', 'never', 'false', 'incorrect', 'wrong']
        
        has_positive = False
        has_negative = False
        
        for statement in statements:
            statement_lower = statement.lower()
            
            if any(neg in statement_lower for neg in negation_words):
                has_negative = True
            else:
                has_positive = True
                
        return has_positive and has_negative
        
    def _calculate_integration_confidence(self, 
                                        integrated: Dict[str, Any],
                                        sources: List[KnowledgeSource]) -> float:
        """Calculate confidence in the integrated knowledge."""
        
        if not integrated or not sources:
            return 0.0
            
        # Calculate based on source reliability and fact confidence
        total_confidence = 0.0
        fact_count = 0
        
        for fact_info in integrated.values():
            if isinstance(fact_info, dict) and 'confidence' in fact_info:
                total_confidence += fact_info['confidence']
                fact_count += 1
                
        avg_fact_confidence = total_confidence / fact_count if fact_count > 0 else 0.0
        
        # Average source reliability
        avg_source_reliability = sum(s.reliability for s in sources) / len(sources)
        
        # Combine factors
        integration_confidence = (avg_fact_confidence + avg_source_reliability) / 2.0
        
        return min(1.0, integration_confidence)
        
    def _prune_sources(self):
        """Remove old or low-reliability sources to maintain performance."""
        
        # Sort by reliability and timestamp
        self.knowledge_sources.sort(key=lambda s: (s.reliability, s.timestamp), reverse=True)
        
        # Keep top sources
        self.knowledge_sources = self.knowledge_sources[:self.max_sources]
        
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge integration activity."""
        
        if not self.integration_history:
            return {'total_integrations': 0}
            
        total_integrations = len(self.integration_history)
        avg_confidence = sum(r.confidence for r in self.integration_history) / total_integrations
        total_conflicts = sum(len(r.conflicts) for r in self.integration_history)
        
        return {
            'total_integrations': total_integrations,
            'average_confidence': avg_confidence,
            'total_conflicts': total_conflicts,
            'total_sources': len(self.knowledge_sources),
            'source_types': list(set(s.source_type for s in self.knowledge_sources))
        }
