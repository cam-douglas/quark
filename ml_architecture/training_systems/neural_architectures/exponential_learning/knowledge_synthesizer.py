#!/usr/bin/env python3
"""
Knowledge Synthesizer for Exponential Learning System
Combines research findings into coherent knowledge structures
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass
from datetime import datetime
import json
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SynthesizedKnowledge:
    """Represents synthesized knowledge from multiple sources"""
    topic: str
    core_concepts: List[str]
    definitions: Dict[str, str]
    relationships: List[Dict[str, Any]]
    insights: List[str]
    confidence: float
    sources: List[str]
    timestamp: datetime
    complexity_score: float

class KnowledgeSynthesizer:
    """
    Synthesizes knowledge from multiple research sources
    Creates coherent knowledge structures and identifies patterns
    """
    
    def __init__(self):
        self.knowledge_graph = defaultdict(dict)
        self.concept_relationships = defaultdict(list)
        self.synthesis_patterns = []
        self.confidence_threshold = 0.7
        
    async def synthesize_research_findings(self, research_results: Dict[str, List[Any]]) -> SynthesizedKnowledge:
        """Synthesize findings from multiple research sources"""
        logger.info("🔬 Starting knowledge synthesis...")
        
        # Extract all concepts and definitions
        all_concepts = set()
        all_definitions = {}
        all_relationships = []
        all_insights = []
        source_count = defaultdict(int)
        
        for source, results in research_results.items():
            for result in results:
                # Count source usage
                source_count[source] += 1
                
                # Extract concepts
                if hasattr(result, 'content'):
                    concepts = self.extract_concepts_from_text(result.content)
                    all_concepts.update(concepts)
                
                # Extract definitions
                if hasattr(result, 'metadata') and 'definition' in result.metadata:
                    all_definitions[result.metadata.get('word', 'unknown')] = result.metadata['definition']
                
                # Extract relationships
                if hasattr(result, 'connections'):
                    all_relationships.extend(result.connections)
                
                # Extract insights
                if hasattr(result, 'content'):
                    insights = self.extract_insights_from_text(result.content)
                    all_insights.extend(insights)
        
        # Synthesize core concepts
        core_concepts = self.identify_core_concepts(all_concepts, source_count)
        
        # Build relationships
        relationships = self.build_knowledge_relationships(core_concepts, all_relationships)
        
        # Generate insights
        insights = self.generate_synthetic_insights(core_concepts, all_insights)
        
        # Calculate confidence
        confidence = self.calculate_synthesis_confidence(source_count, len(core_concepts))
        
        # Calculate complexity
        complexity = self.calculate_knowledge_complexity(core_concepts, relationships)
        
        # Determine topic from concepts
        topic = self.determine_topic(core_concepts)
        
        synthesized = SynthesizedKnowledge(
            topic=topic,
            core_concepts=list(core_concepts),
            definitions=all_definitions,
            relationships=relationships,
            insights=insights,
            confidence=confidence,
            sources=list(source_count.keys()),
            timestamp=datetime.now(),
            complexity_score=complexity
        )
        
        logger.info(f"✅ Synthesized knowledge: {len(core_concepts)} concepts, {len(relationships)} relationships")
        return synthesized
    
    def extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Enhanced concept extraction
        concepts = []
        
        # Look for capitalized phrases (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend(proper_nouns)
        
        # Look for technical terms in quotes
        quoted_terms = re.findall(r'"([^"]+)"', text)
        concepts.extend(quoted_terms)
        
        # Look for terms after "is a" or "refers to"
        definition_terms = re.findall(r'(?:is\s+a|refers\s+to)\s+([^.]*?)(?:\.|,|;|$)', text, re.IGNORECASE)
        concepts.extend(definition_terms)
        
        # Remove duplicates and clean
        concepts = list(set([c.strip() for c in concepts if len(c.strip()) > 2]))
        return concepts[:30]  # Limit to top 30
    
    def extract_insights_from_text(self, text: str) -> List[str]:
        """Extract insights and implications from text"""
        insights = []
        
        # Look for insight patterns
        patterns = [
            r'(?:this\s+means|this\s+suggests|this\s+implies|therefore|consequently)\s+([^.]*?)(?:\.|,|;|$)',
            r'(?:the\s+implication\s+is|the\s+result\s+is|this\s+leads\s+to)\s+([^.]*?)(?:\.|,|;|$)',
            r'(?:key\s+finding|important\s+discovery|significant\s+result)\s*:?\s*([^.]*?)(?:\.|,|;|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            insights.extend(matches)
        
        # Clean and limit
        insights = [i.strip() for i in insights if len(i.strip()) > 10]
        return insights[:20]
    
    def identify_core_concepts(self, concepts: set, source_count: Dict[str, int]) -> set:
        """Identify core concepts that appear across multiple sources"""
        concept_frequency = defaultdict(int)
        
        # Count concept frequency across sources
        for concept in concepts:
            for source, count in source_count.items():
                if count > 0:
                    concept_frequency[concept] += 1
        
        # Filter by frequency threshold
        core_concepts = set()
        min_sources = max(1, len(source_count) // 2)  # At least half of sources
        
        for concept, frequency in concept_frequency.items():
            if frequency >= min_sources:
                core_concepts.add(concept)
        
        # If too few core concepts, include some high-frequency ones
        if len(core_concepts) < 5:
            sorted_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)
            for concept, freq in sorted_concepts[:10]:
                core_concepts.add(concept)
        
        return core_concepts
    
    def build_knowledge_relationships(self, concepts: set, raw_relationships: List[str]) -> List[Dict[str, Any]]:
        """Build structured relationships between concepts"""
        relationships = []
        
        # Create concept-to-concept relationships
        concept_list = list(concepts)
        for i, concept1 in enumerate(concept_list):
            for j, concept2 in enumerate(concept_list[i+1:], i+1):
                # Calculate relationship strength based on co-occurrence
                strength = self.calculate_relationship_strength(concept1, concept2, raw_relationships)
                
                if strength > 0.1:  # Minimum relationship threshold
                    relationship = {
                        "source": concept1,
                        "target": concept2,
                        "type": self.determine_relationship_type(concept1, concept2),
                        "strength": strength,
                        "evidence": self.find_relationship_evidence(concept1, concept2, raw_relationships)
                    }
                    relationships.append(relationship)
        
        # Add relationships from raw data
        for rel in raw_relationships:
            if isinstance(rel, str) and len(rel) > 10:
                relationship = {
                    "source": "external",
                    "target": "knowledge",
                    "type": "reference",
                    "strength": 0.5,
                    "evidence": rel
                }
                relationships.append(relationship)
        
        return relationships
    
    def calculate_relationship_strength(self, concept1: str, concept2: str, relationships: List[str]) -> float:
        """Calculate the strength of relationship between two concepts"""
        strength = 0.0
        
        # Check for co-occurrence in relationship text
        for rel in relationships:
            if isinstance(rel, str):
                if concept1.lower() in rel.lower() and concept2.lower() in rel.lower():
                    strength += 0.3
                elif concept1.lower() in rel.lower() or concept2.lower() in rel.lower():
                    strength += 0.1
        
        # Semantic similarity (simple word overlap)
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())
        if words1 and words2:
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            strength += overlap * 0.2
        
        return min(1.0, strength)
    
    def determine_relationship_type(self, concept1: str, concept2: str) -> str:
        """Determine the type of relationship between concepts"""
        # Simple heuristics for relationship types
        if any(word in concept1.lower() for word in ['theory', 'model', 'framework']):
            return "implements"
        elif any(word in concept2.lower() for word in ['application', 'use', 'implementation']):
            return "enables"
        elif any(word in concept1.lower() for word in ['problem', 'challenge', 'issue']):
            return "solves"
        elif any(word in concept2.lower() for word in ['solution', 'answer', 'resolution']):
            return "addresses"
        else:
            return "related_to"
    
    def find_relationship_evidence(self, concept1: str, concept2: str, relationships: List[str]) -> List[str]:
        """Find evidence supporting the relationship between concepts"""
        evidence = []
        
        for rel in relationships:
            if isinstance(rel, str):
                if concept1.lower() in rel.lower() and concept2.lower() in rel.lower():
                    evidence.append(rel[:100] + "..." if len(rel) > 100 else rel)
        
        return evidence[:3]  # Limit to 3 pieces of evidence
    
    def generate_synthetic_insights(self, concepts: set, raw_insights: List[str]) -> List[str]:
        """Generate synthetic insights by combining concepts and raw insights"""
        insights = []
        
        # Add raw insights
        insights.extend(raw_insights[:10])
        
        # Generate synthetic insights
        concept_list = list(concepts)
        if len(concept_list) >= 2:
            # Create insights about relationships
            for i in range(min(5, len(concept_list) - 1)):
                insight = f"The relationship between {concept_list[i]} and {concept_list[i+1]} suggests potential synergies in research and application."
                insights.append(insight)
        
        # Generate domain-specific insights
        if len(concepts) > 0:
            domain_insight = f"The convergence of {len(concepts)} key concepts indicates a mature and well-developed field with multiple research directions."
            insights.append(domain_insight)
        
        return insights[:15]  # Limit to 15 insights
    
    def calculate_synthesis_confidence(self, source_count: Dict[str, int], concept_count: int) -> float:
        """Calculate confidence in the synthesis"""
        # Base confidence on source diversity and concept coverage
        source_diversity = len(source_count) / 4.0  # Normalize to 4 sources
        concept_coverage = min(concept_count / 20.0, 1.0)  # Normalize to 20 concepts
        
        # Weight factors
        source_weight = 0.6
        concept_weight = 0.4
        
        confidence = (source_diversity * source_weight) + (concept_coverage * concept_weight)
        return min(1.0, max(0.0, confidence))
    
    def calculate_knowledge_complexity(self, concepts: set, relationships: List[Dict[str, Any]]) -> float:
        """Calculate the complexity score of the synthesized knowledge"""
        # Complexity based on number of concepts and relationships
        concept_complexity = min(len(concepts) / 50.0, 1.0)  # Normalize to 50 concepts
        relationship_complexity = min(len(relationships) / 100.0, 1.0)  # Normalize to 100 relationships
        
        # Weight factors
        concept_weight = 0.4
        relationship_weight = 0.6
        
        complexity = (concept_complexity * concept_weight) + (relationship_complexity * relationship_weight)
        return min(1.0, max(0.0, complexity))
    
    def determine_topic(self, concepts: set) -> str:
        """Determine the main topic from the set of concepts"""
        if not concepts:
            return "Unknown Topic"
        
        # Find the most representative concept
        concept_scores = {}
        for concept in concepts:
            score = 0
            # Prefer longer, more specific concepts
            score += len(concept.split()) * 0.1
            # Prefer concepts with technical terms
            if any(word in concept.lower() for word in ['theory', 'model', 'system', 'algorithm']):
                score += 0.5
            concept_scores[concept] = score
        
        # Return the concept with highest score
        if concept_scores:
            return max(concept_scores.items(), key=lambda x: x[1])[0]
        
        return list(concepts)[0] if concepts else "Unknown Topic"
    
    def export_knowledge(self, synthesized: SynthesizedKnowledge, format: str = "json") -> str:
        """Export synthesized knowledge in specified format"""
        if format.lower() == "json":
            # Convert datetime to string for JSON serialization
            export_data = {
                "topic": synthesized.topic,
                "core_concepts": synthesized.core_concepts,
                "definitions": synthesized.definitions,
                "relationships": synthesized.relationships,
                "insights": synthesized.insights,
                "confidence": synthesized.confidence,
                "sources": synthesized.sources,
                "timestamp": synthesized.timestamp.isoformat(),
                "complexity_score": synthesized.complexity_score
            }
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "markdown":
            md = f"# {synthesized.topic}\n\n"
            md += f"**Synthesized Knowledge** - {synthesized.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md += f"**Confidence:** {synthesized.confidence:.2f}\n"
            md += f"**Complexity Score:** {synthesized.complexity_score:.2f}\n\n"
            
            md += "## Core Concepts\n\n"
            for concept in synthesized.core_concepts:
                md += f"- {concept}\n"
            
            md += "\n## Key Relationships\n\n"
            for rel in synthesized.relationships[:10]:  # Top 10 relationships
                md += f"- **{rel['source']}** → **{rel['target']}** ({rel['type']}, strength: {rel['strength']:.2f})\n"
            
            md += "\n## Insights\n\n"
            for insight in synthesized.insights[:10]:  # Top 10 insights
                md += f"- {insight}\n"
            
            md += f"\n## Sources\n\n"
            for source in synthesized.sources:
                md += f"- {source}\n"
            
            return md
        
        else:
            raise ValueError(f"Unsupported format: {format}")

async def main():
    """Test the knowledge synthesizer"""
    synthesizer = KnowledgeSynthesizer()
    
    # Mock research results for testing
    mock_results = {
        "wikipedia": [
            type('MockResult', (), {
                'content': 'Quantum computing is a field that uses quantum mechanical phenomena to process information.',
                'connections': ['quantum mechanics', 'information processing'],
                'metadata': {}
            })(),
            type('MockResult', (), {
                'content': 'Quantum algorithms can solve certain problems faster than classical computers.',
                'connections': ['algorithms', 'performance improvement'],
                'metadata': {}
            })()
        ],
        "arxiv": [
            type('MockResult', (), {
                'content': 'Recent advances in quantum error correction show promising results.',
                'connections': ['error correction', 'reliability'],
                'metadata': {}
            })()
        ]
    }
    
    # Synthesize knowledge
    synthesized = await synthesizer.synthesize_research_findings(mock_results)
    
    print(f"🔬 Synthesized Knowledge:")
    print(f"Topic: {synthesized.topic}")
    print(f"Core Concepts: {len(synthesized.core_concepts)}")
    print(f"Relationships: {len(synthesized.relationships)}")
    print(f"Insights: {len(synthesized.insights)}")
    print(f"Confidence: {synthesized.confidence:.2f}")
    print(f"Complexity: {synthesized.complexity_score:.2f}")
    
    # Export in different formats
    json_export = synthesizer.export_knowledge(synthesized, "json")
    md_export = synthesizer.export_knowledge(synthesized, "markdown")
    
    print(f"\n📄 JSON Export (first 200 chars):")
    print(json_export[:200] + "...")
    
    print(f"\n📝 Markdown Export (first 200 chars):")
    print(md_export[:200] + "...")

if __name__ == "__main__":
    asyncio.run(main())
