#!/usr/bin/env python3
"""
Knowledge Validation System for Exponential Learning
Cross-references information from multiple sources for accuracy
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
class ValidationResult:
    """Represents validation result for a piece of knowledge"""
    concept: str
    confidence: float
    sources: List[str]
    conflicts: List[str]
    consensus_score: float
    validation_timestamp: datetime
    status: str  # "validated", "conflicting", "uncertain"

class KnowledgeValidationSystem:
    """
    Validates knowledge by cross-referencing multiple sources
    Identifies conflicts and calculates consensus scores
    """
    
    def __init__(self):
        self.validation_cache = {}
        self.conflict_resolution_rules = []
        self.consensus_threshold = 0.7
        self.min_sources_for_validation = 2
        
    async def validate_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate multiple knowledge items"""
        logger.info(f"🔍 Validating {len(knowledge_items)} knowledge items")
        
        validation_tasks = []
        for item in knowledge_items:
            task = self.validate_single_item(item)
            validation_tasks.append(task)
        
        # Run validations in parallel
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = [r for r in results if isinstance(r, ValidationResult)]
        
        logger.info(f"✅ Validated {len(valid_results)} knowledge items")
        return valid_results
    
    async def validate_single_item(self, item: Dict[str, Any]) -> ValidationResult:
        """Validate a single knowledge item"""
        concept = item.get('concept', 'unknown')
        sources = item.get('sources', [])
        content = item.get('content', '')
        
        # Check cache first
        cache_key = f"{concept}_{hash(content)}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Perform validation
        confidence = self.calculate_confidence(sources, content)
        conflicts = self.identify_conflicts(concept, content, sources)
        consensus_score = self.calculate_consensus_score(sources, conflicts)
        
        # Determine validation status
        status = self.determine_validation_status(consensus_score, len(sources))
        
        result = ValidationResult(
            concept=concept,
            confidence=confidence,
            sources=sources,
            conflicts=conflicts,
            consensus_score=consensus_score,
            validation_timestamp=datetime.now(),
            status=status
        )
        
        # Cache result
        self.validation_cache[cache_key] = result
        
        return result
    
    def calculate_confidence(self, sources: List[str], content: str) -> float:
        """Calculate confidence based on source quality and content"""
        confidence = 0.0
        
        # Source quality scoring
        source_scores = {
            'wikipedia': 0.8,
            'arxiv': 0.9,
            'pubmed': 0.9,
            'dictionary': 0.7,
            'news': 0.6,
            'blog': 0.4
        }
        
        for source in sources:
            source_name = source.lower().split()[0]  # Extract main source name
            score = source_scores.get(source_name, 0.5)
            confidence += score
        
        # Normalize by number of sources
        if sources:
            confidence /= len(sources)
        
        # Content quality factors
        if len(content) > 100:
            confidence += 0.1
        if any(word in content.lower() for word in ['research', 'study', 'analysis', 'evidence']):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def identify_conflicts(self, concept: str, content: str, sources: List[str]) -> List[str]:
        """Identify conflicts between different sources"""
        conflicts = []
        
        # Check for contradictory statements
        contradiction_patterns = [
            (r'is\s+not', r'is\s+(?:a|an)'),
            (r'cannot', r'can'),
            (r'never', r'always'),
            (r'false', r'true'),
            (r'disproves', r'proves')
        ]
        
        for pattern1, pattern2 in contradiction_patterns:
            if re.search(pattern1, content, re.IGNORECASE) and re.search(pattern2, content, re.IGNORECASE):
                conflicts.append(f"Contradictory statements found: {pattern1} vs {pattern2}")
        
        # Check for source-specific conflicts
        if len(sources) > 1:
            # This is a simplified conflict detection
            # In practice, you'd compare actual content from different sources
            conflicts.append(f"Multiple sources ({len(sources)}) may have conflicting information")
        
        return conflicts
    
    def calculate_consensus_score(self, sources: List[str], conflicts: List[str]) -> float:
        """Calculate consensus score based on sources and conflicts"""
        base_score = len(sources) / 10.0  # More sources = higher consensus
        
        # Reduce score for conflicts
        conflict_penalty = len(conflicts) * 0.1
        
        consensus = base_score - conflict_penalty
        return max(0.0, min(1.0, consensus))
    
    def determine_validation_status(self, consensus_score: float, source_count: int) -> str:
        """Determine validation status based on consensus and sources"""
        if source_count < self.min_sources_for_validation:
            return "uncertain"
        elif consensus_score >= self.consensus_threshold:
            return "validated"
        elif consensus_score >= 0.5:
            return "uncertain"
        else:
            return "conflicting"
    
    async def cross_reference_knowledge(self, knowledge_base: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-reference knowledge across the entire base"""
        logger.info("🔗 Starting cross-referencing of knowledge base")
        
        cross_references = {
            "total_items": len(knowledge_base),
            "validated_items": 0,
            "conflicting_items": 0,
            "uncertain_items": 0,
            "conflict_details": [],
            "consensus_summary": {}
        }
        
        # Group by concept for cross-referencing
        concept_groups = defaultdict(list)
        for item in knowledge_base:
            concept = item.get('concept', 'unknown')
            concept_groups[concept].append(item)
        
        # Validate each concept group
        for concept, items in concept_groups.items():
            if len(items) > 1:
                # Multiple sources for same concept - cross-reference
                validation_results = await self.validate_knowledge(items)
                
                # Analyze results
                for result in validation_results:
                    if result.status == "validated":
                        cross_references["validated_items"] += 1
                    elif result.status == "conflicting":
                        cross_references["conflicting_items"] += 1
                        cross_references["conflict_details"].append({
                            "concept": concept,
                            "conflicts": result.conflicts,
                            "sources": result.sources
                        })
                    else:
                        cross_references["uncertain_items"] += 1
                    
                    # Update consensus summary
                    cross_references["consensus_summary"][concept] = result.consensus_score
            else:
                # Single source - mark as uncertain
                cross_references["uncertain_items"] += 1
        
        logger.info(f"✅ Cross-referencing completed: {cross_references['validated_items']} validated, {cross_references['conflicting_items']} conflicting")
        return cross_references
    
    def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts using predefined rules"""
        resolved_conflicts = []
        
        for conflict in conflicts:
            concept = conflict["concept"]
            resolution = self.apply_resolution_rules(concept, conflict)
            
            resolved_conflicts.append({
                "concept": concept,
                "original_conflict": conflict,
                "resolution": resolution,
                "resolved_at": datetime.now()
            })
        
        return resolved_conflicts
    
    def apply_resolution_rules(self, concept: str, conflict: Dict[str, Any]) -> str:
        """Apply resolution rules to a specific conflict"""
        # Source priority rules
        source_priority = {
            'arxiv': 1,
            'pubmed': 2,
            'wikipedia': 3,
            'dictionary': 4,
            'news': 5,
            'blog': 6
        }
        
        # Find highest priority source
        sources = conflict.get("sources", [])
        if sources:
            highest_priority = min(source_priority.get(s.lower().split()[0], 999) for s in sources)
            priority_sources = [s for s in sources if source_priority.get(s.lower().split()[0], 999) == highest_priority]
            
            if priority_sources:
                return f"Resolved using highest priority source: {priority_sources[0]}"
        
        # Default resolution
        return "Conflict remains unresolved - manual review required"
    
    def generate_validation_report(self, validation_results: List[ValidationResult]) -> str:
        """Generate a comprehensive validation report"""
        report = "# Knowledge Validation Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary statistics
        total_items = len(validation_results)
        validated = len([r for r in validation_results if r.status == "validated"])
        conflicting = len([r for r in validation_results if r.status == "conflicting"])
        uncertain = len([r for r in validation_results if r.status == "uncertain"])
        
        report += "## Summary\n\n"
        report += f"- **Total Items:** {total_items}\n"
        report += f"- **Validated:** {validated} ({validated/total_items*100:.1f}%)\n"
        report += f"- **Conflicting:** {conflicting} ({conflicting/total_items*100:.1f}%)\n"
        report += f"- **Uncertain:** {uncertain} ({uncertain/total_items*100:.1f}%)\n\n"
        
        # Detailed results
        report += "## Detailed Results\n\n"
        
        for result in validation_results:
            report += f"### {result.concept}\n\n"
            report += f"- **Status:** {result.status}\n"
            report += f"- **Confidence:** {result.confidence:.2f}\n"
            report += f"- **Consensus Score:** {result.consensus_score:.2f}\n"
            report += f"- **Sources:** {', '.join(result.sources)}\n"
            
            if result.conflicts:
                report += f"- **Conflicts:**\n"
                for conflict in result.conflicts:
                    report += f"  - {conflict}\n"
            
            report += "\n"
        
        return report
    
    def export_validation_data(self, validation_results: List[ValidationResult], format: str = "json") -> str:
        """Export validation data in specified format"""
        if format.lower() == "json":
            # Convert datetime objects to strings
            export_data = []
            for result in validation_results:
                export_item = {
                    "concept": result.concept,
                    "confidence": result.confidence,
                    "sources": result.sources,
                    "conflicts": result.conflicts,
                    "consensus_score": result.consensus_score,
                    "validation_timestamp": result.validation_timestamp.isoformat(),
                    "status": result.status
                }
                export_data.append(export_item)
            
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "csv":
            csv_lines = ["concept,confidence,consensus_score,status,sources,conflicts"]
            
            for result in validation_results:
                sources_str = ";".join(result.sources)
                conflicts_str = ";".join(result.conflicts)
                csv_lines.append(f"{result.concept},{result.confidence:.3f},{result.consensus_score:.3f},{result.status},\"{sources_str}\",\"{conflicts_str}\"")
            
            return "\n".join(csv_lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

async def main():
    """Test the knowledge validation system"""
    validator = KnowledgeValidationSystem()
    
    # Mock knowledge items for testing
    knowledge_items = [
        {
            "concept": "quantum computing",
            "sources": ["wikipedia", "arxiv"],
            "content": "Quantum computing uses quantum mechanical phenomena to process information."
        },
        {
            "concept": "artificial intelligence",
            "sources": ["wikipedia", "news"],
            "content": "AI is the simulation of human intelligence in machines."
        },
        {
            "concept": "machine learning",
            "sources": ["arxiv", "pubmed"],
            "content": "Machine learning enables computers to learn without explicit programming."
        }
    ]
    
    # Validate knowledge
    validation_results = await validator.validate_knowledge(knowledge_items)
    
    print("🔍 Validation Results:")
    for result in validation_results:
        print(f"\n{result.concept}:")
        print(f"  Status: {result.status}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Consensus: {result.consensus_score:.2f}")
        print(f"  Sources: {', '.join(result.sources)}")
        if result.conflicts:
            print(f"  Conflicts: {result.conflicts}")
    
    # Cross-reference knowledge
    cross_refs = await validator.cross_reference_knowledge(knowledge_items)
    print(f"\n🔗 Cross-referencing: {cross_refs['validated_items']} validated, {cross_refs['conflicting_items']} conflicting")
    
    # Generate report
    report = validator.generate_validation_report(validation_results)
    print(f"\n📄 Report generated ({len(report)} characters)")

if __name__ == "__main__":
    asyncio.run(main())
