"""
Pattern Recognizer - Discovers patterns in knowledge and data
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class RecognizedPattern:
    """Represents a recognized pattern."""
    pattern_id: str
    pattern_type: str
    elements: List[Any]
    strength: float
    frequency: int
    description: str

class PatternRecognizer:
    """Recognizes patterns in various types of data and knowledge."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.recognized_patterns: List[RecognizedPattern] = []
        self.pattern_cache: Dict[str, Any] = {}
        
    def recognize_patterns(self, 
                          data: List[Any],
                          pattern_types: Optional[List[str]] = None) -> List[RecognizedPattern]:
        """Recognize patterns in data."""
        
        pattern_types = pattern_types or ['sequence', 'frequency', 'similarity', 'hierarchy']
        patterns = []
        
        for pattern_type in pattern_types:
            if pattern_type == 'sequence':
                patterns.extend(self._find_sequence_patterns(data))
            elif pattern_type == 'frequency':
                patterns.extend(self._find_frequency_patterns(data))
            elif pattern_type == 'similarity':
                patterns.extend(self._find_similarity_patterns(data))
            elif pattern_type == 'hierarchy':
                patterns.extend(self._find_hierarchical_patterns(data))
                
        # Store recognized patterns
        self.recognized_patterns.extend(patterns)
        
        return patterns
        
    def _find_sequence_patterns(self, data: List[Any]) -> List[RecognizedPattern]:
        """Find sequential patterns in data."""
        patterns = []
        
        # Look for repeated sequences
        for seq_length in range(2, min(5, len(data) // 2)):
            sequences = defaultdict(int)
            
            for i in range(len(data) - seq_length + 1):
                seq = tuple(str(item) for item in data[i:i+seq_length])
                sequences[seq] += 1
                
            # Find frequent sequences
            for seq, freq in sequences.items():
                if freq >= 2:  # Appears at least twice
                    pattern = RecognizedPattern(
                        pattern_id=f"seq_{len(patterns)}",
                        pattern_type="sequence",
                        elements=list(seq),
                        strength=min(freq / len(data), 1.0),
                        frequency=freq,
                        description=f"Sequence {' -> '.join(seq)} appears {freq} times"
                    )
                    patterns.append(pattern)
                    
        return patterns
        
    def _find_frequency_patterns(self, data: List[Any]) -> List[RecognizedPattern]:
        """Find frequency-based patterns."""
        patterns = []
        
        # Count item frequencies
        frequencies = defaultdict(int)
        for item in data:
            frequencies[str(item)] += 1
            
        total_items = len(data)
        
        # Find high-frequency items
        for item, freq in frequencies.items():
            if freq >= max(2, total_items * 0.1):  # At least 10% frequency
                pattern = RecognizedPattern(
                    pattern_id=f"freq_{len(patterns)}",
                    pattern_type="frequency",
                    elements=[item],
                    strength=freq / total_items,
                    frequency=freq,
                    description=f"'{item}' appears frequently ({freq}/{total_items})"
                )
                patterns.append(pattern)
                
        return patterns
        
    def _find_similarity_patterns(self, data: List[Any]) -> List[RecognizedPattern]:
        """Find similarity-based patterns."""
        patterns = []
        
        # Group similar items
        similarity_groups = defaultdict(list)
        
        for i, item1 in enumerate(data):
            for j, item2 in enumerate(data[i+1:], i+1):
                similarity = self._calculate_similarity(item1, item2)
                if similarity > 0.7:  # High similarity threshold
                    key = f"similar_{min(i,j)}"
                    similarity_groups[key].extend([item1, item2])
                    
        # Create patterns from similarity groups
        for group_id, items in similarity_groups.items():
            if len(set(str(item) for item in items)) >= 2:  # At least 2 unique items
                pattern = RecognizedPattern(
                    pattern_id=f"sim_{len(patterns)}",
                    pattern_type="similarity",
                    elements=list(set(str(item) for item in items)),
                    strength=0.7,
                    frequency=len(items),
                    description=f"Similar items: {', '.join(set(str(item) for item in items[:3]))}"
                )
                patterns.append(pattern)
                
        return patterns
        
    def _find_hierarchical_patterns(self, data: List[Any]) -> List[RecognizedPattern]:
        """Find hierarchical patterns."""
        patterns = []
        
        # Simple hierarchical detection based on structure
        str_data = [str(item) for item in data]
        
        # Look for items that contain other items (simple containment hierarchy)
        containment_relationships = []
        
        for i, item1 in enumerate(str_data):
            for j, item2 in enumerate(str_data):
                if i != j and item2 in item1 and len(item2) < len(item1):
                    containment_relationships.append((item1, item2))
                    
        if containment_relationships:
            pattern = RecognizedPattern(
                pattern_id=f"hier_{len(patterns)}",
                pattern_type="hierarchy",
                elements=[rel[0] for rel in containment_relationships[:5]],
                strength=len(containment_relationships) / len(data),
                frequency=len(containment_relationships),
                description=f"Hierarchical containment relationships detected"
            )
            patterns.append(pattern)
            
        return patterns
        
    def _calculate_similarity(self, item1: Any, item2: Any) -> float:
        """Calculate similarity between two items."""
        str1 = str(item1).lower()
        str2 = str(item2).lower()
        
        if not str1 or not str2:
            return 0.0
            
        # Simple word-based similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of recognized patterns."""
        
        if not self.recognized_patterns:
            return {'total_patterns': 0}
            
        by_type = defaultdict(int)
        total_strength = 0.0
        
        for pattern in self.recognized_patterns:
            by_type[pattern.pattern_type] += 1
            total_strength += pattern.strength
            
        return {
            'total_patterns': len(self.recognized_patterns),
            'average_strength': total_strength / len(self.recognized_patterns),
            'patterns_by_type': dict(by_type),
            'strongest_pattern': max(self.recognized_patterns, key=lambda p: p.strength).description
        }
