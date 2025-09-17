#!/usr/bin/env python3
"""Data Extraction Patterns for Experimental Papers.

Regex patterns and extraction logic for finding quantitative lineage data
in developmental biology research papers including clone sizes, fate
proportions, division patterns, and temporal progression data.

Integration: Pattern matching component for PDF data extraction
Rationale: Focused pattern definitions separated from extraction logic
"""

from typing import Dict, List
import re
import logging

logger = logging.getLogger(__name__)

class DataExtractionPatterns:
    """Pattern definitions for extracting experimental lineage data.
    
    Provides regex patterns and extraction methods for finding
    quantitative data in developmental biology research papers.
    """
    
    def __init__(self):
        """Initialize data extraction patterns."""
        self.patterns = self._initialize_extraction_patterns()
        
        logger.info("Initialized DataExtractionPatterns")
    
    def _initialize_extraction_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for extracting experimental data."""
        return {
            'clone_sizes': [
                r'clone size[s]?\s*(?:of\s*)?(\d+)(?:\s*[-–]\s*(\d+))?',
                r'(\d+)\s*cells?\s*per\s*clone',
                r'clones?\s*containing\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*cells?',
                r'(\d+)\s*cell\s*clones?',
                r'(\d+)(?:\s*[-–]\s*(\d+))?\s*cells?\s*in\s*each\s*clone'
            ],
            'fate_proportions': [
                r'(\d+(?:\.\d+)?)%?\s*(?:of\s*cells?\s*)?(?:became?|differentiated?\s*into|adopted)\s*(\w+(?:\s+\w+)?)\s*fate',
                r'(\w+(?:\s+\w+)?)\s*(?:fate|cells?):\s*(\d+(?:\.\d+)?)%?',
                r'proportion\s*of\s*(\w+(?:\s+\w+)?)\s*(?:cells?|fate):\s*(\d+(?:\.\d+)?)%?',
                r'(\d+(?:\.\d+)?)%?\s*(\w+(?:\s+\w+)?)\s*cells?',
                r'(\w+(?:\s+\w+)?)\s*neurons?:\s*(\d+(?:\.\d+)?)%?'
            ],
            'division_patterns': [
                r'(\d+(?:\.\d+)?)%?\s*symmetric\s*divisions?',
                r'(\d+(?:\.\d+)?)%?\s*asymmetric\s*divisions?',
                r'symmetric:asymmetric\s*ratio\s*(?:of\s*)?(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)%?\s*(?:of\s*)?divisions?\s*were\s*symmetric',
                r'(\d+(?:\.\d+)?)%?\s*(?:of\s*)?divisions?\s*were\s*asymmetric',
                r'(\d+(?:\.\d+)?)\s*symmetric\s*(?:vs\.?|versus)\s*(\d+(?:\.\d+)?)\s*asymmetric'
            ],
            'temporal_data': [
                r'E(\d+(?:\.\d+)?)\s*[-–]\s*E(\d+(?:\.\d+)?)',
                r'embryonic\s*day\s*(\d+(?:\.\d+)?)',
                r'E(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*days?\s*(?:post\s*)?(?:conception|fertilization)',
                r'gestation\s*day\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*weeks?\s*gestation'
            ]
        }
    
    def extract_clone_size_data(self, text: str) -> List[Dict[str, any]]:
        """Extract clone size data from text."""
        clone_data = []
        
        for pattern in self.patterns['clone_sizes']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if isinstance(match, tuple):
                    # Range of clone sizes
                    if len(match) >= 2 and match[1]:
                        clone_data.append({
                            'type': 'clone_size_range',
                            'min_size': int(match[0]),
                            'max_size': int(match[1]),
                            'source_text': str(match)
                        })
                    else:
                        clone_data.append({
                            'type': 'clone_size',
                            'size': int(match[0]),
                            'source_text': str(match)
                        })
                else:
                    clone_data.append({
                        'type': 'clone_size',
                        'size': int(match),
                        'source_text': str(match)
                    })
        
        return clone_data
    
    def extract_fate_proportion_data(self, text: str) -> List[Dict[str, any]]:
        """Extract cell fate proportion data from text."""
        fate_data = []
        
        for pattern in self.patterns['fate_proportions']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    try:
                        # Handle different tuple formats
                        if match[0].replace('.', '').isdigit():
                            proportion = float(match[0])
                            fate_type = match[1]
                        else:
                            fate_type = match[0]
                            proportion = float(match[1])
                        
                        fate_data.append({
                            'fate_type': fate_type.lower().strip(),
                            'proportion': proportion / 100.0 if proportion > 1.0 else proportion,
                            'source_text': str(match)
                        })
                    except (ValueError, IndexError):
                        continue
        
        return fate_data
    
    def extract_division_pattern_data(self, text: str) -> List[Dict[str, any]]:
        """Extract division pattern data from text."""
        division_data = []
        
        for pattern in self.patterns['division_patterns']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        # Ratio format
                        if len(match) >= 2 and match[1]:
                            symmetric_ratio = float(match[0])
                            asymmetric_ratio = float(match[1])
                            total = symmetric_ratio + asymmetric_ratio
                            
                            if total > 0:
                                division_data.append({
                                    'symmetric_fraction': symmetric_ratio / total,
                                    'asymmetric_fraction': asymmetric_ratio / total,
                                    'source_text': str(match)
                                })
                    else:
                        # Percentage format
                        percentage = float(match)
                        division_type = 'symmetric' if 'symmetric' in text.lower() else 'asymmetric'
                        
                        division_data.append({
                            'division_type': division_type,
                            'percentage': percentage / 100.0 if percentage > 1.0 else percentage,
                            'source_text': str(match)
                        })
                except (ValueError, ZeroDivisionError):
                    continue
        
        return division_data
    
    def extract_temporal_data(self, text: str) -> List[Dict[str, any]]:
        """Extract temporal progression data from text."""
        temporal_data = []
        
        for pattern in self.patterns['temporal_data']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        # Range format
                        if len(match) >= 2 and match[1]:
                            start_time = float(match[0])
                            end_time = float(match[1])
                            
                            temporal_data.append({
                                'type': 'developmental_window',
                                'start_time': start_time,
                                'end_time': end_time,
                                'source_text': str(match)
                            })
                    else:
                        # Single timepoint
                        timepoint = float(match)
                        temporal_data.append({
                            'type': 'timepoint',
                            'time': timepoint,
                            'source_text': str(match)
                        })
                except ValueError:
                    continue
        
        return temporal_data
    
    def calculate_extraction_confidence(self, clone_data: List, fate_data: List,
                                      division_data: List, temporal_data: List) -> float:
        """Calculate confidence in extracted data quality."""
        total_data_points = len(clone_data) + len(fate_data) + len(division_data) + len(temporal_data)
        
        if total_data_points == 0:
            return 0.0
        
        # Base confidence on number and types of data extracted
        confidence = min(1.0, total_data_points / 10.0)  # Scale to 0-1
        
        # Bonus for having multiple data types
        data_type_count = sum(1 for data_list in [clone_data, fate_data, division_data, temporal_data] 
                             if len(data_list) > 0)
        
        confidence += 0.1 * data_type_count  # Bonus for diversity
        
        return min(1.0, confidence)
