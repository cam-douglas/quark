#!/usr/bin/env python3
"""Marker Expression Engine.

Dynamic molecular marker expression system for neuroepithelial cells
including temporal regulation, morphogen responsiveness, and cell state
transitions with comprehensive marker identification capabilities.

Integration: Expression engine component for marker expression system
Rationale: Focused marker expression logic separated from identification system
"""

from typing import Dict, List, Optional
import numpy as np
import logging

from .neuroepithelial_cell_types import NeuroepithelialCellType
from .competency_window_types import FateType
from .molecular_markers import MolecularMarker

logger = logging.getLogger(__name__)

class MarkerExpressionEngine:
    """Engine for dynamic molecular marker expression.
    
    Provides dynamic marker expression based on developmental time,
    morphogen levels, cell type, and competency state with realistic
    biological regulation and responsiveness.
    """
    
    def __init__(self):
        """Initialize marker expression engine."""
        self.expression_rules = self._initialize_expression_rules()
        
        logger.info("Initialized MarkerExpressionEngine")
    
    def _initialize_expression_rules(self) -> Dict[str, Dict[str, any]]:
        """Initialize marker expression rules for different markers."""
        return {
            'Nestin': {
                'base_expression': 1.0,
                'temporal_peak': 9.0,
                'temporal_decay_rate': 0.1,
                'morphogen_responsiveness': {
                    'FGF': 0.5,     # Enhanced by FGF (maintenance)
                    'BMP': -0.2,    # Slightly inhibited by BMP
                },
                'cell_type_modulation': {
                    NeuroepithelialCellType.EARLY_MULTIPOTENT: 1.0,
                    NeuroepithelialCellType.LATE_MULTIPOTENT: 0.8,
                    NeuroepithelialCellType.COMMITTED_PROGENITOR: 0.3
                }
            },
            'Sox2': {
                'base_expression': 0.9,
                'temporal_peak': 8.5,
                'temporal_decay_rate': 0.15,
                'morphogen_responsiveness': {
                    'FGF': 0.8,     # Strongly enhanced by FGF
                    'WNT': 0.3,     # Moderately enhanced by WNT
                    'BMP': -0.1     # Slightly inhibited by BMP
                },
                'cell_type_modulation': {
                    NeuroepithelialCellType.EARLY_MULTIPOTENT: 1.0,
                    NeuroepithelialCellType.LATE_MULTIPOTENT: 0.9,
                    NeuroepithelialCellType.COMMITTED_PROGENITOR: 0.2
                }
            },
            'Pax6': {
                'base_expression': 0.8,
                'temporal_peak': 10.0,
                'temporal_decay_rate': 0.05,
                'morphogen_responsiveness': {
                    'SHH': 0.3,     # Moderately enhanced by low SHH
                    'BMP': 0.2,     # Enhanced by BMP
                    'FGF': 0.4      # Enhanced by FGF
                },
                'cell_type_modulation': {
                    NeuroepithelialCellType.EARLY_MULTIPOTENT: 0.8,
                    NeuroepithelialCellType.LATE_MULTIPOTENT: 1.0,
                    NeuroepithelialCellType.COMMITTED_PROGENITOR: 0.6
                }
            },
            'PCNA': {
                'base_expression': 0.7,
                'temporal_peak': 9.5,
                'temporal_decay_rate': 0.08,
                'morphogen_responsiveness': {
                    'FGF': 0.6,     # Enhanced by FGF (proliferation)
                    'SHH': 0.2,     # Slightly enhanced by SHH
                },
                'cell_type_modulation': {
                    NeuroepithelialCellType.EARLY_MULTIPOTENT: 1.0,
                    NeuroepithelialCellType.LATE_MULTIPOTENT: 0.8,
                    NeuroepithelialCellType.COMMITTED_PROGENITOR: 0.4
                }
            },
            'Ki67': {
                'base_expression': 0.6,
                'temporal_peak': 9.0,
                'temporal_decay_rate': 0.12,
                'morphogen_responsiveness': {
                    'FGF': 0.5,     # Enhanced by FGF
                    'WNT': 0.3      # Enhanced by WNT
                },
                'cell_type_modulation': {
                    NeuroepithelialCellType.EARLY_MULTIPOTENT: 1.0,
                    NeuroepithelialCellType.LATE_MULTIPOTENT: 0.7,
                    NeuroepithelialCellType.COMMITTED_PROGENITOR: 0.3
                }
            },
            'Tbr2': {
                'base_expression': 0.0,  # Only in intermediate progenitors
                'temporal_peak': 11.0,
                'temporal_decay_rate': 0.1,
                'morphogen_responsiveness': {
                    'FGF': 0.4,     # Enhanced by FGF
                    'SHH': 0.2,     # Slightly enhanced by SHH
                    'BMP': 0.2      # Slightly enhanced by BMP
                },
                'cell_type_modulation': {
                    NeuroepithelialCellType.EARLY_MULTIPOTENT: 0.0,
                    NeuroepithelialCellType.LATE_MULTIPOTENT: 0.1,
                    NeuroepithelialCellType.COMMITTED_PROGENITOR: 1.0
                }
            }
        }
    
    def calculate_dynamic_expression(self, marker_name: str, 
                                   cell_type: NeuroepithelialCellType,
                                   developmental_time: float,
                                   morphogen_levels: Dict[str, float],
                                   cell_cycle_phase: str = 'G1') -> float:
        """Calculate dynamic marker expression level.
        
        Args:
            marker_name: Name of molecular marker
            cell_type: Current cell type
            developmental_time: Current developmental time (weeks)
            morphogen_levels: Current morphogen concentrations
            cell_cycle_phase: Current cell cycle phase
            
        Returns:
            Expression level (0-1)
        """
        if marker_name not in self.expression_rules:
            return 0.0
        
        rules = self.expression_rules[marker_name]
        
        # Base expression level
        base_expression = rules['base_expression']
        
        # Temporal modulation
        temporal_factor = self._calculate_temporal_modulation(
            developmental_time, rules['temporal_peak'], rules['temporal_decay_rate'])
        
        # Morphogen modulation
        morphogen_factor = self._calculate_morphogen_modulation(
            morphogen_levels, rules['morphogen_responsiveness'])
        
        # Cell type modulation
        cell_type_factor = rules['cell_type_modulation'].get(cell_type, 0.5)
        
        # Cell cycle modulation (some markers are cell cycle dependent)
        cycle_factor = self._calculate_cell_cycle_modulation(marker_name, cell_cycle_phase)
        
        # Final expression level
        expression_level = (base_expression * temporal_factor * 
                          morphogen_factor * cell_type_factor * cycle_factor)
        
        return np.clip(expression_level, 0.0, 1.0)
    
    def _calculate_temporal_modulation(self, current_time: float, 
                                     peak_time: float, decay_rate: float) -> float:
        """Calculate temporal modulation of marker expression."""
        # Gaussian-like curve centered on peak time
        time_diff = current_time - peak_time
        temporal_factor = np.exp(-decay_rate * time_diff**2)
        
        return temporal_factor
    
    def _calculate_morphogen_modulation(self, morphogen_levels: Dict[str, float],
                                      responsiveness: Dict[str, float]) -> float:
        """Calculate morphogen-dependent modulation."""
        modulation_factor = 1.0
        
        for morphogen, response_strength in responsiveness.items():
            if morphogen in morphogen_levels:
                morphogen_level = morphogen_levels[morphogen]
                
                if response_strength > 0:
                    # Positive modulation
                    modulation_factor *= (1.0 + response_strength * morphogen_level)
                else:
                    # Negative modulation
                    modulation_factor *= (1.0 + response_strength * morphogen_level)
        
        return max(0.1, modulation_factor)  # Minimum 10% expression
    
    def _calculate_cell_cycle_modulation(self, marker_name: str, 
                                       cell_cycle_phase: str) -> float:
        """Calculate cell cycle-dependent modulation."""
        # Cell cycle-specific markers
        if marker_name == 'PCNA':
            # PCNA peaks during S phase
            if cell_cycle_phase.lower() == 's':
                return 1.5
            elif cell_cycle_phase.lower() in ['g1', 'g2']:
                return 1.0
            else:  # M or G0
                return 0.3
                
        elif marker_name == 'Ki67':
            # Ki67 absent in G0, present in cycling
            if cell_cycle_phase.lower() == 'g0':
                return 0.1
            elif cell_cycle_phase.lower() == 'm':
                return 1.2
            else:
                return 1.0
        
        # Most markers not strongly cell cycle dependent
        return 1.0
    
    def calculate_marker_signature(self, cell_type: NeuroepithelialCellType,
                                  developmental_time: float,
                                  morphogen_levels: Dict[str, float],
                                  cell_cycle_phase: str = 'G1') -> Dict[str, float]:
        """Calculate complete marker signature for cell state.
        
        Args:
            cell_type: Current cell type
            developmental_time: Current developmental time
            morphogen_levels: Current morphogen levels
            cell_cycle_phase: Cell cycle phase
            
        Returns:
            Complete marker expression signature
        """
        marker_signature = {}
        
        for marker_name in self.expression_rules.keys():
            expression_level = self.calculate_dynamic_expression(
                marker_name, cell_type, developmental_time, 
                morphogen_levels, cell_cycle_phase)
            marker_signature[marker_name] = expression_level
        
        return marker_signature
    
    def predict_cell_type_from_markers(self, marker_levels: Dict[str, float]) -> NeuroepithelialCellType:
        """Predict cell type from marker expression levels.
        
        Args:
            marker_levels: Current marker expression levels
            
        Returns:
            Predicted cell type
        """
        # Cell type prediction based on marker combinations
        nestin = marker_levels.get('Nestin', 0.0)
        sox2 = marker_levels.get('Sox2', 0.0)
        tbr2 = marker_levels.get('Tbr2', 0.0)
        ki67 = marker_levels.get('Ki67', 0.0)
        
        # Committed progenitor: High Tbr2, reduced Sox2
        if tbr2 > 0.4 and sox2 < 0.5:
            return NeuroepithelialCellType.COMMITTED_PROGENITOR
        
        # Early multipotent: High Nestin, Sox2, Ki67
        elif nestin > 0.8 and sox2 > 0.6 and ki67 > 0.5:
            return NeuroepithelialCellType.EARLY_MULTIPOTENT
        
        # Late multipotent: Moderate markers
        elif nestin > 0.5 and sox2 > 0.4:
            return NeuroepithelialCellType.LATE_MULTIPOTENT
        
        # Transitioning state
        else:
            return NeuroepithelialCellType.TRANSITIONING
