"""
Progenitor Pool Manager

This module manages progenitor pools for specific neuron types,
tracks pool statistics, and validates lineage potency.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict, List
from .progenitor_type_classifier import NeuronType
from .committed_progenitor_types import CommittedProgenitor


class ProgenitorPoolManager:
    """
    Manages progenitor pools for specific neuron types,
    tracks pool statistics, and validates lineage potency.
    """
    
    def __init__(self):
        """Initialize progenitor pool manager"""
        self.progenitor_pools: Dict[NeuronType, List[CommittedProgenitor]] = {}
    
    def add_to_progenitor_pool(self, progenitor: CommittedProgenitor) -> None:
        """Add committed progenitor to appropriate progenitor pool"""
        for neuron_type in progenitor.lineage_potency.allowed_fates:
            if neuron_type not in self.progenitor_pools:
                self.progenitor_pools[neuron_type] = []
            self.progenitor_pools[neuron_type].append(progenitor)
    
    def get_progenitor_pool(self, neuron_type: NeuronType) -> List[CommittedProgenitor]:
        """Get progenitor pool for specific neuron type"""
        return self.progenitor_pools.get(neuron_type, [])
    
    def get_progenitor_pool_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all progenitor pools"""
        statistics = {}
        
        for neuron_type, progenitors in self.progenitor_pools.items():
            statistics[neuron_type.value] = {
                "total_progenitors": len(progenitors),
                "commitment_levels": {},
                "zone_distribution": {}
            }
            
            # Count commitment levels
            for progenitor in progenitors:
                level = progenitor.lineage_potency.commitment_level.value
                statistics[neuron_type.value]["commitment_levels"][level] = \
                    statistics[neuron_type.value]["commitment_levels"].get(level, 0) + 1
                
                # Count zone distribution
                zone = progenitor.zone_type.value
                statistics[neuron_type.value]["zone_distribution"][zone] = \
                    statistics[neuron_type.value]["zone_distribution"].get(zone, 0) + 1
        
        return statistics
    
    def validate_lineage_potency(self, progenitor: CommittedProgenitor, 
                               target_fate: NeuronType) -> bool:
        """Validate if progenitor can differentiate into target fate"""
        return target_fate in progenitor.lineage_potency.allowed_fates
    
    def get_pool_sizes(self) -> Dict[str, int]:
        """Get sizes of all progenitor pools"""
        return {neuron_type.value: len(progenitors) 
                for neuron_type, progenitors in self.progenitor_pools.items()}
    
    def clear_pools(self) -> None:
        """Clear all progenitor pools"""
        self.progenitor_pools.clear()
    
    def remove_progenitor(self, progenitor_id: str) -> bool:
        """Remove progenitor from all pools"""
        removed = False
        for neuron_type, progenitors in self.progenitor_pools.items():
            original_length = len(progenitors)
            self.progenitor_pools[neuron_type] = [p for p in progenitors if p.cell_id != progenitor_id]
            if len(self.progenitor_pools[neuron_type]) < original_length:
                removed = True
        return removed
