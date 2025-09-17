"""Apoptosis Manager

Implements stochastic cell death based on BMP levels, p53 activity, and DNA damage.
Based on human embryonic apoptosis rates from Nowakowski et al. 2016 (Cell) and
McClelland et al. 2015 (Development).

Human-specific baseline: ~2-5% apoptotic cells in VZ/SVZ at 8-16 pcw.
"""

from typing import Dict, List, Set, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ApoptosisParameters:
    """Parameters for apoptosis calculation"""
    baseline_rate: float = 0.002  # 0.2% per hour baseline
    bmp_sensitivity: float = 0.001  # Additional death per µM BMP
    p53_sensitivity: float = 0.003  # Additional death when p53 active
    dna_damage_sensitivity: float = 0.002  # Additional death per damage unit
    max_rate: float = 0.01  # 1% per hour maximum


class ApoptosisManager:
    """Manages programmed cell death in developing neural tissue"""
    
    def __init__(self, parameters: Optional[ApoptosisParameters] = None):
        """Initialize apoptosis manager
        
        Args:
            parameters: Apoptosis parameters, uses defaults if None
        """
        self.params = parameters or ApoptosisParameters()
        self.total_removed = 0
        self.removal_history: List[int] = []
        
    def calculate_death_probability(self, 
                                   bmp_level: float = 0.0,
                                   p53_active: bool = False,
                                   dna_damage: float = 0.0,
                                   dt_hours: float = 1.0) -> float:
        """Calculate probability of cell death
        
        Args:
            bmp_level: BMP concentration in µM
            p53_active: Whether p53 pathway is active
            dna_damage: DNA damage level (0-1)
            dt_hours: Time step in hours
            
        Returns:
            Probability of death (0-1)
        """
        rate = self.params.baseline_rate
        rate += self.params.bmp_sensitivity * bmp_level
        rate += self.params.p53_sensitivity * (1.0 if p53_active else 0.0)
        rate += self.params.dna_damage_sensitivity * dna_damage
        
        # Cap at maximum rate
        rate = min(rate, self.params.max_rate)
        
        # Convert rate to probability for time step
        probability = 1.0 - np.exp(-rate * dt_hours)
        return probability
    
    def apply_apoptosis(self, 
                       cells: Dict,
                       dt_hours: float = 1.0,
                       get_bmp: Optional[callable] = None) -> Set[str]:
        """Apply apoptosis to cell population
        
        Args:
            cells: Dictionary of cell_id -> cell object
            dt_hours: Time step in hours
            get_bmp: Optional function to get BMP level for a cell
            
        Returns:
            Set of removed cell IDs
        """
        removed_ids = set()
        
        for cell_id, cell in list(cells.items()):
            # Get BMP level if function provided
            bmp_level = 0.0
            if get_bmp and hasattr(cell, 'position'):
                bmp_level = get_bmp(cell.position)
            
            # Check for p53 activation (stub - could be based on cell cycle)
            p53_active = getattr(cell, 'p53_active', False)
            
            # Check DNA damage (stub - could accumulate over divisions)
            dna_damage = getattr(cell, 'dna_damage', 0.0)
            
            # Calculate death probability
            death_prob = self.calculate_death_probability(
                bmp_level=bmp_level,
                p53_active=p53_active,
                dna_damage=dna_damage,
                dt_hours=dt_hours
            )
            
            # Stochastic death decision
            if np.random.random() < death_prob:
                removed_ids.add(cell_id)
                del cells[cell_id]
        
        # Update statistics
        self.total_removed += len(removed_ids)
        self.removal_history.append(len(removed_ids))
        
        return removed_ids
    
    def get_apoptosis_fraction(self, total_cells: int) -> float:
        """Get fraction of cells that underwent apoptosis
        
        Args:
            total_cells: Total number of cells before apoptosis
            
        Returns:
            Fraction of cells removed (0-1)
        """
        if total_cells == 0:
            return 0.0
        return self.total_removed / (total_cells + self.total_removed)
    
    def validate_against_human_data(self, 
                                   simulated_fraction: float,
                                   pcw: float) -> bool:
        """Validate apoptosis fraction against human data
        
        Args:
            simulated_fraction: Simulated apoptotic fraction
            pcw: Post-conception weeks
            
        Returns:
            True if within expected range
        """
        # Human data from Nowakowski 2016: 2-5% at 8-16 pcw
        if 8 <= pcw <= 16:
            expected_min = 0.02
            expected_max = 0.05
        else:
            # Extrapolate for other stages
            expected_min = 0.01
            expected_max = 0.06
        
        return expected_min <= simulated_fraction <= expected_max
