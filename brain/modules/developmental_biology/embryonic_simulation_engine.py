"""
Embryonic Simulation Engine

Core engine for running embryonic development simulations.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
import time
from typing import Dict, List

from .neuroepithelial_cells import NeuroepithelialCell
from .neuroepithelial_cell_types import NeuroepithelialCellType
from .committed_progenitor_generator import CommittedProgenitorGenerator
from .apoptosis_manager import ApoptosisManager


class EmbryonicSimulationEngine:
    """Core engine for embryonic development simulation"""
    
    def __init__(self):
        """Initialize simulation engine"""
        self.progenitor_generator = CommittedProgenitorGenerator()
        self.apoptosis_manager = ApoptosisManager()  # NEW
    
    def run_simulation(self,
                      initial_cell_count: int,
                      simulation_duration: float,
                      time_step: float) -> Dict[str, any]:
        """
        Run embryonic development simulation
        
        Args:
            initial_cell_count: Number of initial cells
            simulation_duration: Duration in hours
            time_step: Time step in hours
        
        Returns:
            Simulation results
        """
        start_time = time.time()
        
        # Initialize neuroepithelial cells
        neuroepithelial_cells = {}
        for i in range(initial_cell_count):
            cell_id = f'sim_cell_{i+1}'
            neuroepithelial_cells[cell_id] = NeuroepithelialCell(
                cell_type=NeuroepithelialCellType.EARLY_MULTIPOTENT,
                position=(np.random.random(), np.random.random(), np.random.random()),
                developmental_time=9.0
            )
        
        # Simulate development over time
        current_time = 9.0
        end_time = 9.0 + simulation_duration
        committed_progenitors = {}
        
        step_count = 0
        max_steps = int(simulation_duration / time_step)
        
        while current_time < end_time and step_count < max_steps:
            # Update developmental time
            for cell in neuroepithelial_cells.values():
                cell.birth_time = current_time
            
            # Generate committed progenitors at appropriate stages
            if 9.5 <= current_time <= 11.5:
                morphogen_concentrations = {
                    'SHH': 0.8 * (1.0 - (current_time - 9.0) / 10.0),
                    'BMP': 0.3,
                    'WNT': 0.2,
                    'FGF': 0.4
                }
                
                new_progenitors = self.progenitor_generator.generate_committed_progenitors(
                    neuroepithelial_cells, current_time, morphogen_concentrations
                )
                committed_progenitors.update(new_progenitors)
            
            # --- NEW: Apply apoptosis each step (uses BMP level) ---
            bmp_level = morphogen_concentrations.get('BMP', 0.3) if 'morphogen_concentrations' in locals() else 0.3
            self.apoptosis_manager.apply_apoptosis(
                neuroepithelial_cells,
                dt_hours=time_step,
                get_bmp=lambda _pos, lvl=bmp_level: lvl
            )
            
            # Update time
            current_time += time_step
            step_count += 1
        
        simulation_time = time.time() - start_time
        
        return {
            'neuroepithelial_cells': neuroepithelial_cells,
            'committed_progenitors': committed_progenitors,
            'simulation_time': simulation_time,
            'final_time': current_time,
            'steps_completed': step_count
        }
