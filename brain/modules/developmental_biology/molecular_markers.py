#!/usr/bin/env python3
"""Molecular Marker Expression System.

Implements dynamic molecular marker expression for neuroepithelial cells
including Nestin, Sox2, Pax6, and other neural progenitor markers with
temporal regulation and morphogen responsiveness.

Integration: Molecular marker component for neuroepithelial cell system
Rationale: Focused marker expression logic separated from main cell system
"""

from typing import Dict, List, Optional
import numpy as np
import logging

from .neuroepithelial_cell_types import MolecularMarker, NeuroepithelialCellType

logger = logging.getLogger(__name__)

class MolecularMarkerDatabase:
    """Database of molecular markers for neuroepithelial cells.
    
    Manages molecular marker definitions, expression patterns, and
    temporal regulation for neural progenitor identification and
    cell state tracking during embryonic development.
    """
    
    def __init__(self):
        """Initialize molecular marker database."""
        self.markers = self._initialize_marker_database()
        
        logger.info("Initialized MolecularMarkerDatabase")
        logger.info(f"Available markers: {len(self.markers)}")
    
    def _initialize_marker_database(self) -> Dict[str, MolecularMarker]:
        """Initialize database of neural progenitor molecular markers."""
        markers = {}
        
        # Nestin - Neural progenitor marker
        markers['Nestin'] = MolecularMarker(
            marker_name='Nestin',
            expression_level=1.0,           # High expression in progenitors
            temporal_window=(3.0, 12.0),    # E3.0-E12.0 (neural tube closure to neurogenesis)
            cellular_localization='cytoplasm_intermediate_filaments',
            functional_role='Cytoskeletal organization and progenitor maintenance'
        )
        
        # Sox2 - Neural progenitor transcription factor
        markers['Sox2'] = MolecularMarker(
            marker_name='Sox2',
            expression_level=0.9,           # High nuclear expression
            temporal_window=(2.5, 11.0),    # E2.5-E11.0 (neural induction to commitment)
            cellular_localization='nucleus',
            functional_role='Transcriptional control of progenitor identity'
        )
        
        # Pax6 - Neural progenitor specification
        markers['Pax6'] = MolecularMarker(
            marker_name='Pax6',
            expression_level=0.8,           # Moderate to high expression
            temporal_window=(3.5, 13.0),    # E3.5-E13.0 (neural tube patterning)
            cellular_localization='nucleus',
            functional_role='Dorsal-ventral patterning and progenitor specification'
        )
        
        # PCNA - Proliferation marker
        markers['PCNA'] = MolecularMarker(
            marker_name='PCNA',
            expression_level=0.7,           # Variable with cell cycle
            temporal_window=(3.0, 18.0),    # Throughout active proliferation
            cellular_localization='nucleus_s_phase',
            functional_role='DNA replication and cell cycle progression'
        )
        
        # Ki67 - Cell cycle marker
        markers['Ki67'] = MolecularMarker(
            marker_name='Ki67',
            expression_level=0.6,           # Present in cycling cells
            temporal_window=(3.0, 16.0),    # Active proliferation period
            cellular_localization='nucleus_non_g0',
            functional_role='Cell cycle progression marker (absent in G0)'
        )
        
        # Tbr2 (Eomes) - Intermediate progenitor marker
        markers['Tbr2'] = MolecularMarker(
            marker_name='Tbr2',
            expression_level=0.5,           # Moderate expression in IPs
            temporal_window=(9.0, 15.0),    # E9.0-E15.0 (intermediate progenitor stage)
            cellular_localization='nucleus',
            functional_role='Intermediate progenitor specification'
        )
        
        return markers
    
    def get_marker(self, marker_name: str) -> Optional[MolecularMarker]:
        """Get molecular marker definition by name.
        
        Args:
            marker_name: Name of molecular marker
            
        Returns:
            MolecularMarker object or None if not found
        """
        return self.markers.get(marker_name)
    
    def get_markers_for_cell_type(self, cell_type: NeuroepithelialCellType) -> List[MolecularMarker]:
        """Get appropriate markers for specific cell type.
        
        Args:
            cell_type: Neuroepithelial cell type
            
        Returns:
            List of relevant molecular markers
        """
        if cell_type == NeuroepithelialCellType.EARLY_MULTIPOTENT:
            return [self.markers['Nestin'], self.markers['Sox2'], self.markers['Pax6'], 
                   self.markers['PCNA'], self.markers['Ki67']]
        
        elif cell_type == NeuroepithelialCellType.LATE_MULTIPOTENT:
            return [self.markers['Nestin'], self.markers['Sox2'], self.markers['Pax6'], 
                   self.markers['Ki67']]
        
        elif cell_type == NeuroepithelialCellType.COMMITTED_PROGENITOR:
            return [self.markers['Tbr2'], self.markers['Ki67'], self.markers['PCNA']]
        
        elif cell_type == NeuroepithelialCellType.TRANSITIONING:
            return [self.markers['Sox2'], self.markers['Tbr2'], self.markers['Ki67']]
        
        return []
    
    def calculate_marker_expression(self, cell_type: NeuroepithelialCellType,
                                   developmental_time: float,
                                   morphogen_levels: Dict[str, float]) -> Dict[str, float]:
        """Calculate dynamic marker expression levels.
        
        Args:
            cell_type: Current cell type
            developmental_time: Current developmental time (weeks)
            morphogen_levels: Current morphogen concentration levels
            
        Returns:
            Dictionary of marker expression levels
        """
        markers = self.get_markers_for_cell_type(cell_type)
        expression_levels = {}
        
        for marker in markers:
            # Base expression level
            base_expression = marker.expression_level
            
            # Temporal modulation
            start_time, end_time = marker.temporal_window
            if start_time <= developmental_time <= end_time:
                temporal_factor = 1.0
                
                # Ramp up at beginning
                if developmental_time - start_time < 0.5:
                    temporal_factor = (developmental_time - start_time) / 0.5
                
                # Ramp down at end
                elif end_time - developmental_time < 1.0:
                    temporal_factor = (end_time - developmental_time) / 1.0
                    
            else:
                temporal_factor = 0.1  # Low expression outside window
            
            # Morphogen modulation
            morphogen_factor = self._calculate_morphogen_modulation(marker, morphogen_levels)
            
            # Final expression level
            final_expression = base_expression * temporal_factor * morphogen_factor
            final_expression = np.clip(final_expression, 0.0, 1.0)
            
            expression_levels[marker.marker_name] = final_expression
        
        return expression_levels
    
    def _calculate_morphogen_modulation(self, marker: MolecularMarker, 
                                       morphogen_levels: Dict[str, float]) -> float:
        """Calculate morphogen-dependent modulation of marker expression."""
        modulation_factor = 1.0
        
        # Morphogen-specific modulation rules
        if marker.marker_name == 'Pax6':
            # Pax6 is enhanced by moderate SHH, inhibited by high SHH
            shh_level = morphogen_levels.get('SHH', 0.0)
            if shh_level < 0.3:
                modulation_factor *= (1.0 + 0.5 * shh_level)  # Enhancement
            else:
                modulation_factor *= (1.5 - shh_level)  # Inhibition at high SHH
                
        elif marker.marker_name == 'Sox2':
            # Sox2 is maintained by FGF signaling
            fgf_level = morphogen_levels.get('FGF', 0.0)
            modulation_factor *= (0.5 + 0.5 * fgf_level)  # FGF dependency
            
        elif marker.marker_name == 'Tbr2':
            # Tbr2 expression in intermediate progenitors
            # Enhanced by moderate morphogen levels
            avg_morphogen = np.mean(list(morphogen_levels.values()))
            if 0.2 < avg_morphogen < 0.6:
                modulation_factor *= 1.5  # Enhanced in intermediate conditions
        
        return np.clip(modulation_factor, 0.1, 2.0)
    
    def validate_marker_expression(self, expression_levels: Dict[str, float],
                                  cell_type: NeuroepithelialCellType) -> Dict[str, bool]:
        """Validate marker expression pattern for cell type.
        
        Args:
            expression_levels: Current marker expression levels
            cell_type: Cell type being validated
            
        Returns:
            Dictionary of validation results for each marker
        """
        validation_results = {}
        expected_markers = self.get_markers_for_cell_type(cell_type)
        
        for marker in expected_markers:
            marker_name = marker.marker_name
            
            if marker_name in expression_levels:
                expression = expression_levels[marker_name]
                
                # Validation criteria based on cell type
                if cell_type == NeuroepithelialCellType.EARLY_MULTIPOTENT:
                    # Should have high Nestin, Sox2, Pax6
                    if marker_name in ['Nestin', 'Sox2', 'Pax6']:
                        validation_results[marker_name] = expression > 0.6
                    else:
                        validation_results[marker_name] = True
                        
                elif cell_type == NeuroepithelialCellType.COMMITTED_PROGENITOR:
                    # Should have Tbr2, reduced Sox2
                    if marker_name == 'Tbr2':
                        validation_results[marker_name] = expression > 0.4
                    elif marker_name == 'Sox2':
                        validation_results[marker_name] = expression < 0.5
                    else:
                        validation_results[marker_name] = True
                        
                else:
                    validation_results[marker_name] = expression > 0.1  # Minimum expression
            else:
                validation_results[marker_name] = False  # Missing marker
        
        return validation_results
