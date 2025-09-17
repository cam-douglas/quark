#!/usr/bin/env python3
"""WNT/FGF Patterning Analysis System.

Analyzes anterior-posterior patterning quality, regional boundaries, and
cross-regulation effectiveness for integrated WNT/FGF gradient systems.

Integration: Analysis component for WNT/FGF integration system
Rationale: Focused patterning analysis separated from main integration coordinator
"""

from typing import Dict, Any, List
import numpy as np
import logging

from .spatial_grid import GridDimensions
from .wnt_fgf_types import RegionalMarker, RegionalSpecification

logger = logging.getLogger(__name__)

class WNTFGFPatternAnalyzer:
    """Analyzer for WNT/FGF anterior-posterior patterning.
    
    Provides comprehensive analysis of A-P patterning quality including
    regional boundary detection, gradient strength assessment, and
    cross-regulation effectiveness validation.
    """
    
    def __init__(self, grid_dimensions: GridDimensions,
                 regional_specs: Dict[RegionalMarker, RegionalSpecification]):
        """Initialize WNT/FGF pattern analyzer.
        
        Args:
            grid_dimensions: 3D spatial grid dimensions
            regional_specs: Regional specification definitions
        """
        self.dims = grid_dimensions
        self.regional_specs = regional_specs
        
        logger.info("Initialized WNTFGFPatternAnalyzer")
    
    def analyze_regional_boundaries(self, wnt_field: np.ndarray, 
                                  fgf_field: np.ndarray) -> Dict[str, Any]:
        """Analyze regional boundaries based on morphogen concentrations."""
        # Create A-P profiles (average over x,z dimensions)
        wnt_profile = np.mean(wnt_field, axis=(0, 2))
        fgf_profile = np.mean(fgf_field, axis=(0, 2))
        
        # Identify regions based on concentration thresholds
        regional_boundaries = {}
        
        for region_marker, spec in self.regional_specs.items():
            # Find positions where both WNT and FGF are in specified ranges
            wnt_in_range = ((wnt_profile >= spec.wnt_concentration_range[0]) & 
                           (wnt_profile <= spec.wnt_concentration_range[1]))
            fgf_in_range = ((fgf_profile >= spec.fgf_concentration_range[0]) & 
                           (fgf_profile <= spec.fgf_concentration_range[1]))
            
            region_mask = wnt_in_range & fgf_in_range
            
            if np.any(region_mask):
                positions = np.where(region_mask)[0] / self.dims.y_size  # Normalized positions
                regional_boundaries[region_marker.value] = {
                    "detected": True,
                    "anterior_boundary": float(np.min(positions)),
                    "posterior_boundary": float(np.max(positions)),
                    "extent": float(np.max(positions) - np.min(positions)),
                    "characteristic_genes": spec.characteristic_genes
                }
            else:
                regional_boundaries[region_marker.value] = {
                    "detected": False,
                    "characteristic_genes": spec.characteristic_genes
                }
        
        return regional_boundaries
    
    def calculate_patterning_metrics(self, wnt_field: np.ndarray, 
                                   fgf_field: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for A-P patterning."""
        # Gradient strength (how well-defined are the gradients)
        wnt_gradient_strength = self._calculate_gradient_strength(wnt_field)
        fgf_gradient_strength = self._calculate_gradient_strength(fgf_field)
        
        # Regional separation (how distinct are the regions)
        regional_separation = self._calculate_regional_separation(wnt_field, fgf_field)
        
        # Cross-regulation effectiveness
        cross_regulation = self._calculate_cross_regulation_effectiveness(wnt_field, fgf_field)
        
        return {
            "wnt_gradient_strength": float(wnt_gradient_strength),
            "fgf_gradient_strength": float(fgf_gradient_strength),
            "regional_separation_score": float(regional_separation),
            "cross_regulation_score": float(cross_regulation),
            "overall_patterning_score": float((wnt_gradient_strength + fgf_gradient_strength + 
                                             regional_separation + cross_regulation) / 4.0)
        }
    
    def _calculate_gradient_strength(self, field: np.ndarray) -> float:
        """Calculate gradient strength along A-P axis."""
        # Calculate gradient along y-axis (A-P direction)
        dy = self.dims.resolution * 1e-6  # Convert to m
        
        gradient_y = np.zeros_like(field)
        gradient_y[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2 * dy)
        
        # Average gradient magnitude
        gradient_magnitude = np.abs(gradient_y)
        mean_gradient = np.mean(gradient_magnitude)
        
        # Normalize by maximum concentration
        max_concentration = np.max(field)
        normalized_gradient = mean_gradient / max_concentration if max_concentration > 0 else 0.0
        
        return normalized_gradient
    
    def _calculate_regional_separation(self, wnt_field: np.ndarray, 
                                     fgf_field: np.ndarray) -> float:
        """Calculate how well regions are separated."""
        # Calculate variance in regional marker expression
        regional_variances = []
        
        for region_marker, spec in self.regional_specs.items():
            # Check if concentrations fall within expected ranges
            wnt_in_range = ((wnt_field >= spec.wnt_concentration_range[0]) & 
                           (wnt_field <= spec.wnt_concentration_range[1]))
            fgf_in_range = ((fgf_field >= spec.fgf_concentration_range[0]) & 
                           (fgf_field <= spec.fgf_concentration_range[1]))
            
            region_match = wnt_in_range & fgf_in_range
            region_coverage = np.sum(region_match) / region_match.size
            regional_variances.append(region_coverage)
        
        # Good separation means high variance between regions
        separation_score = np.std(regional_variances) if len(regional_variances) > 1 else 0.0
        
        return separation_score
    
    def _calculate_cross_regulation_effectiveness(self, wnt_field: np.ndarray,
                                                fgf_field: np.ndarray) -> float:
        """Calculate effectiveness of WNT-FGF cross-regulation."""
        # WNT and FGF should show synergistic patterns in posterior regions
        
        # Posterior region analysis
        posterior_region = slice(2*self.dims.y_size//3, self.dims.y_size)
        wnt_posterior = wnt_field[:, posterior_region, :]
        fgf_posterior = fgf_field[:, posterior_region, :]
        
        # Calculate correlation in posterior region (should be positive)
        if wnt_posterior.size > 0 and fgf_posterior.size > 0:
            correlation = np.corrcoef(wnt_posterior.flatten(), fgf_posterior.flatten())[0, 1]
            cross_regulation_score = max(0.0, correlation)  # Positive correlation only
        else:
            cross_regulation_score = 0.0
        
        return cross_regulation_score
    
    def calculate_posterior_enrichment(self, wnt_field: np.ndarray) -> float:
        """Calculate WNT enrichment in posterior regions."""
        # Compare posterior vs anterior concentrations
        anterior_region = wnt_field[:, :self.dims.y_size//3, :]  # First third
        posterior_region = wnt_field[:, 2*self.dims.y_size//3:, :]  # Last third
        
        anterior_mean = np.mean(anterior_region)
        posterior_mean = np.mean(posterior_region)
        
        enrichment = posterior_mean / anterior_mean if anterior_mean > 0 else 0.0
        
        return enrichment
    
    def calculate_isthmus_peak(self, fgf_field: np.ndarray) -> float:
        """Calculate FGF peak at isthmic organizer position."""
        # Isthmus is around 40% along A-P axis
        isthmus_position = int(0.4 * self.dims.y_size)
        isthmus_width = max(1, int(0.1 * self.dims.y_size))
        
        y_start = max(0, isthmus_position - isthmus_width)
        y_end = min(self.dims.y_size, isthmus_position + isthmus_width)
        
        isthmus_region = fgf_field[:, y_start:y_end, :]
        isthmus_peak = np.max(isthmus_region)
        
        return isthmus_peak
    
    def calculate_neural_coverage(self, fgf_field: np.ndarray) -> float:
        """Calculate neural maintenance coverage by FGF."""
        # FGF should maintain neural progenitors throughout neural tube
        maintenance_threshold = 0.1  # nM
        
        neural_coverage = np.sum(fgf_field > maintenance_threshold) / fgf_field.size
        
        return neural_coverage
