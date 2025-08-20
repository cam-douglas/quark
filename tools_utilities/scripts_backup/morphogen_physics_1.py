"""
Morphogen Physics for Brain Development

Simulates morphogen diffusion and concentration effects including:
- Diffusion equations
- Concentration gradients
- Morphogen-receptor interactions
- Developmental patterning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MorphogenPhysics:
    """Simulates morphogen physics and diffusion"""
    
    def __init__(self, grid_size: int = 100, domain_size: float = 1.0):
        """
        Initialize morphogen physics simulator
        
        Args:
            grid_size: Number of grid points
            domain_size: Physical domain size in meters
        """
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size
        
        # Initialize morphogen types
        self.morphogens = {
            'shh': {  # Sonic hedgehog
                'diffusion_coefficient': 1e-10,  # m²/s
                'degradation_rate': 1e-3,  # 1/s
                'production_rate': 1e-6,  # mol/m³/s
                'concentration': np.zeros((grid_size, grid_size))
            },
            'bmp': {  # Bone morphogenetic protein
                'diffusion_coefficient': 5e-11,
                'degradation_rate': 2e-3,
                'production_rate': 5e-7,
                'concentration': np.zeros((grid_size, grid_size))
            },
            'wnt': {  # Wnt signaling
                'diffusion_coefficient': 8e-11,
                'degradation_rate': 1.5e-3,
                'production_rate': 8e-7,
                'concentration': np.zeros((grid_size, grid_size))
            }
        }
        
        # Source and sink positions
        self.sources = {}
        self.sinks = {}
        
        # Time tracking
        self.simulation_time = 0.0
        self.time_step = 0.1
        
    def add_morphogen_source(self, 
                           morphogen_type: str,
                           x: int, y: int,
                           production_rate: Optional[float] = None):
        """
        Add a morphogen source at specified position
        
        Args:
            morphogen_type: Type of morphogen
            x: X grid position
            y: Y grid position
            production_rate: Custom production rate
        """
        if morphogen_type not in self.morphogens:
            raise ValueError(f"Unknown morphogen type: {morphogen_type}")
        
        if morphogen_type not in self.sources:
            self.sources[morphogen_type] = []
        
        source = {
            'x': x, 'y': y,
            'production_rate': production_rate or self.morphogens[morphogen_type]['production_rate']
        }
        
        self.sources[morphogen_type].append(source)
        logger.info(f"Added {morphogen_type} source at ({x}, {y})")
    
    def add_morphogen_sink(self, 
                          morphogen_type: str,
                          x: int, y: int,
                          degradation_rate: Optional[float] = None):
        """
        Add a morphogen sink at specified position
        
        Args:
            morphogen_type: Type of morphogen
            x: X grid position
            y: Y grid position
            degradation_rate: Custom degradation rate
        """
        if morphogen_type not in self.morphogens:
            raise ValueError(f"Unknown morphogen type: {morphogen_type}")
        
        if morphogen_type not in self.sinks:
            self.sinks[morphogen_type] = []
        
        sink = {
            'x': x, 'y': y,
            'degradation_rate': degradation_rate or self.morphogens[morphogen_type]['degradation_rate']
        }
        
        self.sinks[morphogen_type].append(sink)
        logger.info(f"Added {morphogen_type} sink at ({x}, {y})")
    
    def step_diffusion(self, steps: int = 1) -> Dict:
        """
        Step the diffusion simulation forward
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            Simulation results
        """
        results = {
            'time': [],
            'concentrations': {},
            'gradients': {},
            'total_mass': {}
        }
        
        for step in range(steps):
            self.simulation_time += self.time_step
            
            # Update each morphogen
            for morphogen_type, morphogen in self.morphogens.items():
                self._update_morphogen_concentration(morphogen_type)
            
            # Record results
            results['time'].append(self.simulation_time)
            
            for morphogen_type in self.morphogens:
                results['concentrations'][morphogen_type] = self.morphogens[morphogen_type]['concentration'].copy()
                results['gradients'][morphogen_type] = self._calculate_gradient(morphogen_type)
                results['total_mass'][morphogen_type] = np.sum(self.morphogens[morphogen_type]['concentration'])
        
        return results
    
    def _update_morphogen_concentration(self, morphogen_type: str):
        """Update concentration for a specific morphogen"""
        morphogen = self.morphogens[morphogen_type]
        D = morphogen['diffusion_coefficient']
        k = morphogen['degradation_rate']
        dt = self.time_step
        
        # Get current concentration
        c = morphogen['concentration']
        
        # Calculate diffusion using finite difference method
        c_new = c.copy()
        
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                # 2D diffusion equation: ∂c/∂t = D∇²c - kc
                laplacian = (c[i+1, j] + c[i-1, j] + c[i, j+1] + c[i, j-1] - 4*c[i, j]) / (self.dx**2)
                
                c_new[i, j] = c[i, j] + dt * (D * laplacian - k * c[i, j])
        
        # Apply sources
        if morphogen_type in self.sources:
            for source in self.sources[morphogen_type]:
                x, y = source['x'], source['y']
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    c_new[x, y] += source['production_rate'] * dt
        
        # Apply sinks
        if morphogen_type in self.sinks:
            for sink in self.sinks[morphogen_type]:
                x, y = sink['x'], sink['y']
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    c_new[x, y] *= (1 - sink['degradation_rate'] * dt)
        
        # Update concentration
        morphogen['concentration'] = c_new
    
    def _calculate_gradient(self, morphogen_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate concentration gradient for a morphogen"""
        c = self.morphogens[morphogen_type]['concentration']
        
        # Calculate gradients using finite differences
        grad_x = np.gradient(c, axis=1) / self.dx
        grad_y = np.gradient(c, axis=0) / self.dx
        
        return grad_x, grad_y
    
    def get_concentration_at_position(self, 
                                    morphogen_type: str,
                                    x: float, y: float) -> float:
        """
        Get morphogen concentration at a specific position
        
        Args:
            morphogen_type: Type of morphogen
            x: X position in meters
            y: Y position in meters
            
        Returns:
            Concentration value
        """
        if morphogen_type not in self.morphogens:
            return 0.0
        
        # Convert to grid coordinates
        grid_x = int(x / self.dx)
        grid_y = int(y / self.dx)
        
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            return self.morphogens[morphogen_type]['concentration'][grid_x, grid_y]
        
        return 0.0
    
    def calculate_morphogen_gradient_magnitude(self, 
                                            morphogen_type: str,
                                            x: float, y: float) -> float:
        """
        Calculate gradient magnitude at a specific position
        
        Args:
            morphogen_type: Type of morphogen
            x: X position in meters
            y: Y position in meters
            
        Returns:
            Gradient magnitude
        """
        if morphogen_type not in self.morphogens:
            return 0.0
        
        # Convert to grid coordinates
        grid_x = int(x / self.dx)
        grid_y = int(y / self.dx)
        
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            grad_x, grad_y = self._calculate_gradient(morphogen_type)
            magnitude = np.sqrt(grad_x[grid_x, grid_y]**2 + grad_y[grid_x, grid_y]**2)
            return magnitude
        
        return 0.0
    
    def simulate_developmental_patterning(self, 
                                        duration: float,
                                        target_pattern: Optional[np.ndarray] = None) -> Dict:
        """
        Simulate developmental patterning over time
        
        Args:
            duration: Simulation duration in seconds
            target_pattern: Target concentration pattern (optional)
            
        Returns:
            Patterning simulation results
        """
        steps = int(duration / self.time_step)
        
        pattern_evolution = {
            'time_points': [],
            'concentration_patterns': {},
            'pattern_similarity': {},
            'gradient_stability': {}
        }
        
        for step in range(steps):
            # Step simulation
            step_results = self.step_diffusion(1)
            
            # Record patterns
            current_time = step * self.time_step
            pattern_evolution['time_points'].append(current_time)
            
            for morphogen_type in self.morphogens:
                if morphogen_type not in pattern_evolution['concentration_patterns']:
                    pattern_evolution['concentration_patterns'][morphogen_type] = []
                
                pattern_evolution['concentration_patterns'][morphogen_type].append(
                    self.morphogens[morphogen_type]['concentration'].copy()
                )
                
                # Calculate pattern similarity if target provided
                if target_pattern is not None:
                    similarity = self._calculate_pattern_similarity(
                        self.morphogens[morphogen_type]['concentration'],
                        target_pattern
                    )
                    
                    if morphogen_type not in pattern_evolution['pattern_similarity']:
                        pattern_evolution['pattern_similarity'][morphogen_type] = []
                    
                    pattern_evolution['pattern_similarity'][morphogen_type].append(similarity)
                
                # Calculate gradient stability
                grad_x, grad_y = self._calculate_gradient(morphogen_type)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                stability = np.std(gradient_magnitude)  # Lower std = more stable
                
                if morphogen_type not in pattern_evolution['gradient_stability']:
                    pattern_evolution['gradient_stability'][morphogen_type] = []
                
                pattern_evolution['gradient_stability'][morphogen_type].append(stability)
        
        logger.info(f"Developmental patterning simulation completed: {duration}s")
        return pattern_evolution
    
    def _calculate_pattern_similarity(self, 
                                    pattern1: np.ndarray, 
                                    pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns using correlation"""
        # Flatten arrays and calculate correlation
        flat1 = pattern1.flatten()
        flat2 = pattern2.flatten()
        
        # Normalize
        flat1 = (flat1 - np.mean(flat1)) / (np.std(flat1) + 1e-8)
        flat2 = (flat2 - np.mean(flat2)) / (np.std(flat2) + 1e-8)
        
        # Calculate correlation
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def get_morphogen_summary(self) -> Dict:
        """Get summary of all morphogen states"""
        summary = {}
        
        for morphogen_type, morphogen in self.morphogens.items():
            concentration = morphogen['concentration']
            
            summary[morphogen_type] = {
                'total_concentration': np.sum(concentration),
                'max_concentration': np.max(concentration),
                'min_concentration': np.min(concentration),
                'mean_concentration': np.mean(concentration),
                'concentration_std': np.std(concentration),
                'gradient_magnitude': np.mean(np.sqrt(
                    self._calculate_gradient(morphogen_type)[0]**2 + 
                    self._calculate_gradient(morphogen_type)[1]**2
                ))
            }
        
        return summary
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        for morphogen_type in self.morphogens:
            self.morphogens[morphogen_type]['concentration'] = np.zeros((self.grid_size, self.grid_size))
        
        self.simulation_time = 0.0
        logger.info("Morphogen simulation reset to initial state")
