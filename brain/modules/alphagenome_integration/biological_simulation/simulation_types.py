#!/usr/bin/env python3
"""Simulation Types for AlphaGenome Biological Simulation.

Defines core data types and enums for biological simulation system.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


class SimulationMode(Enum):
    """Simulation execution modes."""
    REAL_TIME = "real_time"
    ACCELERATED = "accelerated" 
    BATCH = "batch"
    INTERACTIVE = "interactive"


class BiologicalProcess(Enum):
    """Core biological processes in development."""
    NEURAL_INDUCTION = "neural_induction"
    NEURONAL_MIGRATION = "neuronal_migration"
    SYNAPTOGENESIS = "synaptogenesis"
    MYELINATION = "myelination"
    APOPTOSIS = "apoptosis"
    DIFFERENTIATION = "differentiation"


class DevelopmentalStage(Enum):
    """Developmental stages."""
    EMBRYONIC = "embryonic"
    NEURAL_INDUCTION = "neural_induction"
    NEURONAL_MIGRATION = "neuronal_migration"
    SYNAPTOGENESIS = "synaptogenesis"
    MATURATION = "maturation"


@dataclass
class MorphogenGradient:
    """Morphogen gradient configuration."""
    morphogen_name: str
    production_rate: float
    diffusion_rate: float
    decay_rate: float
    source_location: tuple
    gradient_shape: str = "exponential"


@dataclass
class DevelopmentalEvent:
    """Developmental event specification."""
    event_id: str
    event_type: BiologicalProcess
    start_time: float
    duration: float
    affected_regions: List[str]
    parameters: Dict[str, Any]


@dataclass
class SimulationParameters:
    """Complete simulation configuration."""
    simulation_id: str
    mode: SimulationMode = SimulationMode.REAL_TIME
    time_step: float = 0.1  # hours
    total_time: float = 168.0  # 1 week
    save_frequency: float = 1.0  # hours
    visualization_enabled: bool = True
    morphogen_gradients: List[MorphogenGradient] = None
    developmental_events: List[DevelopmentalEvent] = None
    
    def __post_init__(self):
        """Initialize default gradients and events if not provided."""
        if self.morphogen_gradients is None:
            self.morphogen_gradients = self._default_morphogens()
        if self.developmental_events is None:
            self.developmental_events = self._default_events()
    
    def _default_morphogens(self) -> List[MorphogenGradient]:
        """Create default morphogen gradients."""
        return [
            MorphogenGradient("SHH", 1.0, 0.5, 0.1, (0, 0, 0)),
            MorphogenGradient("BMP", 0.8, 0.3, 0.15, (1, 0, 0)),
            MorphogenGradient("WNT", 0.9, 0.4, 0.12, (0, 1, 0)),
            MorphogenGradient("FGF", 1.2, 0.6, 0.08, (0, 0, 1))
        ]
    
    def _default_events(self) -> List[DevelopmentalEvent]:
        """Create default developmental events."""
        return [
            DevelopmentalEvent(
                "neural_induction", BiologicalProcess.NEURAL_INDUCTION,
                0.0, 24.0, ["neural_plate"], {}
            ),
            DevelopmentalEvent(
                "neuronal_migration", BiologicalProcess.NEURONAL_MIGRATION,
                24.0, 72.0, ["cortex", "hippocampus"], {}
            ),
            DevelopmentalEvent(
                "synaptogenesis", BiologicalProcess.SYNAPTOGENESIS,
                96.0, 48.0, ["cortex", "hippocampus", "cerebellum"], {}
            )
        ]
