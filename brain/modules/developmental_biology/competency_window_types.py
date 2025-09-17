#!/usr/bin/env python3
"""Competency Window Type Definitions.

Type definitions for temporal competency windows including fate specifications,
competency curves, and restriction mechanisms for progenitor cells.

Integration: Type definitions for competency modeling system
Rationale: Centralized competency type definitions with biological accuracy
"""

from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

class FateType(Enum):
    """Neural cell fate types with temporal competency."""
    MOTOR_NEURON = "motor_neuron"           # Ventral motor neurons (E8.5-E11.0)
    INTERNEURON_V0 = "interneuron_v0"       # V0 interneurons (E8.5-E10.5)
    INTERNEURON_V1 = "interneuron_v1"       # V1 interneurons (E9.0-E11.5)
    INTERNEURON_V2 = "interneuron_v2"       # V2 interneurons (E9.0-E12.0)
    INTERNEURON_DORSAL = "interneuron_dorsal"  # Dorsal interneurons (E8.0-E12.0)
    NEURAL_CREST = "neural_crest"           # Neural crest cells (E8.0-E10.0)
    OLIGODENDROCYTE = "oligodendrocyte"     # Oligodendrocyte progenitors (E10.0-E14.0)
    ASTROCYTE = "astrocyte"                 # Astrocyte progenitors (E11.0-E16.0)
    EPENDYMAL = "ependymal"                 # Ependymal cells (E9.0-E13.0)

class CompetencyProfile(Enum):
    """Competency profile shapes over time."""
    EARLY_PEAK = "early_peak"               # Peak early, decline rapidly
    SUSTAINED = "sustained"                 # Sustained over long period
    LATE_ONSET = "late_onset"              # Low early, peak late
    BELL_CURVE = "bell_curve"              # Gaussian-like curve
    STEP_FUNCTION = "step_function"        # Sharp on/off

class RestrictionMechanism(Enum):
    """Mechanisms for competency restriction."""
    TEMPORAL_DECAY = "temporal_decay"       # Natural temporal decline
    MORPHOGEN_INHIBITION = "morphogen_inhibition"  # Morphogen-mediated restriction
    EPIGENETIC_SILENCING = "epigenetic_silencing"  # Chromatin modifications
    TRANSCRIPTIONAL_REPRESSION = "transcriptional_repression"  # TF-mediated
    METABOLIC_CONSTRAINT = "metabolic_constraint"  # Energy/resource limits

@dataclass
class CompetencyWindow:
    """Temporal competency window for specific cell fate."""
    fate_type: FateType
    competency_start: float                 # Start time (developmental weeks)
    competency_peak: float                  # Peak competency time
    competency_end: float                   # End time
    max_competency_strength: float          # Maximum competency level (0-1)
    competency_profile: CompetencyProfile   # Shape of competency curve
    required_morphogens: List[str]          # Required morphogen signals
    inhibitory_morphogens: List[str]        # Inhibitory signals
    restriction_mechanisms: List[RestrictionMechanism]  # How competency is lost

@dataclass
class CompetencyState:
    """Current competency state for a progenitor cell."""
    cell_id: str
    current_competencies: Dict[FateType, float]  # Current competency levels
    competency_history: List[Dict[FateType, float]]  # Historical competency
    restricted_fates: Set[FateType]         # Fates that are permanently restricted
    competency_last_update: float           # Last update time
    
@dataclass
class FateRestrictionEvent:
    """Event representing fate restriction."""
    fate_type: FateType
    restriction_time: float                 # When restriction occurred
    restriction_mechanism: RestrictionMechanism
    triggering_signal: Optional[str]       # What triggered restriction
    irreversible: bool                      # Whether restriction can be reversed
