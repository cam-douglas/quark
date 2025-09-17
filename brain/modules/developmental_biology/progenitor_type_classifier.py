"""
Progenitor Type Classifier

This module determines progenitor types based on morphogen exposure
and manages lineage potency definitions.

Author: Quark AI
Date: 2025-01-27
"""

import numpy as np
from typing import Dict, Set
from dataclasses import dataclass
from enum import Enum
from .neuroepithelial_cells import NeuroepithelialCell


class NeuronType(Enum):
    """Types of neurons that can be generated"""
    MOTOR_NEURON = "motor_neuron"
    INTERNEURON_V0 = "interneuron_v0"
    INTERNEURON_V1 = "interneuron_v1"
    INTERNEURON_V2 = "interneuron_v2"
    INTERNEURON_V3 = "interneuron_v3"
    DORSAL_INTERNEURON = "dorsal_interneuron"
    NEURAL_CREST = "neural_crest"
    OLIGODENDROCYTE = "oligodendrocyte"
    ASTROCYTE = "astrocyte"


class CommitmentLevel(Enum):
    """Levels of lineage commitment"""
    MULTIPOTENT = "multipotent"
    BIPOTENT = "bipotent"
    UNIPOTENT = "unipotent"
    COMMITTED = "committed"


@dataclass
class LineagePotency:
    """Defines the lineage potency of a committed progenitor"""
    commitment_level: CommitmentLevel
    allowed_fates: Set[NeuronType]
    restriction_factors: Dict[str, float]
    competency_window: tuple[float, float]


class ProgenitorTypeClassifier:
    """
    Determines progenitor types based on morphogen exposure
    and manages lineage potency definitions.
    """
    
    def __init__(self):
        """Initialize progenitor type classifier"""
        self.lineage_potencies: Dict[str, LineagePotency] = {}
        self._setup_lineage_potencies()
    
    def _setup_lineage_potencies(self) -> None:
        """Setup lineage potency definitions for different progenitor types"""
        self.lineage_potencies = {
            "motor_neuron_progenitor": LineagePotency(
                commitment_level=CommitmentLevel.UNIPOTENT,
                allowed_fates={NeuronType.MOTOR_NEURON},
                restriction_factors={"SHH": 0.9, "retinoic_acid": 0.8},
                competency_window=(8.5, 11.0)
            ),
            "ventral_interneuron_progenitor": LineagePotency(
                commitment_level=CommitmentLevel.MULTIPOTENT,
                allowed_fates={NeuronType.INTERNEURON_V0, NeuronType.INTERNEURON_V1, 
                             NeuronType.INTERNEURON_V2, NeuronType.INTERNEURON_V3},
                restriction_factors={"SHH": 0.7, "BMP": 0.3},
                competency_window=(8.0, 12.0)
            ),
            "dorsal_interneuron_progenitor": LineagePotency(
                commitment_level=CommitmentLevel.BIPOTENT,
                allowed_fates={NeuronType.DORSAL_INTERNEURON, NeuronType.NEURAL_CREST},
                restriction_factors={"BMP": 0.8, "WNT": 0.6},
                competency_window=(8.0, 10.5)
            ),
            "glial_progenitor": LineagePotency(
                commitment_level=CommitmentLevel.BIPOTENT,
                allowed_fates={NeuronType.OLIGODENDROCYTE, NeuronType.ASTROCYTE},
                restriction_factors={"PDGF": 0.7, "FGF": 0.5},
                competency_window=(10.0, 16.0)
            )
        }
    
    def determine_progenitor_type(self, cell: NeuroepithelialCell, 
                                morphogen_concentrations: Dict[str, float]) -> str:
        """Determine the type of committed progenitor based on morphogen exposure"""
        shh_level = morphogen_concentrations.get("SHH", 0.0)
        bmp_level = morphogen_concentrations.get("BMP", 0.0)
        wnt_level = morphogen_concentrations.get("WNT", 0.0)
        
        if shh_level > 0.8:
            return "motor_neuron_progenitor"
        elif shh_level > 0.5:
            return "ventral_interneuron_progenitor"
        elif bmp_level > 0.6:
            return "dorsal_interneuron_progenitor"
        elif wnt_level > 0.4:
            return "glial_progenitor"
        else:
            return "ventral_interneuron_progenitor"
    
    def get_lineage_potency(self, progenitor_type: str) -> LineagePotency:
        """Get lineage potency for progenitor type"""
        return self.lineage_potencies.get(progenitor_type)
    
    def validate_competency_window(self, progenitor_type: str, developmental_time: float) -> bool:
        """Validate if developmental time is within competency window"""
        if progenitor_type not in self.lineage_potencies:
            return False
        
        potency = self.lineage_potencies[progenitor_type]
        return potency.competency_window[0] <= developmental_time <= potency.competency_window[1]
    
    def generate_molecular_markers(self, progenitor_type: str, 
                                 morphogen_concentrations: Dict[str, float]) -> Dict[str, float]:
        """Generate molecular markers for committed progenitor"""
        markers = {}
        
        if progenitor_type == "motor_neuron_progenitor":
            markers.update({
                "Olig2": 0.9,
                "Nkx2.2": 0.8,
                "Isl1": 0.7,
                "HB9": 0.6
            })
        elif progenitor_type == "ventral_interneuron_progenitor":
            markers.update({
                "Nkx2.2": 0.7,
                "Dbx1": 0.6,
                "Irx3": 0.5,
                "En1": 0.4
            })
        elif progenitor_type == "dorsal_interneuron_progenitor":
            markers.update({
                "Pax3": 0.8,
                "Pax7": 0.7,
                "Msx1": 0.6,
                "Lmx1b": 0.5
            })
        elif progenitor_type == "glial_progenitor":
            markers.update({
                "Olig1": 0.8,
                "Sox10": 0.7,
                "PDGFRA": 0.6,
                "NG2": 0.5
            })
        
        # Adjust markers based on morphogen concentrations
        for marker, base_level in markers.items():
            morphogen_influence = sum(morphogen_concentrations.values()) / len(morphogen_concentrations)
            markers[marker] = min(1.0, base_level * (1.0 + morphogen_influence * 0.2))
        
        return markers
    
    def determine_most_likely_fate(self, molecular_markers: Dict[str, float], 
                                 allowed_fates: Set[NeuronType]) -> NeuronType:
        """Determine most likely fate based on molecular markers"""
        if molecular_markers.get("Olig2", 0) > 0.8:
            return NeuronType.MOTOR_NEURON
        elif molecular_markers.get("Pax3", 0) > 0.7:
            return NeuronType.DORSAL_INTERNEURON
        elif molecular_markers.get("Sox10", 0) > 0.6:
            return NeuronType.OLIGODENDROCYTE
        else:
            return list(allowed_fates)[0]
    
    def restrict_lineage_potency(self, current_potency: LineagePotency, 
                               time_since_commitment: float) -> LineagePotency:
        """Restrict lineage potency over time"""
        restriction_factor = min(1.0, time_since_commitment / 2.0)
        
        if restriction_factor > 0.5 and len(current_potency.allowed_fates) > 1:
            # Restrict to single fate
            restricted_fates = {list(current_potency.allowed_fates)[0]}
            new_commitment_level = CommitmentLevel.COMMITTED
        else:
            restricted_fates = current_potency.allowed_fates
            new_commitment_level = current_potency.commitment_level
        
        return LineagePotency(
            commitment_level=new_commitment_level,
            allowed_fates=restricted_fates,
            restriction_factors=current_potency.restriction_factors,
            competency_window=current_potency.competency_window
        )
