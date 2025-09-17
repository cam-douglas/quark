"""
AlphaGenome type definitions - enums and data structures.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class OutputType(Enum):
    """Available prediction output types from AlphaGenome."""
    RNA_SEQ = "rna_seq"  # Gene expression
    ATAC = "atac"  # Chromatin accessibility
    HISTONE_H3K27AC = "histone_h3k27ac"  # Active enhancer marks
    HISTONE_H3K27ME3 = "histone_h3k27me3"  # Repressive marks
    HISTONE_H3K9ME3 = "histone_h3k9me3"  # Heterochromatin marks
    DNASE = "dnase"  # DNase hypersensitivity
    CAGE = "cage"  # Cap analysis gene expression
    CHI_C = "chi_c"  # Chromatin conformation
    CONSERVATION = "conservation"  # Evolutionary conservation
    REGULATORY_SCORE = "regulatory_score"  # Regulatory potential

    def __str__(self):
        return self.value


@dataclass
class GenomicInterval:
    """Represents a genomic interval."""
    chromosome: str
    start: int
    end: int
    strand: str = "+"

    @property
    def length(self):
        return self.end - self.start

    def __str__(self):
        return f"{self.chromosome}:{self.start}-{self.end}({self.strand})"


@dataclass
class Variant:
    """Represents a genetic variant."""
    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str
    variant_id: Optional[str] = None

    def __str__(self):
        return f"{self.chromosome}:{self.position} {self.ref_allele}>{self.alt_allele}"


@dataclass
class PredictionOutputs:
    """Container for AlphaGenome prediction outputs."""

    # Core prediction tracks
    rna_seq: Optional[np.ndarray] = None
    atac: Optional[np.ndarray] = None
    histone_h3k27ac: Optional[np.ndarray] = None
    histone_h3k27me3: Optional[np.ndarray] = None
    histone_h3k9me3: Optional[np.ndarray] = None
    dnase: Optional[np.ndarray] = None
    cage: Optional[np.ndarray] = None
    chi_c: Optional[np.ndarray] = None
    conservation: Optional[np.ndarray] = None
    regulatory_score: Optional[np.ndarray] = None

    # Metadata
    interval: Optional[GenomicInterval] = None
    variant: Optional[Variant] = None
    model_version: str = "1.0"
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    def get_track(self, output_type: Union[str, OutputType]) -> Optional[np.ndarray]:
        """Get prediction track by type."""
        if isinstance(output_type, OutputType):
            output_type = output_type.value
        return getattr(self, output_type, None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for field_name in ['rna_seq', 'atac', 'histone_h3k27ac', 'histone_h3k27me3',
                          'histone_h3k9me3', 'dnase', 'cage', 'chi_c', 'conservation',
                          'regulatory_score']:
            track = getattr(self, field_name)
            if track is not None:
                result[field_name] = track.tolist()

        if self.interval:
            result['interval'] = {
                'chromosome': self.interval.chromosome,
                'start': self.interval.start,
                'end': self.interval.end,
                'strand': self.interval.strand
            }

        result['model_version'] = self.model_version
        result['confidence_scores'] = self.confidence_scores

        return result
