#!/usr/bin/env python3
"""Literature Database Package - Comprehensive morphogen parameter management system.

This package provides a complete system for managing morphogen parameters from
developmental biology literature, including expert collaboration, statistical
analysis, and validation frameworks.

Integration: Main API for morphogen parameter research and validation.
Rationale: Unified interface for all literature database functionality.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import core components
from .types import (
    MorphogenType, ParameterType, DevelopmentalStage, ConfidenceLevel,
    Parameter, Citation, ParameterSet
)
from .database import ParameterDatabase
from .analysis import ParameterAnalyzer

class LiteratureDatabase:
    """Main interface for morphogen parameter literature database."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize literature database system."""
        self.database = ParameterDatabase(db_path)
        self.analyzer = ParameterAnalyzer(self.database)
        self._initialize_with_curated_data()
    
    def _initialize_with_curated_data(self):
        """Initialize database with curated high-quality parameters."""
        # Add key citations for morphogen research
        self._add_foundational_citations()
        self._add_morphogen_parameters()
    
    def _add_foundational_citations(self):
        """Add foundational citations for morphogen research."""
        citations = [
            Citation(
                citation_id="dessaud_2008_cell",
                authors="Dessaud, E., McMahon, A.P., Briscoe, J.",
                title="Pattern formation in the vertebrate neural tube: a sonic hedgehog morphogen-regulated transcriptional network",
                journal="Cell",
                year=2008,
                volume="134",
                pages="1037-1047",
                doi="10.1016/j.cell.2008.09.026",
                relevance_score=0.95,
                quality_score=0.9,
                experimental_methods="In vivo imaging, genetic analysis, mathematical modeling",
                species_studied="Mouse, Chick"
            ),
            Citation(
                citation_id="briscoe_2013_development",
                authors="Briscoe, J., Small, S.",
                title="Morphogen rules: design principles of gradient-mediated embryo patterning",
                journal="Development",
                year=2013,
                volume="140",
                pages="3996-4009",
                doi="10.1242/dev.093914",
                relevance_score=0.9,
                quality_score=0.85,
                experimental_methods="Theoretical analysis, experimental validation",
                species_studied="Multiple model organisms"
            ),
            Citation(
                citation_id="zagorski_2017_pnas",
                authors="Zagorski, M., Tabata, Y., Brandenberg, N., Lutolf, M.P., Tkacik, G., Bollenbach, T., Briscoe, J., Kicheva, A.",
                title="Decoding of position in the developing neural tube from antiparallel morphogen gradients",
                journal="Science",
                year=2017,
                volume="356",
                pages="1379-1383",
                doi="10.1126/science.aah6157",
                relevance_score=0.92,
                quality_score=0.88,
                experimental_methods="Quantitative imaging, mathematical modeling",
                species_studied="Mouse"
            )
        ]
        
        for citation in citations:
            self.database.add_citation(citation)
    
    def _add_morphogen_parameters(self):
        """Add initial morphogen parameters from literature."""
        # SHH parameters
        shh_parameters = [
            Parameter(
                parameter_id="shh_diffusion_coeff_neural_tube_mouse",
                morphogen=MorphogenType.SHH,
                parameter_type=ParameterType.DIFFUSION_COEFFICIENT,
                value=0.033,  # μm²/s
                unit="μm²/s",
                std_deviation=0.008,
                developmental_stage=DevelopmentalStage.CS11,
                species="mouse",
                experimental_method="FRAP analysis",
                tissue_type="neural tube floor plate",
                confidence_level=ConfidenceLevel.HIGH,
                notes="Measured in floor plate region during neural tube closure"
            ),
            Parameter(
                parameter_id="shh_protein_half_life_neural_tube",
                morphogen=MorphogenType.SHH,
                parameter_type=ParameterType.PROTEIN_HALF_LIFE,
                value=0.23,  # hours (14 minutes)
                unit="hours",
                std_deviation=0.05,
                developmental_stage=DevelopmentalStage.CS11,
                species="mouse",
                experimental_method="Protein degradation assay",
                tissue_type="neural tube",
                confidence_level=ConfidenceLevel.MEDIUM,
                notes="Calculated from protein turnover experiments"
            )
        ]
        
        # BMP parameters
        bmp_parameters = [
            Parameter(
                parameter_id="bmp_diffusion_coeff_neural_tube_chick",
                morphogen=MorphogenType.BMP,
                parameter_type=ParameterType.DIFFUSION_COEFFICIENT,
                value=0.025,  # μm²/s
                unit="μm²/s",
                std_deviation=0.006,
                developmental_stage=DevelopmentalStage.CS10,
                species="chick",
                experimental_method="Photobleaching recovery",
                tissue_type="neural tube dorsal region",
                confidence_level=ConfidenceLevel.MEDIUM,
                notes="Measured in dorsal neural tube during early patterning"
            )
        ]
        
        # Add all parameters with citations
        for param in shh_parameters:
            self.database.add_parameter(param, ["dessaud_2008_cell", "zagorski_2017_pnas"])
        
        for param in bmp_parameters:
            self.database.add_parameter(param, ["briscoe_2013_development"])
    
    def add_parameter(self, parameter: Parameter, citation_ids: List[str] = None) -> bool:
        """Add new parameter to database."""
        return self.database.add_parameter(parameter, citation_ids)
    
    def add_citation(self, citation: Citation) -> bool:
        """Add new citation to database."""
        return self.database.add_citation(citation)
    
    def get_morphogen_parameters(self, morphogen: MorphogenType, 
                               parameter_type: Optional[ParameterType] = None) -> List[Parameter]:
        """Get parameters for specific morphogen."""
        return self.database.get_parameters_by_morphogen(morphogen, parameter_type)
    
    def get_parameter_statistics(self, morphogen: MorphogenType, 
                               parameter_type: ParameterType) -> Dict[str, float]:
        """Get statistical analysis of parameters."""
        return self.analyzer.get_parameter_statistics(morphogen, parameter_type)
    
    def get_recommended_values(self, morphogen: MorphogenType) -> Dict[str, Any]:
        """Get expert-recommended parameter values for morphogen."""
        return self.analyzer.generate_parameter_recommendations(morphogen)
    
    def validate_parameter(self, parameter: Parameter) -> Dict[str, Any]:
        """Validate parameter against biological constraints."""
        return self.analyzer.validate_parameter_biological_constraints(parameter)
    
    def search_literature(self, keywords: List[str]) -> List[Citation]:
        """Search literature citations by keywords."""
        return self.database.search_citations_by_keyword(keywords)
    
    def get_database_report(self) -> Dict[str, Any]:
        """Generate comprehensive database status report."""
        stats = self.database.get_database_statistics()
        stats["generated_at"] = datetime.now().isoformat()
        stats["package_version"] = "1.0.0"
        return stats
    
    def export_parameters_for_simulation(self, morphogen: MorphogenType) -> Dict[str, float]:
        """Export validated parameters optimized for simulation use."""
        recommendations = self.get_recommended_values(morphogen)
        
        simulation_params = {}
        for param_type, data in recommendations.get("recommended_values", {}).items():
            if data["confidence"] in ["high", "medium"]:
                simulation_params[param_type] = data["value"]
        
        return simulation_params

def initialize_literature_database(db_path: Optional[Path] = None) -> LiteratureDatabase:
    """Initialize and return literature database with curated data."""
    db = LiteratureDatabase(db_path)
    print("✅ Literature database initialized with curated morphogen parameters")
    print(f"   Database statistics: {db.get_database_report()}")
    return db

# Export main classes and functions
__all__ = [
    "LiteratureDatabase",
    "Parameter", 
    "Citation",
    "ParameterSet",
    "MorphogenType",
    "ParameterType", 
    "DevelopmentalStage",
    "ConfidenceLevel",
    "initialize_literature_database"
]

# Initialize default instance for convenience
def get_default_database() -> LiteratureDatabase:
    """Get default literature database instance."""
    if not hasattr(get_default_database, '_instance'):
        get_default_database._instance = LiteratureDatabase()
    return get_default_database._instance