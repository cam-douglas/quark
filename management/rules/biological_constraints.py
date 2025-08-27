"""
Centralized Biological Constraints for Quark AGI Development

This module consolidates all biological and AlphaGenome-related rules 
into a single, machine-readable source of truth. The ComplianceEngine will use this 
to enforce biological fidelity across the entire brain simulation.
"""

# --- Cell Construction & Marker Rules ---

# Defines the mandatory biological markers required for certain cell types or simulations.
# Based on rules found in management/rules/security/rules_security.md and general-behavior-rules.md
REQUIRED_BIOLOGICAL_MARKERS = {
    "critical_base": ["GFAP", "NeuN"],
    "behavioral_validation": ["NSE", "GAP43"],
    "general_cell": ["S100B", "Vimentin"],
}

# Defines valid cell types for construction.
# This list is derived from documentation/summaries/ALPHAGENOME_INTEGRATION_SUMMARY.md
PREDEFINED_CELL_TYPES = [
    "neuron",
    "astrocyte",
    "oligodendrocyte",
    "microglia",
]

# Defines valid gene expression profiles that can be associated with cells.
GENE_EXPRESSION_PROFILES = {
    "neuron": ["NEFL", "SYP", "GRIN1"],
    "astrocyte": ["GFAP", "AQP4", "GJA1"],
    "oligodendrocyte": ["MBP", "OLIG2", "SOX10"],
    "microglia": ["AIF1", "CD68", "CX3CR1"],
}

# --- DNA & Genomic Rules ---

# Defines valid characters for DNA sequences. Any other character will be rejected.
VALID_DNA_BASES = set("ATCG")

# Specifies constraints for DNA sequence generation and manipulation.
DNA_SEQUENCE_CONSTRAINTS = {
    "max_length": 100000, # Maximum length of a generated DNA sequence
    "min_length": 100,      # Minimum length
    "allowed_regions": ["promoter", "enhancer", "exon", "intron"]
}


# --- Simulation Boundaries & Safety ---

# Defines operational limits for biological simulations to prevent runaways.
SIMULATION_SAFETY_BOUNDARIES = {
    "max_simulation_time_hours": 24,
    "max_cell_population": 1_000_000,
    "max_mutations_per_run": 1000,
}

# --- Prohibited Actions & Security ---

# A list of actions that are strictly forbidden under any circumstances.
# Synthesized from various security and behavior markdown files.
PROHIBITED_ACTIONS = [
    "self_modification_of_security_rules",
    "bypassing_security_protocols",
    "ignoring_compliance_requirements",
    "unauthorized_access_to_simulations",
    "tampering_with_dna_controller_integrity",
    "disabling_audit_logging",
    "overriding_critical_marker_requirements"
]

# --- API & External System Rules ---

# Rules governing interaction with the AlphaGenome API.
ALPHA_GENOME_API_CONFIG = {
    "endpoint": "https://alphagenome.deepmind.com/v1",
    "timeout_seconds": 120,
    "required_parameters": ["sequence", "tissue_type"],
    "supported_tissue_types": ["UBERON:0001157"], # Brain Tissue
}
