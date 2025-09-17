"""
Role Configuration Management

Configuration management for engineering roles based on .quark/rules/engineering-roles.mdc.

Author: Quark AI
Date: 2025-01-27
"""

from typing import Dict, Any

# Import from the same directory
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from orchestrator_types import AgentRole


def initialize_role_configs() -> Dict[AgentRole, Dict[str, Any]]:
    """Initialize role-specific configurations based on engineering-roles.mdc"""
    return {
        AgentRole.PROGRAM_ORCHESTRATOR: {
            "deliverables": ["/docs/architecture.md", "/docs/roadmap.md"],
            "domain": "end-to-end planning, repos, branching, CI",
            "max_concurrent": 1,  # Only one orchestrator
            "priority_boost": 2
        },
        AgentRole.DEVELOPMENTAL_NEUROBIOLOGIST: {
            "deliverables": ["/biology/neurulation_model.cc3d", "/docs/neurulation_timeline.md"],
            "domain": "stages & timelines",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.MOLECULAR_GENETICIST: {
            "deliverables": ["/grn/*.sbml", "/grn/grn_config.json"],
            "domain": "GRNs (Gene Regulatory Networks)",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.COMPUTATIONAL_BIOLOGIST: {
            "deliverables": ["/data/fate_fields/*.npy"],
            "domain": "fate fields",
            "max_concurrent": 3,
            "priority_boost": 1
        },
        AgentRole.TISSUE_MORPHOGENESIS_ENGINEER: {
            "deliverables": ["/morpho/*.cc3d"],
            "domain": "neurulation simulations",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.CONNECTOMICS_ENGINEER: {
            "deliverables": ["/connectome/connectome.graphml"],
            "domain": "wiring graphs",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.CIRCUIT_BUILDER: {
            "deliverables": ["/circuit/*"],
            "domain": "microcircuits",
            "max_concurrent": 3,
            "priority_boost": 1
        },
        AgentRole.NEUROPLASTICITY_LEARNING_SCIENTIST: {
            "deliverables": ["/learning/plasticity.py", "tests"],
            "domain": "plasticity rules",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.SELF_ORGANIZATION_ENGINEER: {
            "deliverables": ["/self_org/*"],
            "domain": "proto-cortex self-organization",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.MULTIMODAL_SENSORY_ENGINEER: {
            "deliverables": ["/sensors/*.py"],
            "domain": "sensory generators",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.COGNITIVE_ARCHITECT: {
            "deliverables": ["/cognition/*"],
            "domain": "high-level cognition specs",
            "max_concurrent": 1,
            "priority_boost": 2
        },
        AgentRole.NEUROMORPHIC_SYSTEMS_ENGINEER: {
            "deliverables": ["/runtime/*"],
            "domain": "runtime optimization",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.DATA_PIPELINES_ENGINEER: {
            "deliverables": ["/data/registry.yaml"],
            "domain": "data layer",
            "max_concurrent": 3,
            "priority_boost": 1
        },
        AgentRole.EVALUATION_BEHAVIOR_SCIENTIST: {
            "deliverables": ["/eval/*"],
            "domain": "assay benches",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.SAFETY_ETHICS_OFFICER: {
            "deliverables": ["/safety/*"],
            "domain": "guardrails & kill-switch",
            "max_concurrent": 1,
            "priority_boost": 3  # Highest priority for safety
        },
        AgentRole.PRODUCT_HCI: {
            "deliverables": ["/ui/*"],
            "domain": "dashboards & UI",
            "max_concurrent": 2,
            "priority_boost": 1
        },
        AgentRole.QA_REPRODUCIBILITY_ENGINEER: {
            "deliverables": ["/repro/*"],
            "domain": "deterministic CI",
            "max_concurrent": 2,
            "priority_boost": 1
        }
    }
