# brain_modules/complexity_evolution_agent/complexity_evolver.py

"""
Purpose: Main Complexity Evolution Agent that orchestrates progressive enhancement
Inputs: Current development stage, complexity requirements, biological constraints
Outputs: Enhanced documentation, updated agent knowledge, connectome consistency
Dependencies: DocumentEnhancer, ConnectomeSynchronizer, all roadmap documents
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .document_enhancer import DocumentEnhancer
from .connectome_synchronizer import ConnectomeSynchronizer

class ComplexityEvolutionAgent:
    """
    Main agent responsible for progressively enhancing roadmaps, rules, and criteria
    with each development phase to align with human brain complexity.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.current_stage = self._detect_current_stage()
        self.complexity_levels = self._define_complexity_levels()
        self.document_registry = self._build_document_registry()
        self.agent_dependencies = self._map_agent_dependencies()
        
        # Initialize sub-components (with graceful fallback)
        try:
            self.document_enhancer = DocumentEnhancer(project_root)
        except ImportError:
            self.logger.warning("DocumentEnhancer not available - using placeholder")
            self.document_enhancer = None
        
        try:
            self.connectome_synchronizer = ConnectomeSynchronizer(project_root)
        except ImportError as e:
            self.logger.warning(f"ConnectomeSynchronizer not available - using placeholder: {e}")
            self.connectome_synchronizer = None
        
        self.logger.info(f"Complexity Evolution Agent initialized for stage: {self.current_stage}")
    
    def _detect_current_stage(self) -> str:
        """Detect current development stage based on completed pillars"""
        # Check which pillars are completed
        completed_pillars = []
        
        # Check Pillar 1: Basic Neural Dynamics
        if os.path.exists("brain_modules/basal_ganglia/enhanced_gating_system.py"):
            completed_pillars.append("P1")
        
        # Check Pillar 2: Gating & Reinforcement  
        if os.path.exists("brain_modules/basal_ganglia/ENHANCED_PILLAR2_SUMMARY.md"):
            completed_pillars.append("P2")
        
        # Check Pillar 3: Working Memory & Control
        if os.path.exists("docs/pillar_summaries/PILLAR_3_COMPLETION_SUMMARY.md"):
            completed_pillars.append("P3")
        
        # Determine stage based on completed pillars
        if len(completed_pillars) >= 3:
            return "N1"  # Early Postnatal - Ready for Pillar 4
        elif len(completed_pillars) >= 2:
            return "N0"  # Neonate - Pillar 3 in progress
        elif len(completed_pillars) >= 1:
            return "F"   # Fetal - Basic foundation
        else:
            return "F"   # Default to fetal stage
    
    def _define_complexity_levels(self) -> Dict[str, Dict[str, Any]]:
        """Define complexity levels for each development stage"""
        return {
            "F": {  # Fetal Stage
                "name": "Basic Neural Dynamics",
                "complexity_factor": 1.0,
                "document_depth": "foundational",
                "technical_detail": "basic",
                "biological_accuracy": "core_principles",
                "ml_sophistication": "fundamental",
                "consciousness_level": "pre_conscious"
            },
            "N0": {  # Neonate Stage
                "name": "Learning & Consolidation",
                "complexity_factor": 2.5,
                "document_depth": "developmental",
                "technical_detail": "intermediate",
                "biological_accuracy": "developmental_patterns",
                "ml_sophistication": "learning_algorithms",
                "consciousness_level": "proto_conscious"
            },
            "N1": {  # Early Postnatal Stage
                "name": "Enhanced Control & Memory",
                "complexity_factor": 4.0,
                "document_depth": "advanced",
                "technical_detail": "detailed",
                "biological_accuracy": "sophisticated_models",
                "ml_sophistication": "advanced_architectures",
                "consciousness_level": "basic_consciousness"
            },
            "N2": {  # Advanced Postnatal Stage
                "name": "Meta-Control & Simulation",
                "complexity_factor": 6.0,
                "document_depth": "expert",
                "technical_detail": "comprehensive",
                "biological_accuracy": "high_fidelity",
                "ml_sophistication": "state_of_art",
                "consciousness_level": "enhanced_consciousness"
            },
            "N3": {  # Mature Stage
                "name": "Proto-Consciousness Integration",
                "complexity_factor": 8.0,
                "document_depth": "research_grade",
                "technical_detail": "publication_ready",
                "biological_accuracy": "research_validation",
                "ml_sophistication": "research_frontier",
                "consciousness_level": "research_consciousness"
            }
        }
    
    def _build_document_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build registry of all roadmap and rule documents"""
        return {
            "roadmaps": {
                "cognitive_brain_roadmap.md": {
                    "path": ".cursor/rules/cognitive_brain_roadmap.md",
                    "type": "primary_roadmap",
                    "complexity_level": "adaptive",
                    "last_updated": None,
                    "dependencies": ["roles.md", "master-config.mdc"]
                },
                "agi_capabilities.md": {
                    "path": ".cursor/rules/agi_capabilities.md", 
                    "type": "capability_framework",
                    "complexity_level": "adaptive",
                    "last_updated": None,
                    "dependencies": ["cognitive_brain_roadmap.md"]
                },
                "biological_agi_blueprint.md": {
                    "path": ".cursor/rules/biological_agi_blueprint.md",
                    "type": "biological_framework",
                    "complexity_level": "adaptive",
                    "last_updated": None,
                    "dependencies": ["cognitive_brain_roadmap.md"]
                }
            },
            "rules": {
                "ml_workflow.md": {
                    "path": ".cursor/rules/ml_workflow.md",
                    "type": "ml_framework",
                    "complexity_level": "adaptive",
                    "last_updated": None,
                    "dependencies": ["cognitive_brain_roadmap.md"]
                },
                "compliance_review.md": {
                    "path": ".cursor/rules/compliance_review.md",
                    "type": "supreme_authority",
                    "complexity_level": "fixed",  # Supreme authority doesn't change
                    "last_updated": None,
                    "dependencies": []
                },
                "roles.md": {
                    "path": ".cursor/rules/roles.md",
                    "type": "role_framework",
                    "complexity_level": "adaptive",
                    "last_updated": None,
                    "dependencies": ["cognitive_brain_roadmap.md"]
                }
            },
            "configs": {
                "connectome.yaml": {
                    "path": "connectome.yaml",
                    "type": "connectome_config",
                    "complexity_level": "adaptive",
                    "last_updated": None,
                    "dependencies": ["cognitive_brain_roadmap.md"]
                }
            }
        }
    
    def _map_agent_dependencies(self) -> Dict[str, List[str]]:
        """Map which agents depend on which documents"""
        return {
            "cognitive_brain_roadmap.md": [
                "Architecture Agent",
                "Prefrontal Cortex",
                "Basal Ganglia", 
                "Thalamus",
                "Working Memory",
                "Default Mode Network",
                "Hippocampus",
                "Salience Networks"
            ],
            "agi_capabilities.md": [
                "AGI Capability Validator",
                "Cognitive Architect",
                "Evaluation Engineer"
            ],
            "biological_agi_blueprint.md": [
                "Developmental Neurobiologist",
                "Molecular Geneticist",
                "Biological Validator"
            ],
            "ml_workflow.md": [
                "Machine Learning Engineer",
                "Computational Biologist",
                "Training Pipeline Manager"
            ],
            "compliance_review.md": [
                "ALL_AGENTS"  # Supreme authority affects everyone
            ],
            "roles.md": [
                "ALL_AGENTS"  # Role definitions affect everyone
            ],
            "connectome.yaml": [
                "Connectome Engineer",
                "Circuit Builder",
                "Network Expert"
            ]
        }
    
    def evolve_complexity(self, target_stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Evolve complexity of all documents to match target development stage
        
        Args:
            target_stage: Target stage to evolve to (defaults to next stage)
        
        Returns:
            Evolution report with changes made
        """
        if target_stage is None:
            target_stage = self._get_next_stage()
        
        self.logger.info(f"Evolving complexity from {self.current_stage} to {target_stage}")
        
        # Get complexity requirements for target stage
        target_complexity = self.complexity_levels[target_stage]
        
        # Evolve each document
        evolution_report = {
            "from_stage": self.current_stage,
            "to_stage": target_stage,
            "complexity_changes": {},
            "document_updates": {},
            "agent_notifications": {},
            "connectome_updates": {}
        }
        
        # Evolve roadmaps
        if self.document_enhancer:
            for doc_name, doc_info in self.document_registry["roadmaps"].items():
                if doc_info["complexity_level"] == "adaptive":
                    try:
                        changes = self.document_enhancer.enhance_document(
                            doc_info["path"], 
                            target_complexity,
                            doc_info["type"]
                        )
                        evolution_report["document_updates"][doc_name] = changes
                    except Exception as e:
                        self.logger.warning(f"Could not enhance {doc_name}: {str(e)}")
                        evolution_report["document_updates"][doc_name] = {"status": "error", "error": str(e)}
        else:
            evolution_report["document_updates"] = {"status": "document_enhancer_unavailable"}
        
        # Evolve rules
        if self.document_enhancer:
            for doc_name, doc_info in self.document_registry["rules"].items():
                if doc_info["complexity_level"] == "adaptive":
                    try:
                        changes = self.document_enhancer.enhance_document(
                            doc_info["path"],
                            target_complexity,
                            doc_info["type"]
                        )
                        evolution_report["document_updates"][doc_name] = changes
                    except Exception as e:
                        self.logger.warning(f"Could not enhance {doc_name}: {str(e)}")
                        evolution_report["document_updates"][doc_name] = {"status": "error", "error": str(e)}
        
        # Evolve configs
        if self.document_enhancer:
            for doc_name, doc_info in self.document_registry["configs"].items():
                if doc_info["complexity_level"] == "adaptive":
                    try:
                        changes = self.document_enhancer.enhance_document(
                            doc_info["path"],
                            target_complexity,
                            doc_info["type"]
                        )
                        evolution_report["connectome_updates"][doc_name] = changes
                    except Exception as e:
                        self.logger.warning(f"Could not enhance {doc_name}: {str(e)}")
                        evolution_report["connectome_updates"][doc_name] = {"status": "error", "error": str(e)}
        
        # Synchronize all agents with updated information
        if self.connectome_synchronizer:
            try:
                sync_report = self.connectome_synchronizer.synchronize_all_agents(
                    evolution_report["document_updates"]
                )
                evolution_report["agent_notifications"] = sync_report
            except Exception as e:
                self.logger.warning(f"Could not synchronize agents: {str(e)}")
                evolution_report["agent_notifications"] = {"status": "error", "error": str(e)}
        else:
            evolution_report["agent_notifications"] = {"status": "connectome_synchronizer_unavailable"}
        
        # Update current stage
        self.current_stage = target_stage
        
        # Log evolution completion
        self.logger.info(f"Complexity evolution to {target_stage} completed successfully")
        
        return evolution_report
    
    def _get_next_stage(self) -> str:
        """Get the next development stage based on current stage"""
        stage_sequence = ["F", "N0", "N1", "N2", "N3"]
        current_index = stage_sequence.index(self.current_stage)
        
        if current_index < len(stage_sequence) - 1:
            return stage_sequence[current_index + 1]
        else:
            return self.current_stage  # Already at highest stage
    
    def get_complexity_analysis(self) -> Dict[str, Any]:
        """Get current complexity analysis and recommendations"""
        current_complexity = self.complexity_levels[self.current_stage]
        next_stage = self._get_next_stage()
        next_complexity = self.complexity_levels[next_stage]
        
        return {
            "current_stage": self.current_stage,
            "current_complexity": current_complexity,
            "next_stage": next_stage,
            "next_complexity": next_complexity,
            "complexity_gap": {
                "factor_increase": next_complexity["complexity_factor"] / current_complexity["complexity_factor"],
                "document_depth_upgrade": f"{current_complexity['document_depth']} → {next_complexity['document_depth']}",
                "technical_detail_upgrade": f"{current_complexity['technical_detail']} → {next_complexity['technical_detail']}",
                "biological_accuracy_upgrade": f"{current_complexity['biological_accuracy']} → {next_complexity['biological_accuracy']}",
                "ml_sophistication_upgrade": f"{current_complexity['ml_sophistication']} → {next_complexity['ml_sophistication']}",
                "consciousness_level_upgrade": f"{current_complexity['consciousness_level']} → {next_complexity['consciousness_level']}"
            },
            "evolution_ready": next_stage != self.current_stage
        }
    
    def monitor_complexity_alignment(self) -> Dict[str, Any]:
        """Monitor if current implementation aligns with documented complexity"""
        alignment_report = {
            "stage_alignment": {},
            "complexity_gaps": [],
            "recommendations": []
        }
        
        # Check each pillar's complexity alignment
        for pillar in range(1, 6):
            pillar_path = f"docs/pillar_summaries/PILLAR_{pillar}_COMPLETION_SUMMARY.md"
            if os.path.exists(pillar_path):
                # Analyze pillar complexity vs. expected stage complexity
                expected_stage = self._get_expected_stage_for_pillar(pillar)
                expected_complexity = self.complexity_levels[expected_stage]
                
                # This would involve more sophisticated analysis of the actual implementation
                # For now, we'll provide a basic framework
                alignment_report["stage_alignment"][f"Pillar_{pillar}"] = {
                    "expected_stage": expected_stage,
                    "expected_complexity": expected_complexity,
                    "status": "implemented"  # Placeholder for actual analysis
                }
        
        return alignment_report
    
    def _get_expected_stage_for_pillar(self, pillar: int) -> str:
        """Get expected development stage for a given pillar"""
        pillar_stage_mapping = {
            1: "F",    # Basic Neural Dynamics
            2: "N0",   # Gating & Reinforcement
            3: "N1",   # Working Memory & Control
            4: "N2",   # Meta-Control & Simulation
            5: "N3"    # Proto-Consciousness
        }
        return pillar_stage_mapping.get(pillar, "F")
    
    def create_complexity_report(self) -> str:
        """Create a comprehensive complexity evolution report"""
        analysis = self.get_complexity_analysis()
        alignment = self.monitor_complexity_alignment()
        
        report = f"""
# 🧠 Complexity Evolution Report

## Current Status
- **Stage**: {analysis['current_stage']} - {analysis['current_complexity']['name']}
- **Complexity Factor**: {analysis['current_complexity']['complexity_factor']}
- **Document Depth**: {analysis['current_complexity']['document_depth']}
- **Technical Detail**: {analysis['current_complexity']['technical_detail']}
- **Biological Accuracy**: {analysis['current_complexity']['biological_accuracy']}
- **ML Sophistication**: {analysis['current_complexity']['ml_sophistication']}
- **Consciousness Level**: {analysis['current_complexity']['consciousness_level']}

## Next Stage Preview
- **Target Stage**: {analysis['next_stage']} - {analysis['next_complexity']['name']}
- **Complexity Increase**: {analysis['complexity_gap']['factor_increase']:.1f}x
- **Document Upgrade**: {analysis['complexity_gap']['document_depth_upgrade']}
- **Technical Upgrade**: {analysis['complexity_gap']['technical_detail_upgrade']}
- **Biological Upgrade**: {analysis['complexity_gap']['biological_accuracy_upgrade']}
- **ML Upgrade**: {analysis['complexity_gap']['ml_sophistication_upgrade']}
- **Consciousness Upgrade**: {analysis['complexity_gap']['consciousness_level_upgrade']}

## Evolution Readiness
- **Ready for Evolution**: {analysis['evolution_ready']}
- **Current Pillars**: {len([p for p in range(1, 6) if os.path.exists(f'docs/pillar_summaries/PILLAR_{p}_COMPLETION_SUMMARY.md')])}/5 completed

## Recommendations
"""
        
        if analysis['evolution_ready']:
            report += "- ✅ Ready to evolve complexity to next stage\n"
            report += "- 🔄 Consider evolving documents and rules\n"
            report += "- 🔗 Update agent knowledge and connectome\n"
        else:
            report += "- ⏳ Continue current stage development\n"
            report += "- 📚 Focus on completing current pillar\n"
            report += "- 🔍 Review complexity alignment\n"
        
        return report

if __name__ == "__main__":
    # Test the Complexity Evolution Agent
    cea = ComplexityEvolutionAgent()
    
    print("🧠 Complexity Evolution Agent Test")
    print("=" * 50)
    
    # Show current status
    print(cea.create_complexity_report())
    
    # Show complexity analysis
    analysis = cea.get_complexity_analysis()
    print(f"\n📊 Complexity Analysis:")
    print(f"Current Stage: {analysis['current_stage']}")
    print(f"Next Stage: {analysis['next_stage']}")
    print(f"Ready for Evolution: {analysis['evolution_ready']}")
    
    print("\n✅ Complexity Evolution Agent test completed!")
