#!/usr/bin/env python3
"""
Brain Region Mapper
Maps different types of knowledge to appropriate brain regions based on biological learning patterns
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

class BrainRegionMapper:
    def __init__(self, database_path: str = "database"):
        self.database_path = database_path
        self.brain_regions = self._initialize_brain_regions()
        self.knowledge_mappings = self._initialize_knowledge_mappings()
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_brain_regions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize brain regions with their learning capabilities"""
        return {
            "prefrontal_cortex": {
                "name": "Prefrontal Cortex",
                "function": "Executive control, planning, decision-making, working memory",
                "learning_capabilities": ["executive_functions", "planning", "reasoning", "meta_cognition"],
                "knowledge_types": ["strategic_planning", "decision_making", "goal_management", "cognitive_control"],
                "capacity": 1000,
                "current_usage": 0,
                "plasticity_rate": 0.8
            },
            "hippocampus": {
                "name": "Hippocampus",
                "function": "Episodic memory, spatial navigation, pattern completion",
                "learning_capabilities": ["episodic_memory", "spatial_memory", "pattern_recognition", "consolidation"],
                "knowledge_types": ["experiences", "spatial_information", "temporal_sequences", "context_memory"],
                "capacity": 2000,
                "current_usage": 0,
                "plasticity_rate": 0.9
            },
            "amygdala": {
                "name": "Amygdala",
                "function": "Emotional processing, fear conditioning, reward learning",
                "learning_capabilities": ["emotional_memory", "fear_conditioning", "reward_processing", "social_learning"],
                "knowledge_types": ["emotional_states", "threat_assessment", "reward_values", "social_cues"],
                "capacity": 500,
                "current_usage": 0,
                "plasticity_rate": 0.7
            },
            "basal_ganglia": {
                "name": "Basal Ganglia",
                "function": "Action selection, habit formation, motor learning",
                "learning_capabilities": ["action_selection", "habit_formation", "motor_learning", "reinforcement_learning"],
                "knowledge_types": ["action_patterns", "motor_skills", "habitual_behaviors", "reward_prediction"],
                "capacity": 800,
                "current_usage": 0,
                "plasticity_rate": 0.6
            },
            "cerebellum": {
                "name": "Cerebellum",
                "function": "Motor coordination, timing, procedural learning",
                "learning_capabilities": ["motor_coordination", "timing", "procedural_learning", "error_correction"],
                "knowledge_types": ["motor_skills", "timing_patterns", "procedures", "coordination"],
                "capacity": 1500,
                "current_usage": 0,
                "plasticity_rate": 0.8
            },
            "visual_cortex": {
                "name": "Visual Cortex",
                "function": "Visual processing, object recognition, spatial awareness",
                "learning_capabilities": ["visual_recognition", "spatial_processing", "object_detection", "visual_memory"],
                "knowledge_types": ["visual_patterns", "spatial_relationships", "object_representations", "visual_scenes"],
                "capacity": 1200,
                "current_usage": 0,
                "plasticity_rate": 0.7
            },
            "auditory_cortex": {
                "name": "Auditory Cortex",
                "function": "Auditory processing, speech recognition, sound localization",
                "learning_capabilities": ["auditory_recognition", "speech_processing", "sound_localization", "auditory_memory"],
                "knowledge_types": ["sound_patterns", "speech_representations", "auditory_scenes", "temporal_sequences"],
                "capacity": 1000,
                "current_usage": 0,
                "plasticity_rate": 0.7
            },
            "somatosensory_cortex": {
                "name": "Somatosensory Cortex",
                "function": "Touch processing, body awareness, proprioception",
                "learning_capabilities": ["tactile_learning", "body_awareness", "proprioception", "sensory_integration"],
                "knowledge_types": ["tactile_patterns", "body_representations", "sensory_maps", "touch_memory"],
                "capacity": 800,
                "current_usage": 0,
                "plasticity_rate": 0.6
            },
            "temporal_cortex": {
                "name": "Temporal Cortex",
                "function": "Language processing, semantic memory, face recognition",
                "learning_capabilities": ["language_processing", "semantic_memory", "face_recognition", "conceptual_learning"],
                "knowledge_types": ["semantic_concepts", "language_representations", "face_memory", "conceptual_knowledge"],
                "capacity": 1500,
                "current_usage": 0,
                "plasticity_rate": 0.8
            },
            "parietal_cortex": {
                "name": "Parietal Cortex",
                "function": "Spatial attention, numerical processing, multisensory integration",
                "learning_capabilities": ["spatial_attention", "numerical_processing", "multisensory_integration", "spatial_reasoning"],
                "knowledge_types": ["spatial_representations", "numerical_concepts", "attention_patterns", "spatial_reasoning"],
                "capacity": 1000,
                "current_usage": 0,
                "plasticity_rate": 0.7
            },
            "thalamus": {
                "name": "Thalamus",
                "function": "Sensory relay, attention modulation, consciousness",
                "learning_capabilities": ["sensory_relay", "attention_modulation", "consciousness_integration", "information_routing"],
                "knowledge_types": ["sensory_integration", "attention_patterns", "consciousness_states", "information_flow"],
                "capacity": 600,
                "current_usage": 0,
                "plasticity_rate": 0.5
            },
            "brainstem": {
                "name": "Brainstem",
                "function": "Autonomic functions, arousal, basic reflexes",
                "learning_capabilities": ["autonomic_learning", "arousal_regulation", "reflex_conditioning", "basic_learning"],
                "knowledge_types": ["autonomic_patterns", "arousal_states", "reflex_responses", "basic_behaviors"],
                "capacity": 300,
                "current_usage": 0,
                "plasticity_rate": 0.4
            }
        }
    
    def _initialize_knowledge_mappings(self) -> Dict[str, List[str]]:
        """Initialize mappings from knowledge types to brain regions"""
        return {
            # Neuroscience knowledge
            "neural_activity": ["hippocampus", "prefrontal_cortex", "temporal_cortex"],
            "brain_connectivity": ["prefrontal_cortex", "hippocampus", "thalamus"],
            "cognitive_functions": ["prefrontal_cortex", "temporal_cortex", "parietal_cortex"],
            "sensory_processing": ["visual_cortex", "auditory_cortex", "somatosensory_cortex"],
            "motor_control": ["cerebellum", "basal_ganglia", "motor_cortex"],
            "emotional_processing": ["amygdala", "prefrontal_cortex", "hippocampus"],
            
            # Biochemistry knowledge
            "molecular_structures": ["temporal_cortex", "parietal_cortex"],
            "metabolic_pathways": ["temporal_cortex", "prefrontal_cortex"],
            "protein_functions": ["temporal_cortex", "hippocampus"],
            "enzymatic_reactions": ["temporal_cortex", "cerebellum"],
            
            # Physics knowledge
            "mathematical_concepts": ["parietal_cortex", "prefrontal_cortex"],
            "spatial_relationships": ["parietal_cortex", "visual_cortex"],
            "temporal_patterns": ["cerebellum", "temporal_cortex"],
            "causal_relationships": ["prefrontal_cortex", "temporal_cortex"],
            
            # Computer science knowledge
            "algorithmic_thinking": ["prefrontal_cortex", "parietal_cortex"],
            "logical_reasoning": ["prefrontal_cortex", "temporal_cortex"],
            "pattern_recognition": ["temporal_cortex", "visual_cortex"],
            "procedural_knowledge": ["cerebellum", "basal_ganglia"],
            
            # Psychology knowledge
            "behavioral_patterns": ["basal_ganglia", "amygdala"],
            "social_cognition": ["temporal_cortex", "amygdala"],
            "memory_formation": ["hippocampus", "prefrontal_cortex"],
            "attention_processes": ["parietal_cortex", "thalamus"],
            
            # General knowledge
            "semantic_knowledge": ["temporal_cortex", "hippocampus"],
            "episodic_memory": ["hippocampus", "prefrontal_cortex"],
            "procedural_memory": ["cerebellum", "basal_ganglia"],
            "working_memory": ["prefrontal_cortex", "parietal_cortex"]
        }
    
    def map_knowledge_to_regions(self, knowledge_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Map knowledge data to appropriate brain regions"""
        mapped_knowledge = {region: [] for region in self.brain_regions.keys()}
        
        # Extract knowledge type from data
        knowledge_type = self._extract_knowledge_type(knowledge_data)
        
        # Get target regions for this knowledge type
        target_regions = self.knowledge_mappings.get(knowledge_type, ["temporal_cortex"])
        
        # Distribute knowledge across target regions
        for region in target_regions:
            if region in self.brain_regions:
                # Check capacity
                if self.brain_regions[region]["current_usage"] < self.brain_regions[region]["capacity"]:
                    knowledge_entry = {
                        "id": f"{knowledge_type}_{len(mapped_knowledge[region])}",
                        "type": knowledge_type,
                        "data": knowledge_data,
                        "timestamp": datetime.now().isoformat(),
                        "confidence": self._calculate_confidence(knowledge_data, region),
                        "plasticity": self.brain_regions[region]["plasticity_rate"]
                    }
                    mapped_knowledge[region].append(knowledge_entry)
                    
                    # Update usage
                    self.brain_regions[region]["current_usage"] += 1
        
        return mapped_knowledge
    
    def _extract_knowledge_type(self, knowledge_data: Dict[str, Any]) -> str:
        """Extract knowledge type from data"""
        if "domain" in knowledge_data:
            domain = knowledge_data["domain"]
            if domain == "neuroscience":
                if "neural_activity" in str(knowledge_data):
                    return "neural_activity"
                elif "connectivity" in str(knowledge_data):
                    return "brain_connectivity"
                else:
                    return "cognitive_functions"
            elif domain == "biochemistry":
                if "protein" in str(knowledge_data):
                    return "protein_functions"
                elif "metabolic" in str(knowledge_data):
                    return "metabolic_pathways"
                else:
                    return "molecular_structures"
            elif domain == "physics":
                return "mathematical_concepts"
            elif domain == "computer_science":
                return "algorithmic_thinking"
            elif domain == "psychology":
                return "behavioral_patterns"
        
        # Default to semantic knowledge
        return "semantic_knowledge"
    
    def _calculate_confidence(self, knowledge_data: Dict[str, Any], region: str) -> float:
        """Calculate confidence score for knowledge placement in region"""
        base_confidence = 0.7
        
        # Adjust based on region specialization
        region_capabilities = self.brain_regions[region]["learning_capabilities"]
        knowledge_type = self._extract_knowledge_type(knowledge_data)
        
        if knowledge_type in region_capabilities:
            base_confidence += 0.2
        
        # Adjust based on data quality
        if "confidence" in knowledge_data:
            base_confidence = (base_confidence + knowledge_data["confidence"]) / 2
        
        return min(base_confidence, 1.0)
    
    def get_region_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all brain regions"""
        status = {}
        for region_name, region_data in self.brain_regions.items():
            status[region_name] = {
                "name": region_data["name"],
                "usage_percentage": (region_data["current_usage"] / region_data["capacity"]) * 100,
                "available_capacity": region_data["capacity"] - region_data["current_usage"],
                "plasticity_rate": region_data["plasticity_rate"],
                "function": region_data["function"]
            }
        return status
    
    def consolidate_knowledge(self, region_name: str) -> Dict[str, Any]:
        """Consolidate knowledge in a specific brain region"""
        if region_name not in self.brain_regions:
            raise ValueError(f"Unknown brain region: {region_name}")
        
        # This would implement memory consolidation processes
        # For now, return region status
        return {
            "region": region_name,
            "consolidation_timestamp": datetime.now().isoformat(),
            "knowledge_count": self.brain_regions[region_name]["current_usage"],
            "consolidation_status": "completed"
        }
    
    def save_brain_state(self, filename: str = None):
        """Save current brain state to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"brain_state_{timestamp}.json"
        
        brain_state = {
            "timestamp": datetime.now().isoformat(),
            "regions": self.brain_regions,
            "knowledge_mappings": self.knowledge_mappings
        }
        
        file_path = os.path.join(self.database_path, "brain_regions", filename)
        with open(file_path, 'w') as f:
            json.dump(brain_state, f, indent=2)
        
        self.logger.info(f"Brain state saved to {file_path}")

def main():
    """Test the brain region mapper"""
    mapper = BrainRegionMapper()
    
    print("🧠 Brain Region Mapper - Knowledge Distribution System")
    print("=" * 60)
    
    # Test knowledge mapping
    test_knowledge = {
        "domain": "neuroscience",
        "type": "neural_activity",
        "data": {"neurons": 100, "firing_rate": 5.0},
        "confidence": 0.8
    }
    
    mapped = mapper.map_knowledge_to_regions(test_knowledge)
    
    print("📊 Knowledge Mapping Results:")
    for region, knowledge_list in mapped.items():
        if knowledge_list:
            print(f"  {region}: {len(knowledge_list)} knowledge items")
    
    # Show brain region status
    print("\n🏥 Brain Region Status:")
    status = mapper.get_region_status()
    for region, info in status.items():
        print(f"  {info['name']}: {info['usage_percentage']:.1f}% used")
    
    # Save brain state
    mapper.save_brain_state()
    print("\n✅ Brain state saved successfully!")

if __name__ == "__main__":
    main()
