#!/usr/bin/env python3
"""
🧠 Enhanced Progressive Capacity Expansion
Implements stage-specific cognitive capacity growth and expansion

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Progressive expansion of cognitive capabilities across developmental stages
**Validation Level:** Developmental milestone verification
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
import math

# Import DevelopmentalStage from developmental_timeline to avoid conflicts
from ................................................developmental_timeline import DevelopmentalStage

@dataclass
class CapacityMetric:
    """Capacity metric with developmental constraints"""
    name: str
    current_value: float
    max_value: float
    growth_rate: float
    stage_constraints: Dict[str, float]
    validation_criteria: List[str]

@dataclass
class CognitiveCapacity:
    """Cognitive capacity with developmental progression"""
    working_memory_slots: int
    attention_span: float
    processing_speed: float
    learning_rate: float
    executive_control: float
    pattern_recognition: float
    abstraction_level: float

@dataclass
class DevelopmentalMilestone:
    """Developmental milestone with capacity requirements"""
    name: str
    stage: DevelopmentalStage
    required_capacities: Dict[str, float]
    achievement_criteria: List[str]
    biological_basis: str

class CapacityProgression:
    """Enhanced capacity progression with developmental constraints"""
    
    def __init__(self, initial_stage: DevelopmentalStage = DevelopmentalStage.FETAL):
        self.current_stage = initial_stage
        self.stage_progression = {
            "F": {
                "wm_slots": 3,
                "moe_k": 1,
                "cerebellar_active": False,
                "attention_span": 0.3,
                "processing_speed": 0.2,
                "learning_rate": 0.1,
                "executive_control": 0.1,
                "pattern_recognition": 0.2,
                "abstraction_level": 0.1
            },
            "N0": {
                "wm_slots": 3,
                "moe_k": 2,
                "cerebellar_active": False,
                "attention_span": 0.5,
                "processing_speed": 0.4,
                "learning_rate": 0.3,
                "executive_control": 0.2,
                "pattern_recognition": 0.4,
                "abstraction_level": 0.2
            },
            "N1": {
                "wm_slots": 4,
                "moe_k": 2,
                "cerebellar_active": True,
                "attention_span": 0.7,
                "processing_speed": 0.6,
                "learning_rate": 0.5,
                "executive_control": 0.4,
                "pattern_recognition": 0.6,
                "abstraction_level": 0.3
            }
        }
        
        # Capacity metrics
        self.capacity_metrics = {
            "working_memory": CapacityMetric(
                name="Working Memory Capacity",
                current_value=3.0,
                max_value=7.0,  # Adult capacity
                growth_rate=0.1,
                stage_constraints={"F": 3, "N0": 3, "N1": 4},
                validation_criteria=["wm_slots", "memory_precision", "rehearsal_ability"]
            ),
            "attention": CapacityMetric(
                name="Attention Span",
                current_value=0.3,
                max_value=1.0,
                growth_rate=0.05,
                stage_constraints={"F": 0.3, "N0": 0.5, "N1": 0.7},
                validation_criteria=["sustained_attention", "selective_attention", "divided_attention"]
            ),
            "processing_speed": CapacityMetric(
                name="Processing Speed",
                current_value=0.2,
                max_value=1.0,
                growth_rate=0.08,
                stage_constraints={"F": 0.2, "N0": 0.4, "N1": 0.6},
                validation_criteria=["reaction_time", "information_processing", "decision_speed"]
            ),
            "learning_rate": CapacityMetric(
                name="Learning Rate",
                current_value=0.1,
                max_value=1.0,
                growth_rate=0.06,
                stage_constraints={"F": 0.1, "N0": 0.3, "N1": 0.5},
                validation_criteria=["acquisition_speed", "retention_rate", "generalization_ability"]
            ),
            "executive_control": CapacityMetric(
                name="Executive Control",
                current_value=0.1,
                max_value=1.0,
                growth_rate=0.04,
                stage_constraints={"F": 0.1, "N0": 0.2, "N1": 0.4},
                validation_criteria=["inhibition", "planning", "cognitive_flexibility"]
            ),
            "pattern_recognition": CapacityMetric(
                name="Pattern Recognition",
                current_value=0.2,
                max_value=1.0,
                growth_rate=0.07,
                stage_constraints={"F": 0.2, "N0": 0.4, "N1": 0.6},
                validation_criteria=["visual_patterns", "auditory_patterns", "temporal_patterns"]
            ),
            "abstraction": CapacityMetric(
                name="Abstraction Level",
                current_value=0.1,
                max_value=1.0,
                growth_rate=0.03,
                stage_constraints={"F": 0.1, "N0": 0.2, "N1": 0.3},
                validation_criteria=["concept_formation", "rule_learning", "symbolic_processing"]
            )
        }
        
        # Developmental milestones
        self.milestones = {
            "basic_neural_dynamics": DevelopmentalMilestone(
                name="Basic Neural Dynamics",
                stage=DevelopmentalStage.FETAL,
                required_capacities={
                    "working_memory": 0.3,
                    "attention": 0.3,
                    "processing_speed": 0.2
                },
                achievement_criteria=[
                    "Neural firing rates within biological range",
                    "Basic synchrony patterns established",
                    "Plasticity mechanisms functional"
                ],
                biological_basis="Spontaneous neural activity begins at 8-10 weeks gestation"
            ),
            "minimal_cognitive_scaffold": DevelopmentalMilestone(
                name="Minimal Cognitive Scaffold",
                stage=DevelopmentalStage.FETAL,
                required_capacities={
                    "working_memory": 0.5,
                    "attention": 0.4,
                    "executive_control": 0.1
                },
                achievement_criteria=[
                    "All core brain modules functional",
                    "Basic inter-module communication established",
                    "Minimal working memory operational"
                ],
                biological_basis="Primary brain regions formed and connected"
            ),
            "sleep_consolidation": DevelopmentalMilestone(
                name="Sleep-Consolidation Cycles",
                stage=DevelopmentalStage.NEONATE,
                required_capacities={
                    "learning_rate": 0.3,
                    "attention": 0.5,
                    "processing_speed": 0.4
                },
                achievement_criteria=[
                    "Sleep-wake cycles with biological timing",
                    "Memory consolidation during sleep",
                    "Fatigue management system functional"
                ],
                biological_basis="REM/NREM sleep cycles begin at 30-32 weeks gestation"
            ),
            "attention_switching": DevelopmentalMilestone(
                name="Attention Network Switching",
                stage=DevelopmentalStage.NEONATE,
                required_capacities={
                    "attention": 0.6,
                    "executive_control": 0.2,
                    "pattern_recognition": 0.4
                },
                achievement_criteria=[
                    "Successful switching between internal and external modes",
                    "Appropriate response to salient stimuli",
                    "Stable attention allocation patterns"
                ],
                biological_basis="Salience network development in late gestation"
            ),
            "working_memory_expansion": DevelopmentalMilestone(
                name="Working Memory Expansion",
                stage=DevelopmentalStage.EARLY_POSTNATAL,
                required_capacities={
                    "working_memory": 0.7,
                    "executive_control": 0.4,
                    "processing_speed": 0.6
                },
                achievement_criteria=[
                    "Working memory capacity increased to 4 slots",
                    "Maintained precision and recall accuracy",
                    "Stable performance under load"
                ],
                biological_basis="Prefrontal cortex maturation in early postnatal period"
            ),
            "cerebellar_integration": DevelopmentalMilestone(
                name="Cerebellar Integration",
                stage=DevelopmentalStage.EARLY_POSTNATAL,
                required_capacities={
                    "processing_speed": 0.7,
                    "pattern_recognition": 0.6,
                    "learning_rate": 0.5
                },
                achievement_criteria=[
                    "Improved timing precision in motor and cognitive tasks",
                    "Enhanced coordination between brain regions",
                    "Stable cerebellar modulation signals"
                ],
                biological_basis="Cerebellar growth and connectivity in early postnatal period"
            )
        }
        
        # Current cognitive capacity
        self.cognitive_capacity = CognitiveCapacity(
            working_memory_slots=3,
            attention_span=0.3,
            processing_speed=0.2,
            learning_rate=0.1,
            executive_control=0.1,
            pattern_recognition=0.2,
            abstraction_level=0.1
        )
        
        # Progression tracking
        self.achieved_milestones = []
        self.progression_history = []
        self.growth_rate_modifiers = {}
    
    def step(self, dt: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step capacity progression"""
        # Update capacity metrics
        self.update_capacity_metrics(dt, context)
        
        # Check for milestone achievements
        new_milestones = self.check_milestone_achievements(context)
        
        # Update cognitive capacity
        self.update_cognitive_capacity()
        
        # Check for stage progression
        stage_progression = self.check_stage_progression(context)
        
        # Record progression
        self.record_progression(dt, context)
        
        return {
            "current_stage": self.current_stage.value,
            "cognitive_capacity": self.get_capacity_summary(),
            "achieved_milestones": [m.name for m in self.achieved_milestones],
            "new_milestones": [m.name for m in new_milestones],
            "stage_progression": stage_progression,
            "capacity_metrics": self.get_metrics_summary()
        }
    
    def update_capacity_metrics(self, dt: float, context: Dict[str, Any]):
        """Update capacity metrics based on development and experience"""
        for metric_name, metric in self.capacity_metrics.items():
            # Get current stage constraint
            stage_constraint = metric.stage_constraints.get(self.current_stage.value, 0.0)
            
            # Calculate growth rate with modifiers
            base_growth_rate = metric.growth_rate
            modifier = self.growth_rate_modifiers.get(metric_name, 1.0)
            experience_boost = context.get(f"{metric_name}_experience", 0.0)
            
            effective_growth_rate = base_growth_rate * modifier * (1.0 + experience_boost)
            
            # Update metric value
            growth = effective_growth_rate * dt
            new_value = min(metric.max_value, metric.current_value + growth)
            
            # Apply stage constraints
            new_value = min(new_value, stage_constraint)
            
            metric.current_value = new_value
    
    def check_milestone_achievements(self, context: Dict[str, Any]) -> List[DevelopmentalMilestone]:
        """Check for new milestone achievements"""
        new_milestones = []
        
        for milestone_name, milestone in self.milestones.items():
            if milestone in self.achieved_milestones:
                continue
            
            if milestone.stage != self.current_stage:
                continue
            
            # Check if all required capacities are met
            all_capacities_met = True
            for capacity_name, required_value in milestone.required_capacities.items():
                if capacity_name in self.capacity_metrics:
                    current_value = self.capacity_metrics[capacity_name].current_value
                    if current_value < required_value:
                        all_capacities_met = False
                        break
            
            # Check achievement criteria
            criteria_met = self.check_achievement_criteria(milestone.achievement_criteria, context)
            
            if all_capacities_met and criteria_met:
                self.achieved_milestones.append(milestone)
                new_milestones.append(milestone)
                
                # Apply milestone effects
                self.apply_milestone_effects(milestone)
        
        return new_milestones
    
    def check_achievement_criteria(self, criteria: List[str], context: Dict[str, Any]) -> bool:
        """Check if achievement criteria are met"""
        for criterion in criteria:
            criterion_value = context.get(criterion, 0.0)
            if criterion_value < 0.5:  # Threshold for achievement
                return False
        return True
    
    def apply_milestone_effects(self, milestone: DevelopmentalMilestone):
        """Apply effects of achieving a milestone"""
        # Boost growth rates for related capacities
        for capacity_name in milestone.required_capacities.keys():
            if capacity_name in self.capacity_metrics:
                current_modifier = self.growth_rate_modifiers.get(capacity_name, 1.0)
                self.growth_rate_modifiers[capacity_name] = current_modifier * 1.2  # 20% boost
    
    def update_cognitive_capacity(self):
        """Update cognitive capacity based on current metrics"""
        self.cognitive_capacity.working_memory_slots = int(
            self.capacity_metrics["working_memory"].current_value
        )
        self.cognitive_capacity.attention_span = self.capacity_metrics["attention"].current_value
        self.cognitive_capacity.processing_speed = self.capacity_metrics["processing_speed"].current_value
        self.cognitive_capacity.learning_rate = self.capacity_metrics["learning_rate"].current_value
        self.cognitive_capacity.executive_control = self.capacity_metrics["executive_control"].current_value
        self.cognitive_capacity.pattern_recognition = self.capacity_metrics["pattern_recognition"].current_value
        self.cognitive_capacity.abstraction_level = self.capacity_metrics["abstraction"].current_value
    
    def check_stage_progression(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if ready to progress to next stage"""
        current_stage_info = self.stage_progression[self.current_stage.value]
        next_stage = self.get_next_stage()
        
        if next_stage is None:
            return {"ready": False, "reason": "Already at final stage"}
        
        next_stage_info = self.stage_progression[next_stage.value]
        
        # Check if all current stage milestones are achieved
        current_milestones = [m for m in self.milestones.values() if m.stage == self.current_stage]
        achieved_current = [m for m in current_milestones if m in self.achieved_milestones]
        
        if len(achieved_current) < len(current_milestones):
            return {
                "ready": False,
                "reason": f"Only {len(achieved_current)}/{len(current_milestones)} milestones achieved"
            }
        
        # Check if capacities are ready for next stage
        ready_capacities = []
        for capacity_name, target_value in next_stage_info.items():
            if capacity_name in self.capacity_metrics:
                current_value = self.capacity_metrics[capacity_name].current_value
                if current_value >= target_value * 0.8:  # 80% of target
                    ready_capacities.append(capacity_name)
        
        if len(ready_capacities) >= len(next_stage_info) * 0.7:  # 70% of capacities ready
            # Progress to next stage
            self.progress_to_stage(next_stage)
            return {
                "ready": True,
                "new_stage": next_stage.value,
                "ready_capacities": ready_capacities
            }
        
        return {
            "ready": False,
            "reason": f"Only {len(ready_capacities)}/{len(next_stage_info)} capacities ready"
        }
    
    def get_next_stage(self) -> Optional[DevelopmentalStage]:
        """Get the next developmental stage"""
        stages = list(DevelopmentalStage)
        try:
            current_index = stages.index(self.current_stage)
            
            if current_index < len(stages) - 1:
                return stages[current_index + 1]
        except ValueError:
            # If current stage not found, return None
            pass
        
        return None
    
    def progress_to_stage(self, new_stage: DevelopmentalStage):
        """Progress to a new developmental stage"""
        self.current_stage = new_stage
        
        # Update capacity constraints for new stage
        stage_info = self.stage_progression[new_stage.value]
        for capacity_name, target_value in stage_info.items():
            if capacity_name in self.capacity_metrics:
                self.capacity_metrics[capacity_name].current_value = min(
                    self.capacity_metrics[capacity_name].current_value,
                    target_value
                )
    
    def record_progression(self, dt: float, context: Dict[str, Any]):
        """Record progression history"""
        progression_record = {
            "timestamp": len(self.progression_history) * dt,
            "stage": self.current_stage.value,
            "capacities": self.get_capacity_summary(),
            "milestones": [m.name for m in self.achieved_milestones],
            "context": context.copy()
        }
        self.progression_history.append(progression_record)
    
    def get_capacity_summary(self) -> Dict[str, Any]:
        """Get summary of current cognitive capacity"""
        return {
            "working_memory_slots": self.cognitive_capacity.working_memory_slots,
            "attention_span": self.cognitive_capacity.attention_span,
            "processing_speed": self.cognitive_capacity.processing_speed,
            "learning_rate": self.cognitive_capacity.learning_rate,
            "executive_control": self.cognitive_capacity.executive_control,
            "pattern_recognition": self.cognitive_capacity.pattern_recognition,
            "abstraction_level": self.cognitive_capacity.abstraction_level
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of capacity metrics"""
        summary = {}
        for metric_name, metric in self.capacity_metrics.items():
            summary[metric_name] = {
                "current_value": metric.current_value,
                "max_value": metric.max_value,
                "growth_rate": metric.growth_rate,
                "stage_constraint": metric.stage_constraints.get(self.current_stage.value, 0.0)
            }
        return summary
    
    def get_developmental_summary(self) -> Dict[str, Any]:
        """Get comprehensive developmental summary"""
        return {
            "current_stage": self.current_stage.value,
            "cognitive_capacity": self.get_capacity_summary(),
            "achieved_milestones": [m.name for m in self.achieved_milestones],
            "available_milestones": [
                m.name for m in self.milestones.values() 
                if m.stage == self.current_stage and m not in self.achieved_milestones
            ],
            "capacity_metrics": self.get_metrics_summary(),
            "progression_history_length": len(self.progression_history)
        }
    
    def add_experience(self, experience_type: str, intensity: float):
        """Add experience to boost capacity growth"""
        if experience_type in self.capacity_metrics:
            current_boost = self.growth_rate_modifiers.get(experience_type, 1.0)
            self.growth_rate_modifiers[experience_type] = current_boost + intensity * 0.1
    
    def reset_progression(self, stage: DevelopmentalStage = DevelopmentalStage.FETAL):
        """Reset progression to a specific stage"""
        self.current_stage = stage
        self.achieved_milestones = []
        self.progression_history = []
        self.growth_rate_modifiers = {}
        
        # Reset capacity metrics to stage-appropriate values
        stage_info = self.stage_progression[stage]
        for capacity_name, target_value in stage_info.items():
            if capacity_name in self.capacity_metrics:
                self.capacity_metrics[capacity_name].current_value = target_value * 0.5  # Start at 50%
        
        self.update_cognitive_capacity()

# Example usage and testing
if __name__ == "__main__":
    # Create capacity progression system
    progression = CapacityProgression(DevelopmentalStage.FETAL)
    
    print("=== Capacity Progression Simulation ===")
    
    # Simulate development
    for i in range(100):  # Simulate 100 time steps
        context = {
            "neural_firing_rates": 0.8 if i > 20 else 0.3,
            "synchrony_patterns": 0.7 if i > 30 else 0.2,
            "plasticity_mechanisms": 0.9 if i > 40 else 0.4,
            "working_memory_experience": 0.1,
            "attention_experience": 0.1,
            "processing_speed_experience": 0.1
        }
        
        result = progression.step(dt=1.0, context=context)
        
        if i % 20 == 0:  # Print every 20 steps
            print(f"Step {i}: Stage={result['current_stage']}, "
                  f"WM Slots={result['cognitive_capacity']['working_memory_slots']}, "
                  f"Attention={result['cognitive_capacity']['attention_span']:.2f}, "
                  f"Milestones={len(result['achieved_milestones'])}")
            
            if result['new_milestones']:
                print(f"  New milestones: {result['new_milestones']}")
            
            if result['stage_progression']['ready']:
                print(f"  Stage progression: {result['stage_progression']['new_stage']}")
    
    # Print final summary
    summary = progression.get_developmental_summary()
    print(f"\n=== Final Developmental Summary ===")
    print(f"Current stage: {summary['current_stage']}")
    print(f"Achieved milestones: {summary['achieved_milestones']}")
    print(f"Available milestones: {summary['available_milestones']}")
    print(f"Working memory slots: {summary['cognitive_capacity']['working_memory_slots']}")
    print(f"Attention span: {summary['cognitive_capacity']['attention_span']:.2f}")
    print(f"Processing speed: {summary['cognitive_capacity']['processing_speed']:.2f}")
