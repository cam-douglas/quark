#!/usr/bin/env python3
"""
🧠 Enhanced Sleep-Consolidation Framework
Implements sophisticated sleep-driven consolidation mechanisms

**Model:** Claude (Functional Implementation & Testing)
**Purpose:** Sleep cycles, memory consolidation, and fatigue management
**Validation Level:** Biological sleep pattern verification
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, field
from enum import Enum
import random
import math

class SleepPhase(Enum):
    """Sleep phases with biological characteristics"""
    WAKE = "wake"
    NREM_1 = "nrem_1"      # Light sleep
    NREM_2 = "nrem_2"      # Light sleep with sleep spindles
    NREM_3 = "nrem_3"      # Deep sleep (slow wave sleep)
    REM = "rem"            # Rapid eye movement sleep

@dataclass
class SleepCycle:
    """Complete sleep cycle with phase transitions"""
    phase: SleepPhase
    duration: float  # minutes
    start_time: float
    end_time: float
    consolidation_strength: float = 0.0
    replay_activity: float = 0.0

@dataclass
class MemoryTrace:
    """Memory trace for consolidation"""
    trace_id: str
    content: str
    strength: float
    age: float  # hours since encoding
    consolidation_priority: float
    replay_count: int = 0
    last_replay: float = 0.0

@dataclass
class FatigueState:
    """Fatigue and recovery state"""
    current_fatigue: float  # 0.0 = fully rested, 1.0 = exhausted
    recovery_rate: float
    sleep_debt: float
    circadian_phase: float  # 0-24 hour cycle
    sleep_pressure: float

class SleepConsolidationEngine:
    """Enhanced sleep-consolidation engine with biological accuracy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Sleep cycle parameters
        self.sleep_cycle_duration = self.config.get("sleep_cycle_duration", 90.0)  # minutes
        self.wake_duration = self.config.get("wake_duration", 960.0)  # 16 hours
        self.sleep_duration = self.config.get("sleep_duration", 480.0)  # 8 hours
        
        # Phase-specific parameters
        self.phase_durations = {
            SleepPhase.WAKE: self.wake_duration,
            SleepPhase.NREM_1: 5.0,    # minutes
            SleepPhase.NREM_2: 25.0,   # minutes
            SleepPhase.NREM_3: 20.0,   # minutes
            SleepPhase.REM: 20.0       # minutes
        }
        
        # Consolidation parameters
        self.consolidation_strengths = {
            SleepPhase.NREM_1: 0.1,
            SleepPhase.NREM_2: 0.3,
            SleepPhase.NREM_3: 0.8,  # Deep sleep is best for consolidation
            SleepPhase.REM: 0.6,     # REM important for emotional memory
            SleepPhase.WAKE: 0.0
        }
        
        # Replay parameters
        self.replay_probabilities = {
            SleepPhase.NREM_1: 0.1,
            SleepPhase.NREM_2: 0.3,
            SleepPhase.NREM_3: 0.8,  # High replay in deep sleep
            SleepPhase.REM: 0.5,     # Moderate replay in REM
            SleepPhase.WAKE: 0.0
        }
        
        # Current state
        self.current_phase = SleepPhase.WAKE
        self.phase_start_time = 0.0
        self.total_time = 0.0
        self.cycle_count = 0
        
        # Memory traces for consolidation
        self.memory_traces: List[MemoryTrace] = []
        self.consolidated_memories: List[MemoryTrace] = []
        
        # Fatigue management
        self.fatigue_state = FatigueState(
            current_fatigue=0.0,
            recovery_rate=0.1,  # per hour
            sleep_debt=0.0,
            circadian_phase=8.0,  # Start at 8 AM
            sleep_pressure=0.0
        )
        
        # Sleep quality metrics
        self.sleep_metrics = {
            "total_sleep_time": 0.0,
            "deep_sleep_time": 0.0,
            "rem_sleep_time": 0.0,
            "sleep_efficiency": 0.0,
            "consolidation_quality": 0.0,
            "replay_efficiency": 0.0
        }
    
    def step(self, dt: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step the sleep-consolidation engine"""
        # Update time
        self.total_time += dt
        self.fatigue_state.circadian_phase = (self.fatigue_state.circadian_phase + dt/60.0) % 24.0
        
        # Update fatigue
        self.update_fatigue(dt, context)
        
        # Determine sleep/wake state
        if self.should_sleep(context):
            self.current_phase = self.determine_sleep_phase()
        else:
            self.current_phase = SleepPhase.WAKE
        
        # Update phase timing
        phase_duration = self.phase_durations[self.current_phase]
        time_in_phase = self.total_time - self.phase_start_time
        
        # Check for phase transition
        if time_in_phase >= phase_duration:
            self.transition_phase()
        
        # Perform consolidation and replay
        consolidation_result = self.perform_consolidation(dt, context)
        replay_result = self.perform_replay(dt, context)
        
        # Update sleep metrics
        self.update_sleep_metrics(dt)
        
        # Generate output
        output = {
            "current_phase": self.current_phase.value,
            "time_in_phase": time_in_phase,
            "fatigue": self.fatigue_state.current_fatigue,
            "sleep_pressure": self.fatigue_state.sleep_pressure,
            "circadian_phase": self.fatigue_state.circadian_phase,
            "consolidation": consolidation_result,
            "replay": replay_result,
            "sleep_metrics": self.sleep_metrics.copy(),
            "memory_trace_count": len(self.memory_traces),
            "consolidated_count": len(self.consolidated_memories)
        }
        
        return output
    
    def update_fatigue(self, dt: float, context: Dict[str, Any]):
        """Update fatigue and sleep pressure"""
        # Fatigue increases during wake, decreases during sleep
        if self.current_phase == SleepPhase.WAKE:
            # Fatigue increases with time awake and cognitive load
            cognitive_load = context.get("cognitive_load", 0.5)
            fatigue_increase = (0.1 + 0.2 * cognitive_load) * dt / 60.0  # per hour
            self.fatigue_state.current_fatigue = min(1.0, self.fatigue_state.current_fatigue + fatigue_increase)
            self.fatigue_state.sleep_debt += fatigue_increase
        else:
            # Fatigue decreases during sleep, faster in deep sleep
            recovery_rate = self.consolidation_strengths[self.current_phase] * self.fatigue_state.recovery_rate
            fatigue_decrease = recovery_rate * dt / 60.0
            self.fatigue_state.current_fatigue = max(0.0, self.fatigue_state.current_fatigue - fatigue_decrease)
            self.fatigue_state.sleep_debt = max(0.0, self.fatigue_state.sleep_debt - fatigue_decrease)
        
        # Sleep pressure based on circadian rhythm and sleep debt
        circadian_component = self.calculate_circadian_sleep_pressure()
        debt_component = self.fatigue_state.sleep_debt
        self.fatigue_state.sleep_pressure = min(1.0, circadian_component + debt_component)
    
    def calculate_circadian_sleep_pressure(self) -> float:
        """Calculate circadian component of sleep pressure"""
        # Peak sleep pressure around 2-4 AM, lowest around 2-4 PM
        hour = self.fatigue_state.circadian_phase
        if 2 <= hour <= 4:
            return 0.8  # Peak sleep pressure
        elif 14 <= hour <= 16:
            return 0.1  # Lowest sleep pressure
        else:
            # Smooth transition
            if hour < 14:
                return 0.1 + 0.7 * (hour - 4) / 10.0
            else:
                return 0.1 + 0.7 * (hour - 16) / 10.0
    
    def should_sleep(self, context: Dict[str, Any]) -> bool:
        """Determine if should be sleeping based on fatigue and context"""
        # High sleep pressure triggers sleep
        if self.fatigue_state.sleep_pressure > 0.8:
            return True
        
        # High fatigue triggers sleep
        if self.fatigue_state.current_fatigue > 0.9:
            return True
        
        # External sleep signal
        if context.get("sleep_signal", False):
            return True
        
        # Circadian rhythm (night time)
        hour = self.fatigue_state.circadian_phase
        if 22 <= hour or hour <= 6:
            return True
        
        return False
    
    def determine_sleep_phase(self) -> SleepPhase:
        """Determine current sleep phase based on sleep cycle timing"""
        if self.current_phase == SleepPhase.WAKE:
            # Transition to light sleep
            return SleepPhase.NREM_1
        
        # Calculate position in sleep cycle
        cycle_time = (self.total_time - self.phase_start_time) % self.sleep_cycle_duration
        
        # Phase progression within cycle
        if cycle_time < self.phase_durations[SleepPhase.NREM_1]:
            return SleepPhase.NREM_1
        elif cycle_time < self.phase_durations[SleepPhase.NREM_1] + self.phase_durations[SleepPhase.NREM_2]:
            return SleepPhase.NREM_2
        elif cycle_time < self.phase_durations[SleepPhase.NREM_1] + self.phase_durations[SleepPhase.NREM_2] + self.phase_durations[SleepPhase.NREM_3]:
            return SleepPhase.NREM_3
        else:
            return SleepPhase.REM
    
    def transition_phase(self):
        """Handle phase transition"""
        self.phase_start_time = self.total_time
        self.cycle_count += 1
    
    def add_memory_trace(self, trace_id: str, content: str, strength: float = 1.0):
        """Add a new memory trace for consolidation"""
        memory_trace = MemoryTrace(
            trace_id=trace_id,
            content=content,
            strength=strength,
            age=0.0,
            consolidation_priority=strength
        )
        self.memory_traces.append(memory_trace)
    
    def perform_consolidation(self, dt: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory consolidation during sleep"""
        if self.current_phase == SleepPhase.WAKE:
            return {"consolidated_count": 0, "consolidation_strength": 0.0}
        
        consolidation_strength = self.consolidation_strengths[self.current_phase]
        consolidated_count = 0
        consolidation_quality = 0.0
        
        # Age memory traces
        for trace in self.memory_traces:
            trace.age += dt / 3600.0  # Convert to hours
        
        # Sort by consolidation priority (strength * age)
        self.memory_traces.sort(key=lambda t: t.consolidation_priority * t.age, reverse=True)
        
        # Consolidate top traces
        traces_to_consolidate = min(len(self.memory_traces), 5)  # Limit consolidation per cycle
        
        for i in range(traces_to_consolidate):
            if i < len(self.memory_traces):
                trace = self.memory_traces[i]
                
                # Consolidation probability based on phase strength and trace priority
                consolidation_prob = consolidation_strength * trace.consolidation_priority
                
                if random.random() < consolidation_prob:
                    # Consolidate the trace
                    trace.strength *= 1.5  # Strengthen memory
                    self.consolidated_memories.append(trace)
                    self.memory_traces.pop(i)
                    consolidated_count += 1
                    consolidation_quality += trace.strength
        
        return {
            "consolidated_count": consolidated_count,
            "consolidation_strength": consolidation_strength,
            "consolidation_quality": consolidation_quality,
            "remaining_traces": len(self.memory_traces)
        }
    
    def perform_replay(self, dt: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform memory replay during sleep"""
        if self.current_phase == SleepPhase.WAKE:
            return {"replay_count": 0, "replay_strength": 0.0}
        
        replay_prob = self.replay_probabilities[self.current_phase]
        replay_count = 0
        replay_strength = 0.0
        
        # Replay memory traces
        for trace in self.memory_traces:
            if random.random() < replay_prob:
                trace.replay_count += 1
                trace.last_replay = self.total_time
                trace.strength *= 1.1  # Slight strengthening from replay
                replay_count += 1
                replay_strength += trace.strength
        
        # Replay consolidated memories for reinforcement
        for trace in self.consolidated_memories[-10:]:  # Recent consolidated memories
            if random.random() < replay_prob * 0.5:  # Lower probability for consolidated
                trace.replay_count += 1
                trace.last_replay = self.total_time
                replay_count += 1
                replay_strength += trace.strength
        
        return {
            "replay_count": replay_count,
            "replay_strength": replay_strength,
            "replay_probability": replay_prob
        }
    
    def update_sleep_metrics(self, dt: float):
        """Update sleep quality metrics"""
        if self.current_phase != SleepPhase.WAKE:
            self.sleep_metrics["total_sleep_time"] += dt / 60.0  # Convert to minutes
            
            if self.current_phase == SleepPhase.NREM_3:
                self.sleep_metrics["deep_sleep_time"] += dt / 60.0
            elif self.current_phase == SleepPhase.REM:
                self.sleep_metrics["rem_sleep_time"] += dt / 60.0
        
        # Calculate sleep efficiency (time asleep / time in bed)
        if self.sleep_metrics["total_sleep_time"] > 0:
            self.sleep_metrics["sleep_efficiency"] = (
                self.sleep_metrics["total_sleep_time"] / 
                (self.sleep_metrics["total_sleep_time"] + self.sleep_metrics.get("wake_time", 0))
            )
        
        # Calculate consolidation quality
        if len(self.consolidated_memories) > 0:
            avg_strength = sum(t.strength for t in self.consolidated_memories) / len(self.consolidated_memories)
            self.sleep_metrics["consolidation_quality"] = avg_strength
        
        # Calculate replay efficiency
        total_replays = sum(t.replay_count for t in self.memory_traces + self.consolidated_memories)
        if total_replays > 0:
            self.sleep_metrics["replay_efficiency"] = total_replays / (len(self.memory_traces) + len(self.consolidated_memories))
    
    def get_sleep_summary(self) -> Dict[str, Any]:
        """Get comprehensive sleep summary"""
        return {
            "current_phase": self.current_phase.value,
            "total_time": self.total_time,
            "cycle_count": self.cycle_count,
            "fatigue_state": {
                "current_fatigue": self.fatigue_state.current_fatigue,
                "sleep_debt": self.fatigue_state.sleep_debt,
                "sleep_pressure": self.fatigue_state.sleep_pressure,
                "circadian_phase": self.fatigue_state.circadian_phase
            },
            "sleep_metrics": self.sleep_metrics.copy(),
            "memory_stats": {
                "active_traces": len(self.memory_traces),
                "consolidated_traces": len(self.consolidated_memories),
                "total_replays": sum(t.replay_count for t in self.memory_traces + self.consolidated_memories)
            }
        }
    
    def reset_sleep_cycle(self):
        """Reset sleep cycle (e.g., for new day)"""
        self.current_phase = SleepPhase.WAKE
        self.phase_start_time = self.total_time
        self.cycle_count = 0
        self.fatigue_state.sleep_debt = 0.0
        self.sleep_metrics = {
            "total_sleep_time": 0.0,
            "deep_sleep_time": 0.0,
            "rem_sleep_time": 0.0,
            "sleep_efficiency": 0.0,
            "consolidation_quality": 0.0,
            "replay_efficiency": 0.0
        }

class NREMConsolidation:
    """NREM sleep consolidation mechanisms"""
    
    def __init__(self):
        self.slow_wave_activity = 0.0
        self.sleep_spindles = 0.0
        self.delta_power = 0.0
    
    def step(self, dt: float, phase: SleepPhase) -> Dict[str, Any]:
        """Step NREM consolidation"""
        if phase in [SleepPhase.NREM_1, SleepPhase.NREM_2, SleepPhase.NREM_3]:
            # Generate slow wave activity
            self.slow_wave_activity = np.random.random()
            self.delta_power = np.random.random() if phase == SleepPhase.NREM_3 else 0.0
            self.sleep_spindles = np.random.random() if phase == SleepPhase.NREM_2 else 0.0
            
            return {
                "slow_wave_activity": self.slow_wave_activity,
                "delta_power": self.delta_power,
                "sleep_spindles": self.sleep_spindles,
                "consolidation_strength": self.calculate_consolidation_strength(phase)
            }
        
        return {"consolidation_strength": 0.0}
    
    def calculate_consolidation_strength(self, phase: SleepPhase) -> float:
        """Calculate consolidation strength based on NREM characteristics"""
        if phase == SleepPhase.NREM_3:
            return self.slow_wave_activity * 0.8 + self.delta_power * 0.2
        elif phase == SleepPhase.NREM_2:
            return self.sleep_spindles * 0.6 + self.slow_wave_activity * 0.4
        else:
            return self.slow_wave_activity * 0.3

class REMReplay:
    """REM sleep replay mechanisms"""
    
    def __init__(self):
        self.rem_density = 0.0
        self.theta_activity = 0.0
        self.paradoxical_sleep = 0.0
    
    def step(self, dt: float, phase: SleepPhase) -> Dict[str, Any]:
        """Step REM replay"""
        if phase == SleepPhase.REM:
            # Generate REM characteristics
            self.rem_density = np.random.random()
            self.theta_activity = np.random.random()
            self.paradoxical_sleep = np.random.random()
            
            return {
                "rem_density": self.rem_density,
                "theta_activity": self.theta_activity,
                "paradoxical_sleep": self.paradoxical_sleep,
                "replay_strength": self.calculate_replay_strength()
            }
        
        return {"replay_strength": 0.0}
    
    def calculate_replay_strength(self) -> float:
        """Calculate replay strength based on REM characteristics"""
        return self.rem_density * 0.4 + self.theta_activity * 0.4 + self.paradoxical_sleep * 0.2

class FatigueSystem:
    """Fatigue management system"""
    
    def __init__(self):
        self.energy_level = 1.0
        self.mental_fatigue = 0.0
        self.physical_fatigue = 0.0
        self.recovery_rate = 0.1
    
    def step(self, dt: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step fatigue system"""
        cognitive_load = context.get("cognitive_load", 0.5)
        physical_activity = context.get("physical_activity", 0.0)
        
        # Update fatigue based on activity
        self.mental_fatigue += cognitive_load * dt / 3600.0  # per hour
        self.physical_fatigue += physical_activity * dt / 3600.0
        
        # Recovery during rest
        if context.get("is_resting", False):
            self.mental_fatigue = max(0.0, self.mental_fatigue - self.recovery_rate * dt / 3600.0)
            self.physical_fatigue = max(0.0, self.physical_fatigue - self.recovery_rate * dt / 3600.0)
        
        # Calculate overall fatigue
        total_fatigue = (self.mental_fatigue + self.physical_fatigue) / 2.0
        self.energy_level = max(0.0, 1.0 - total_fatigue)
        
        return {
            "energy_level": self.energy_level,
            "mental_fatigue": self.mental_fatigue,
            "physical_fatigue": self.physical_fatigue,
            "total_fatigue": total_fatigue
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sleep consolidation engine
    config = {
        "sleep_cycle_duration": 90.0,
        "wake_duration": 960.0,
        "sleep_duration": 480.0
    }
    
    sleep_engine = SleepConsolidationEngine(config)
    
    # Add some memory traces
    sleep_engine.add_memory_trace("trace_1", "Learning to walk", 0.8)
    sleep_engine.add_memory_trace("trace_2", "First words", 0.9)
    sleep_engine.add_memory_trace("trace_3", "Object permanence", 0.7)
    
    # Simulate sleep cycle
    print("=== Sleep Consolidation Simulation ===")
    
    for i in range(100):  # Simulate 100 time steps
        context = {
            "cognitive_load": 0.3 if i < 50 else 0.0,  # High load then rest
            "sleep_signal": i >= 50,  # Sleep signal after 50 steps
            "is_resting": i >= 50
        }
        
        result = sleep_engine.step(dt=1.0, context=context)
        
        if i % 10 == 0:  # Print every 10 steps
            print(f"Step {i}: Phase={result['current_phase']}, "
                  f"Fatigue={result['fatigue']:.2f}, "
                  f"Consolidated={result['consolidation']['consolidated_count']}, "
                  f"Replay={result['replay']['replay_count']}")
    
    # Print final summary
    summary = sleep_engine.get_sleep_summary()
    print(f"\n=== Final Summary ===")
    print(f"Total time: {summary['total_time']:.1f} minutes")
    print(f"Sleep cycles: {summary['cycle_count']}")
    print(f"Deep sleep: {summary['sleep_metrics']['deep_sleep_time']:.1f} minutes")
    print(f"REM sleep: {summary['sleep_metrics']['rem_sleep_time']:.1f} minutes")
    print(f"Consolidated memories: {summary['memory_stats']['consolidated_traces']}")
    print(f"Total replays: {summary['memory_stats']['total_replays']}")
