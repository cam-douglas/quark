#!/usr/bin/env python3
"""
Adaptive Learning System for Command Evolution

This module implements sophisticated learning algorithms that help commands
evolve, improve, and expand exponentially based on usage patterns and context.
"""

import os
import json
import time
import math
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, asdict
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import hashlib

from .....................................................command_database import CommandDatabase, Command

@dataclass
class CommandDNA:
    """Represents the genetic code of a command that can evolve."""
    command_id: str
    traits: Dict[str, float]  # Numerical traits like complexity, usage_frequency, etc.
    genes: Dict[str, Any]     # Command attributes that can mutate
    fitness_score: float      # How well this command performs
    generation: int           # Which generation this command belongs to
    parent_ids: List[str]     # Parent commands that created this one
    mutations: List[str]      # History of mutations
    creation_time: str
    
@dataclass
class LearningPattern:
    """Represents a learned pattern about command usage."""
    pattern_id: str
    pattern_type: str  # sequence, correlation, temporal, contextual
    confidence: float
    frequency: int
    discovered_at: str
    elements: List[str]  # Commands or features involved
    context: Dict[str, Any]
    
class AdaptiveLearningEngine:
    """Advanced learning engine that evolves commands through usage patterns."""
    
    def __init__(self, command_database: CommandDatabase):
        self.db = command_database
        self.logger = logging.getLogger("adaptive_learning")
        
        # Learning state
        self.command_dna = {}
        self.learning_patterns = {}
        self.usage_sequences = []
        self.context_memory = defaultdict(list)
        self.evolution_history = []
        
        # Learning parameters
        self.mutation_rate = 0.1
        self.learning_rate = 0.05
        self.pattern_threshold = 0.7
        self.max_generations = 100
        
        # Initialize DNA for existing commands
        self._initialize_command_dna()
    
    def _initialize_command_dna(self):
        """Initialize genetic representation for existing commands."""
        commands = self.db.search_commands("")  # Get all commands
        
        for command in commands:
            dna = CommandDNA(
                command_id=command.id,
                traits={
                    "complexity": self._complexity_to_score(command.complexity),
                    "usage_frequency": 0.0,
                    "success_rate": 1.0,
                    "execution_speed": 1.0,
                    "versatility": len(command.keywords) / 10.0,
                    "safety": 1.0 if command.safe_mode else 0.5
                },
                genes={
                    "executable": command.executable,
                    "args": command.args.copy(),
                    "flags": command.flags.copy(),
                    "keywords": command.keywords.copy(),
                    "category": command.category,
                    "description": command.description
                },
                fitness_score=1.0,
                generation=0,
                parent_ids=[],
                mutations=[],
                creation_time=datetime.now().isoformat()
            )
            self.command_dna[command.id] = dna
        
        self.logger.info(f"Initialized DNA for {len(commands)} commands")
    
    def _complexity_to_score(self, complexity: str) -> float:
        """Convert complexity string to numerical score."""
        return {"low": 0.3, "medium": 0.6, "high": 0.9}.get(complexity, 0.5)
    
    def learn_from_usage(self, command_id: str, context: Dict[str, Any], 
                        success: bool, execution_time: float):
        """Learn from command usage and update fitness."""
        if command_id not in self.command_dna:
            return
        
        dna = self.command_dna[command_id]
        
        # Update traits based on usage
        old_frequency = dna.traits["usage_frequency"]
        dna.traits["usage_frequency"] = min(1.0, old_frequency + 0.1)
        
        # Update success rate
        old_success = dna.traits["success_rate"]
        dna.traits["success_rate"] = (old_success * 0.9 + (1.0 if success else 0.0) * 0.1)
        
        # Update execution speed (inverse of time)
        speed_score = max(0.1, 1.0 / (1.0 + execution_time))
        old_speed = dna.traits["execution_speed"]
        dna.traits["execution_speed"] = (old_speed * 0.8 + speed_score * 0.2)
        
        # Calculate new fitness score
        dna.fitness_score = self._calculate_fitness(dna)
        
        # Record usage sequence for pattern learning
        self.usage_sequences.append({
            "command_id": command_id,
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "success": success,
            "execution_time": execution_time
        })
        
        # Store context for pattern recognition
        self.context_memory[command_id].append(context)
        
        # Discover patterns periodically
        if len(self.usage_sequences) % 10 == 0:
            self._discover_patterns()
        
        self.logger.debug(f"Updated DNA for {command_id}, fitness: {dna.fitness_score:.3f}")
    
    def _calculate_fitness(self, dna: CommandDNA) -> float:
        """Calculate fitness score for a command's DNA."""
        traits = dna.traits
        
        # Weighted combination of traits
        fitness = (
            traits["usage_frequency"] * 0.3 +
            traits["success_rate"] * 0.3 +
            traits["execution_speed"] * 0.2 +
            traits["versatility"] * 0.1 +
            traits["safety"] * 0.1
        )
        
        return max(0.0, min(1.0, fitness))
    
    def _discover_patterns(self):
        """Discover usage patterns from recent command sequences."""
        if len(self.usage_sequences) < 5:
            return
        
        recent_sequences = self.usage_sequences[-50:]  # Look at recent usage
        
        # Discover sequence patterns
        self._discover_sequence_patterns(recent_sequences)
        
        # Discover correlation patterns
        self._discover_correlation_patterns(recent_sequences)
        
        # Discover temporal patterns
        self._discover_temporal_patterns(recent_sequences)
        
        # Discover contextual patterns
        self._discover_contextual_patterns(recent_sequences)
    
    def _discover_sequence_patterns(self, sequences: List[Dict]):
        """Discover common command sequences."""
        # Extract command sequences
        command_sequences = []
        current_sequence = []
        
        for i, seq in enumerate(sequences):
            current_sequence.append(seq["command_id"])
            
            # Break sequence on time gaps or context changes
            if (i > 0 and 
                (datetime.fromisoformat(seq["timestamp"]) - 
                 datetime.fromisoformat(sequences[i-1]["timestamp"])).total_seconds() > 300):
                if len(current_sequence) > 1:
                    command_sequences.append(current_sequence.copy())
                current_sequence = [seq["command_id"]]
        
        # Find common subsequences
        subsequence_counts = Counter()
        for seq in command_sequences:
            for length in range(2, min(5, len(seq) + 1)):
                for start in range(len(seq) - length + 1):
                    subseq = tuple(seq[start:start + length])
                    subsequence_counts[subseq] += 1
        
        # Create patterns for frequent subsequences
        for subseq, count in subsequence_counts.items():
            if count >= 3:  # Minimum frequency
                pattern_id = f"seq_{hashlib.md5(str(subseq).encode()).hexdigest()[:8]}"
                
                if pattern_id not in self.learning_patterns:
                    pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type="sequence",
                        confidence=min(1.0, count / 10.0),
                        frequency=count,
                        discovered_at=datetime.now().isoformat(),
                        elements=list(subseq),
                        context={"length": len(subseq), "type": "command_sequence"}
                    )
                    self.learning_patterns[pattern_id] = pattern
                    self.logger.info(f"Discovered sequence pattern: {' -> '.join(subseq)}")
    
    def _discover_correlation_patterns(self, sequences: List[Dict]):
        """Discover correlations between commands and contexts."""
        # Build correlation matrix
        command_ids = list(set(seq["command_id"] for seq in sequences))
        correlations = defaultdict(lambda: defaultdict(int))
        
        for seq in sequences:
            cmd_id = seq["command_id"]
            context = seq["context"]
            
            # Correlate with context elements
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    correlations[cmd_id][f"context_{key}_{value}"] += 1
        
        # Find strong correlations
        for cmd_id, context_counts in correlations.items():
            total_usage = sum(context_counts.values())
            
            for context_key, count in context_counts.items():
                correlation_strength = count / total_usage
                
                if correlation_strength > 0.7:  # Strong correlation
                    pattern_id = f"corr_{cmd_id}_{context_key}"
                    
                    if pattern_id not in self.learning_patterns:
                        pattern = LearningPattern(
                            pattern_id=pattern_id,
                            pattern_type="correlation",
                            confidence=correlation_strength,
                            frequency=count,
                            discovered_at=datetime.now().isoformat(),
                            elements=[cmd_id, context_key],
                            context={"correlation_type": "command_context"}
                        )
                        self.learning_patterns[pattern_id] = pattern
                        self.logger.info(f"Discovered correlation: {cmd_id} ↔ {context_key}")
    
    def _discover_temporal_patterns(self, sequences: List[Dict]):
        """Discover time-based usage patterns."""
        # Group by hour of day
        hourly_usage = defaultdict(lambda: defaultdict(int))
        
        for seq in sequences:
            timestamp = datetime.fromisoformat(seq["timestamp"])
            hour = timestamp.hour
            cmd_id = seq["command_id"]
            hourly_usage[hour][cmd_id] += 1
        
        # Find commands with strong temporal patterns
        for hour, cmd_counts in hourly_usage.items():
            total_hour_usage = sum(cmd_counts.values())
            
            for cmd_id, count in cmd_counts.items():
                if total_hour_usage > 0:
                    temporal_strength = count / total_hour_usage
                    
                    if temporal_strength > 0.5:  # Used frequently at this hour
                        pattern_id = f"temporal_{cmd_id}_hour_{hour}"
                        
                        if pattern_id not in self.learning_patterns:
                            pattern = LearningPattern(
                                pattern_id=pattern_id,
                                pattern_type="temporal",
                                confidence=temporal_strength,
                                frequency=count,
                                discovered_at=datetime.now().isoformat(),
                                elements=[cmd_id],
                                context={"hour": hour, "pattern_type": "hourly_usage"}
                            )
                            self.learning_patterns[pattern_id] = pattern
                            self.logger.info(f"Discovered temporal pattern: {cmd_id} @ hour {hour}")
    
    def _discover_contextual_patterns(self, sequences: List[Dict]):
        """Discover context-based command patterns."""
        # Group by context similarity
        context_groups = defaultdict(list)
        
        for seq in sequences:
            context_signature = self._create_context_signature(seq["context"])
            context_groups[context_signature].append(seq["command_id"])
        
        # Find context patterns
        for context_sig, cmd_list in context_groups.items():
            if len(cmd_list) >= 3:  # Minimum frequency
                cmd_counter = Counter(cmd_list)
                most_common = cmd_counter.most_common(1)[0]
                
                if most_common[1] >= 2:  # Command used multiple times in this context
                    pattern_id = f"context_{context_sig}_{most_common[0]}"
                    
                    if pattern_id not in self.learning_patterns:
                        pattern = LearningPattern(
                            pattern_id=pattern_id,
                            pattern_type="contextual",
                            confidence=most_common[1] / len(cmd_list),
                            frequency=most_common[1],
                            discovered_at=datetime.now().isoformat(),
                            elements=[most_common[0]],
                            context={"context_signature": context_sig}
                        )
                        self.learning_patterns[pattern_id] = pattern
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for context similarity."""
        # Simplified context signature based on key-value pairs
        sorted_items = sorted(context.items())
        signature_parts = []
        
        for key, value in sorted_items:
            if isinstance(value, (str, int, float, bool)):
                signature_parts.append(f"{key}:{value}")
        
        return hashlib.md5("|".join(signature_parts).encode()).hexdigest()[:8]
    
    def evolve_commands(self) -> List[Command]:
        """Evolve new commands based on learned patterns and genetics."""
        new_commands = []
        
        # Get high-fitness commands for evolution
        fit_commands = [dna for dna in self.command_dna.values() if dna.fitness_score > 0.7]
        
        if len(fit_commands) < 2:
            return new_commands
        
        # Create new commands through various evolution strategies
        
        # 1. Mutation - modify existing high-fitness commands
        for dna in fit_commands[:3]:  # Mutate top 3 commands
            if random.random() < self.mutation_rate:
                mutated_command = self._mutate_command(dna)
                if mutated_command:
                    new_commands.append(mutated_command)
        
        # 2. Crossover - combine successful commands
        for i in range(min(3, len(fit_commands) - 1)):
            parent1 = fit_commands[i]
            parent2 = fit_commands[i + 1]
            
            if random.random() < 0.3:  # 30% chance of crossover
                crossed_command = self._crossover_commands(parent1, parent2)
                if crossed_command:
                    new_commands.append(crossed_command)
        
        # 3. Pattern-based generation - create commands based on learned patterns
        pattern_commands = self._generate_pattern_based_commands()
        new_commands.extend(pattern_commands)
        
        # Store new commands in database
        for command in new_commands:
            try:
                self.db.store_command(command)
                self.logger.info(f"Evolved new command: {command.name}")
            except Exception as e:
                self.logger.error(f"Failed to store evolved command: {e}")
        
        return new_commands
    
    def _mutate_command(self, parent_dna: CommandDNA) -> Optional[Command]:
        """Create a mutated version of a command."""
        try:
            # Get original command
            original_command = self.db.get_command(parent_dna.command_id)
            if not original_command:
                return None
            
            # Create mutated version
            mutated_genes = parent_dna.genes.copy()
            mutations = []
            
            # Mutate description
            if random.random() < 0.3:
                mutated_genes["description"] = self._mutate_description(mutated_genes["description"])
                mutations.append("description")
            
            # Mutate keywords
            if random.random() < 0.4:
                mutated_genes["keywords"] = self._mutate_keywords(mutated_genes["keywords"])
                mutations.append("keywords")
            
            # Mutate flags
            if random.random() < 0.2:
                mutated_genes["flags"] = self._mutate_flags(mutated_genes["flags"])
                mutations.append("flags")
            
            # Create new command
            new_command_id = f"evolved_{parent_dna.command_id}_{int(time.time())}"
            new_command = Command(
                id=new_command_id,
                number=self._generate_evolved_number(parent_dna.genes["category"]),
                name=f"evolved {original_command.name}",
                description=mutated_genes["description"],
                category=mutated_genes["category"],
                subcategory="Evolved",
                executable=mutated_genes["executable"],
                args=mutated_genes["args"],
                flags=mutated_genes["flags"],
                examples=[f"# Evolved command based on {original_command.name}"],
                keywords=mutated_genes["keywords"],
                source_file=f"evolved_from_{original_command.source_file}",
                complexity="medium"
            )
            
            # Create DNA for new command
            new_dna = CommandDNA(
                command_id=new_command_id,
                traits=parent_dna.traits.copy(),
                genes=mutated_genes,
                fitness_score=parent_dna.fitness_score * 0.8,  # Start with reduced fitness
                generation=parent_dna.generation + 1,
                parent_ids=[parent_dna.command_id],
                mutations=mutations,
                creation_time=datetime.now().isoformat()
            )
            
            self.command_dna[new_command_id] = new_dna
            self.evolution_history.append({
                "type": "mutation",
                "parent": parent_dna.command_id,
                "child": new_command_id,
                "mutations": mutations,
                "timestamp": datetime.now().isoformat()
            })
            
            return new_command
            
        except Exception as e:
            self.logger.error(f"Mutation failed: {e}")
            return None
    
    def _crossover_commands(self, parent1_dna: CommandDNA, parent2_dna: CommandDNA) -> Optional[Command]:
        """Create a new command by crossing over two parent commands."""
        try:
            # Combine genes from both parents
            child_genes = {}
            
            # Take best traits from each parent
            child_genes["executable"] = parent1_dna.genes["executable"]  # Keep primary parent's executable
            child_genes["category"] = parent1_dna.genes["category"]
            
            # Combine descriptions
            desc1 = parent1_dna.genes["description"]
            desc2 = parent2_dna.genes["description"]
            child_genes["description"] = f"Hybrid: {desc1[:30]}... + {desc2[:30]}..."
            
            # Combine keywords
            keywords1 = set(parent1_dna.genes["keywords"])
            keywords2 = set(parent2_dna.genes["keywords"])
            child_genes["keywords"] = list(keywords1.union(keywords2))
            
            # Combine flags
            flags1 = parent1_dna.genes["flags"]
            flags2 = parent2_dna.genes["flags"]
            child_genes["flags"] = {**flags1, **flags2}
            
            # Combine args (take from higher fitness parent)
            if parent1_dna.fitness_score >= parent2_dna.fitness_score:
                child_genes["args"] = parent1_dna.genes["args"]
            else:
                child_genes["args"] = parent2_dna.genes["args"]
            
            # Create new command
            new_command_id = f"hybrid_{parent1_dna.command_id}_{parent2_dna.command_id}_{int(time.time())}"
            new_command = Command(
                id=new_command_id,
                number=self._generate_evolved_number(child_genes["category"]),
                name=f"hybrid command",
                description=child_genes["description"],
                category=child_genes["category"],
                subcategory="Hybrid",
                executable=child_genes["executable"],
                args=child_genes["args"],
                flags=child_genes["flags"],
                examples=[f"# Hybrid command from {parent1_dna.command_id} + {parent2_dna.command_id}"],
                keywords=child_genes["keywords"],
                source_file="evolved_hybrid",
                complexity="medium"
            )
            
            # Create DNA for hybrid command
            child_traits = {}
            for trait, value in parent1_dna.traits.items():
                child_traits[trait] = (value + parent2_dna.traits[trait]) / 2
            
            new_dna = CommandDNA(
                command_id=new_command_id,
                traits=child_traits,
                genes=child_genes,
                fitness_score=(parent1_dna.fitness_score + parent2_dna.fitness_score) / 2,
                generation=max(parent1_dna.generation, parent2_dna.generation) + 1,
                parent_ids=[parent1_dna.command_id, parent2_dna.command_id],
                mutations=["crossover"],
                creation_time=datetime.now().isoformat()
            )
            
            self.command_dna[new_command_id] = new_dna
            self.evolution_history.append({
                "type": "crossover",
                "parents": [parent1_dna.command_id, parent2_dna.command_id],
                "child": new_command_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return new_command
            
        except Exception as e:
            self.logger.error(f"Crossover failed: {e}")
            return None
    
    def _generate_pattern_based_commands(self) -> List[Command]:
        """Generate new commands based on learned usage patterns."""
        new_commands = []
        
        # Generate commands from sequence patterns
        for pattern in self.learning_patterns.values():
            if pattern.pattern_type == "sequence" and pattern.confidence > 0.8:
                macro_command = self._create_macro_command(pattern)
                if macro_command:
                    new_commands.append(macro_command)
        
        return new_commands[:3]  # Limit to 3 pattern-based commands per cycle
    
    def _create_macro_command(self, sequence_pattern: LearningPattern) -> Optional[Command]:
        """Create a macro command that executes a sequence of commands."""
        try:
            command_sequence = sequence_pattern.elements
            
            if len(command_sequence) < 2:
                return None
            
            # Create macro command
            macro_id = f"macro_{sequence_pattern.pattern_id}_{int(time.time())}"
            macro_name = f"macro {' -> '.join(cmd.split('_')[-1] for cmd in command_sequence[:2])}"
            
            macro_command = Command(
                id=macro_id,
                number=self._generate_evolved_number("5.3"),
                name=macro_name,
                description=f"Macro command executing sequence: {' → '.join(command_sequence)}",
                category="5.3",
                subcategory="Macros",
                executable="python",
                args=["-c", f"# Execute sequence: {' -> '.join(command_sequence)}"],
                flags={"--dry-run": "Preview sequence without execution"},
                examples=[f"{macro_name} --dry-run"],
                keywords=["macro", "sequence", "automation"] + [cmd.split('_')[-1] for cmd in command_sequence],
                source_file="learned_pattern",
                complexity="medium"
            )
            
            # Create DNA for macro command
            macro_dna = CommandDNA(
                command_id=macro_id,
                traits={
                    "complexity": 0.6,
                    "usage_frequency": sequence_pattern.confidence,
                    "success_rate": 0.8,
                    "execution_speed": 0.5,  # Slower due to multiple commands
                    "versatility": len(command_sequence) / 10.0,
                    "safety": 0.7  # Medium safety due to automation
                },
                genes={
                    "executable": "python",
                    "args": ["-c", f"# Execute sequence: {' -> '.join(command_sequence)}"],
                    "flags": {"--dry-run": "Preview sequence without execution"},
                    "keywords": ["macro", "sequence", "automation"],
                    "category": "5.3",
                    "description": macro_command.description,
                    "sequence": command_sequence  # Special gene for macros
                },
                fitness_score=sequence_pattern.confidence,
                generation=1,
                parent_ids=command_sequence,
                mutations=["pattern_generation"],
                creation_time=datetime.now().isoformat()
            )
            
            self.command_dna[macro_id] = macro_dna
            return macro_command
            
        except Exception as e:
            self.logger.error(f"Failed to create macro command: {e}")
            return None
    
    def _mutate_description(self, description: str) -> str:
        """Mutate command description."""
        adjectives = ["enhanced", "optimized", "improved", "advanced", "intelligent", "adaptive"]
        return f"{random.choice(adjectives)} {description.lower()}"
    
    def _mutate_keywords(self, keywords: List[str]) -> List[str]:
        """Mutate command keywords."""
        new_keywords = keywords.copy()
        
        # Add new keywords
        additional_keywords = ["smart", "auto", "enhanced", "v2", "pro", "plus"]
        if random.random() < 0.5:
            new_keywords.append(random.choice(additional_keywords))
        
        return new_keywords
    
    def _mutate_flags(self, flags: Dict[str, str]) -> Dict[str, str]:
        """Mutate command flags."""
        new_flags = flags.copy()
        
        # Add common useful flags
        additional_flags = {
            "--enhanced": "Enable enhanced mode",
            "--auto": "Automatic mode",
            "--smart": "Smart optimization",
            "--turbo": "High performance mode"
        }
        
        if random.random() < 0.3:
            flag_name, flag_desc = random.choice(list(additional_flags.items()))
            new_flags[flag_name] = flag_desc
        
        return new_flags
    
    def _generate_evolved_number(self, category: str) -> str:
        """Generate command number for evolved commands."""
        base_category = category.split('.')[0] if '.' in category else category
        subcategory = random.randint(80, 99)  # Reserve 80-99 for evolved commands
        command_num = random.randint(1, 999)
        return f"{base_category}.{subcategory}.{command_num}"
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system."""
        total_patterns = len(self.learning_patterns)
        pattern_types = Counter(p.pattern_type for p in self.learning_patterns.values())
        
        # Calculate average fitness by generation
        fitness_by_generation = defaultdict(list)
        for dna in self.command_dna.values():
            fitness_by_generation[dna.generation].append(dna.fitness_score)
        
        avg_fitness_by_gen = {}
        for gen, scores in fitness_by_generation.items():
            avg_fitness_by_gen[gen] = sum(scores) / len(scores)
        
        return {
            "total_commands": len(self.command_dna),
            "total_patterns": total_patterns,
            "pattern_types": dict(pattern_types),
            "usage_sequences": len(self.usage_sequences),
            "evolution_events": len(self.evolution_history),
            "avg_fitness_by_generation": avg_fitness_by_gen,
            "mutation_rate": self.mutation_rate,
            "learning_rate": self.learning_rate
        }
    
    def boost_learning(self, factor: float = 1.5):
        """Boost learning parameters to accelerate evolution."""
        self.mutation_rate = min(0.5, self.mutation_rate * factor)
        self.learning_rate = min(0.2, self.learning_rate * factor)
        self.logger.info(f"Boosted learning: mutation_rate={self.mutation_rate:.3f}, learning_rate={self.learning_rate:.3f}")

def test_adaptive_learning():
    """Test the adaptive learning system."""
    print("🧠 Testing Adaptive Learning System")
    print("=" * 50)
    
    from command_database import CommandDatabase
    db = CommandDatabase()
    learning_engine = AdaptiveLearningEngine(db)
    
    # Show initial stats
    stats = learning_engine.get_learning_stats()
    print(f"Initialized learning for {stats['total_commands']} commands")
    
    # Simulate usage patterns
    print("\n📊 Simulating usage patterns...")
    commands = db.search_commands("")[:5]  # Get first 5 commands
    
    for i in range(20):  # Simulate 20 usage events
        cmd = random.choice(commands)
        context = {
            "user": "test_user",
            "time_of_day": random.choice(["morning", "afternoon", "evening"]),
            "project": random.choice(["neural_sim", "data_analysis", "deployment"])
        }
        success = random.choice([True, True, True, False])  # 75% success rate
        exec_time = random.uniform(0.5, 5.0)
        
        learning_engine.learn_from_usage(cmd.id, context, success, exec_time)
    
    # Try evolution
    print("\n🧬 Attempting command evolution...")
    new_commands = learning_engine.evolve_commands()
    print(f"Evolved {len(new_commands)} new commands")
    
    for cmd in new_commands:
        print(f"  • {cmd.name}: {cmd.description}")
    
    # Show final stats
    final_stats = learning_engine.get_learning_stats()
    print(f"\nFinal stats:")
    print(f"  Patterns discovered: {final_stats['total_patterns']}")
    print(f"  Pattern types: {final_stats['pattern_types']}")
    print(f"  Evolution events: {final_stats['evolution_events']}")
    
    db.close()

if __name__ == "__main__":
    test_adaptive_learning()
