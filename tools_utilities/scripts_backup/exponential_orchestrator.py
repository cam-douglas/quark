#!/usr/bin/env python3
"""
Exponential Growth Orchestrator

This module orchestrates the exponential growth of commands through the coordination
of curious agents, adaptive learning, and intelligent exploration engines.
"""

import os, sys
import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import random

from ................................................command_database import CommandDatabase
from ................................................curious_agents import ExponentialGrowthEngine, CuriousAgent
from ................................................adaptive_learning import AdaptiveLearningEngine
from ................................................neuro_integration import NeuroAgentConnector, SmartCommandDiscovery

@dataclass
class GrowthMetrics:
    """Metrics tracking exponential growth."""
    timestamp: str
    total_commands: int
    new_discoveries: int
    evolution_events: int
    learning_patterns: int
    growth_rate: float
    curiosity_level: float
    exploration_breadth: int
    fitness_improvement: float

class ExponentialOrchestrator:
    """Master orchestrator for exponential command growth."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger("exponential_orchestrator")
        
        # Initialize core systems
        self.db = CommandDatabase()
        self.growth_engine = ExponentialGrowthEngine(self.db)
        self.learning_engine = AdaptiveLearningEngine(self.db)
        self.neuro_connector = NeuroAgentConnector()
        self.smart_discovery = SmartCommandDiscovery(self.neuro_connector)
        
        # Growth state
        self.is_active = False
        self.growth_cycles = 0
        self.total_discoveries = 0
        self.growth_metrics = []
        self.exponential_factor = 1.2  # Commands grow by 20% each cycle
        
        # Growth parameters
        self.cycle_interval = 60  # 1 minute between cycles initially
        self.max_cycle_interval = 1800  # Max 30 minutes between cycles
        self.curiosity_boost_threshold = 0.8
        self.learning_acceleration_factor = 1.5
        
        # Discovery targets
        self.exploration_targets = self._initialize_exploration_targets()
        
        # Threading
        self.orchestrator_thread = None
        self.growth_lock = threading.Lock()
        
        self.logger.info("Exponential Orchestrator initialized")
    
    def _initialize_exploration_targets(self) -> List[Path]:
        """Initialize targets for exploration."""
        targets = []
        
        # Project directories
        for subdir in ["src", "scripts", "examples", "tools", "bin", "cli"]:
            target_path = self.project_root / subdir
            if target_path.exists():
                targets.append(target_path)
        
        # Look for additional interesting directories
        try:
            for item in self.project_root.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and
                    item.name not in ["__pycache__", "node_modules", ".git"]):
                    
                    # Check if directory contains potentially interesting files
                    interesting_files = 0
                    try:
                        for file_item in item.iterdir():
                            if file_item.suffix in ['.py', '.sh', '.js', '.yml', '.json']:
                                interesting_files += 1
                                if interesting_files >= 3:
                                    targets.append(item)
                                    break
                    except PermissionError:
                        continue
        except Exception as e:
            self.logger.debug(f"Error scanning project root: {e}")
        
        self.logger.info(f"Initialized {len(targets)} exploration targets")
        return targets
    
    def start_exponential_growth(self):
        """Start the exponential growth process."""
        if self.is_active:
            self.logger.warning("Exponential growth already active")
            return
        
        self.is_active = True
        self.orchestrator_thread = threading.Thread(
            target=self._orchestration_loop,
            daemon=True
        )
        self.orchestrator_thread.start()
        
        # Start component engines
        self.growth_engine.start_exponential_discovery(self.project_root)
        
        self.logger.info("🚀 Exponential growth process started!")
        self._log_growth_start()
    
    def stop_exponential_growth(self):
        """Stop the exponential growth process."""
        if not self.is_active:
            return
        
        self.is_active = False
        self.growth_engine.stop_discovery()
        
        if self.orchestrator_thread:
            self.orchestrator_thread.join(timeout=10)
        
        self.logger.info("⏹️  Exponential growth process stopped")
        self._log_growth_summary()
    
    def _orchestration_loop(self):
        """Main orchestration loop that coordinates exponential growth."""
        while self.is_active:
            try:
                cycle_start = time.time()
                self.growth_cycles += 1
                
                self.logger.info(f"🔄 Starting growth cycle {self.growth_cycles}")
                
                # Capture initial state
                initial_commands = len(self.db.search_commands(""))
                
                # Run orchestrated growth cycle
                cycle_results = self._run_growth_cycle()
                
                # Calculate growth metrics
                final_commands = len(self.db.search_commands(""))
                new_discoveries = final_commands - initial_commands
                self.total_discoveries += new_discoveries
                
                # Record metrics
                metrics = self._calculate_growth_metrics(
                    initial_commands, final_commands, new_discoveries, cycle_results
                )
                self.growth_metrics.append(metrics)
                
                # Adapt parameters based on performance
                self._adapt_growth_parameters(metrics)
                
                # Log cycle completion
                cycle_time = time.time() - cycle_start
                self.logger.info(
                    f"✅ Cycle {self.growth_cycles} complete: "
                    f"{new_discoveries} new commands in {cycle_time:.1f}s"
                )
                
                # Calculate next cycle interval (exponential backoff with discoveries)
                next_interval = self._calculate_next_interval(new_discoveries)
                time.sleep(next_interval)
                
            except Exception as e:
                self.logger.error(f"Growth cycle error: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _run_growth_cycle(self) -> Dict[str, Any]:
        """Run a single coordinated growth cycle."""
        cycle_results = {
            "agent_discoveries": 0,
            "learning_evolutions": 0,
            "neuro_discoveries": 0,
            "pattern_discoveries": 0,
            "exploration_expansion": 0
        }
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            # 1. Boost curious agents based on current state
            futures['agents'] = executor.submit(self._boost_curious_agents)
            
            # 2. Run adaptive learning cycle
            futures['learning'] = executor.submit(self._run_learning_cycle)
            
            # 3. Expand exploration using neuro agents
            futures['neuro'] = executor.submit(self._run_neuro_exploration)
            
            # 4. Discover new patterns and connections
            futures['patterns'] = executor.submit(self._discover_new_patterns)
            
            # Collect results
            for key, future in futures.items():
                try:
                    result = future.result(timeout=30)
                    if key == 'agents':
                        cycle_results['agent_discoveries'] = result
                    elif key == 'learning':
                        cycle_results['learning_evolutions'] = result
                    elif key == 'neuro':
                        cycle_results['neuro_discoveries'] = result
                    elif key == 'patterns':
                        cycle_results['pattern_discoveries'] = result
                except Exception as e:
                    self.logger.error(f"Growth cycle component '{key}' failed: {e}")
        
        return cycle_results
    
    def _boost_curious_agents(self) -> int:
        """Boost curious agents based on current growth state."""
        discoveries = 0
        
        try:
            # Get current growth stats
            growth_stats = self.growth_engine.get_growth_stats()
            
            # Boost agents with low energy or high discoveries
            for agent_id, agent_stats in growth_stats['agents'].items():
                if (agent_stats['energy'] < 30 or 
                    agent_stats['discoveries'] > 10):
                    
                    # Calculate boost factor based on performance
                    boost_factor = 1.0 + (agent_stats['discoveries'] / 20.0)
                    self.growth_engine.boost_curiosity(agent_id, boost_factor)
                    discoveries += 1
            
            # Add new specialized agents if growth is high
            if self.total_discoveries > 50 and len(growth_stats['agents']) < 10:
                new_specializations = [
                    ["automation", "workflow", "pipeline"],
                    ["security", "audit", "compliance"],
                    ["performance", "optimization", "benchmark"],
                    ["documentation", "tutorial", "example"]
                ]
                
                for specialization in new_specializations:
                    agent_id = f"specialist_{specialization[0]}_{int(time.time())}"
                    new_agent = CuriousAgent(agent_id, specialization)
                    self.growth_engine.agents[agent_id] = new_agent
                    self.logger.info(f"Added new specialist agent: {agent_id}")
                    discoveries += 1
                    break  # Add one per cycle
            
        except Exception as e:
            self.logger.error(f"Failed to boost curious agents: {e}")
        
        return discoveries
    
    def _run_learning_cycle(self) -> int:
        """Run adaptive learning and evolution cycle."""
        evolutions = 0
        
        try:
            # Simulate some usage patterns to feed the learning engine
            self._simulate_usage_patterns()
            
            # Run command evolution
            new_commands = self.learning_engine.evolve_commands()
            evolutions = len(new_commands)
            
            # Boost learning if growth is slow
            if self.growth_cycles > 5 and evolutions == 0:
                self.learning_engine.boost_learning(self.learning_acceleration_factor)
                self.logger.info("Boosted learning parameters due to slow evolution")
            
        except Exception as e:
            self.logger.error(f"Learning cycle failed: {e}")
        
        return evolutions
    
    def _run_neuro_exploration(self) -> int:
        """Use neuro agents to discover new command sources."""
        discoveries = 0
        
        try:
            if self.neuro_connector.is_available():
                # Use neuro agents to discover project-specific commands
                discovered_commands = self.smart_discovery.discover_project_commands(
                    str(self.project_root)
                )
                
                # Store discovered commands
                for cmd_data in discovered_commands:
                    try:
                        # Convert discovered command data to Command object
                        from ................................................command_database import Command
                        
                        command = Command(
                            id=f"neuro_discovered_{int(time.time())}_{discoveries}",
                            number=self._generate_discovery_number(),
                            name=cmd_data['name'],
                            description=cmd_data['description'],
                            category=cmd_data.get('category', '6.1'),
                            subcategory="Discovered",
                            executable=cmd_data['executable'],
                            args=cmd_data['args'],
                            flags={},
                            examples=[f"{cmd_data['executable']} {' '.join(cmd_data['args'])}"],
                            keywords=cmd_data.get('keywords', ['discovered']),
                            source_file=cmd_data.get('source', 'neuro_discovery'),
                            complexity="low"
                        )
                        
                        self.db.store_command(command)
                        discoveries += 1
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to store neuro discovery: {e}")
            
            # Expand exploration targets based on neuro analysis
            if discoveries > 0:
                self._expand_exploration_targets()
            
        except Exception as e:
            self.logger.error(f"Neuro exploration failed: {e}")
        
        return discoveries
    
    def _discover_new_patterns(self) -> int:
        """Discover new patterns and meta-commands."""
        patterns = 0
        
        try:
            # Analyze command usage patterns
            stats = self.learning_engine.get_learning_stats()
            
            # Create meta-commands based on frequent patterns
            if stats['total_patterns'] > 10:
                # Look for commands that are often used together
                all_commands = self.db.search_commands("")
                command_pairs = []
                
                # Simple pattern: find commands in same category used frequently
                category_groups = {}
                for cmd in all_commands:
                    if cmd.category not in category_groups:
                        category_groups[cmd.category] = []
                    category_groups[cmd.category].append(cmd)
                
                # Create workflow commands for large categories
                for category, commands in category_groups.items():
                    if len(commands) > 5:
                        workflow_cmd = self._create_workflow_command(category, commands)
                        if workflow_cmd:
                            self.db.store_command(workflow_cmd)
                            patterns += 1
            
        except Exception as e:
            self.logger.error(f"Pattern discovery failed: {e}")
        
        return patterns
    
    def _simulate_usage_patterns(self):
        """Simulate realistic usage patterns for learning."""
        try:
            commands = self.db.search_commands("")
            
            # Simulate different usage scenarios
            scenarios = [
                {"user": "dev_user", "context": "development", "time": "morning"},
                {"user": "data_scientist", "context": "analysis", "time": "afternoon"},
                {"user": "ops_user", "context": "deployment", "time": "evening"}
            ]
            
            for scenario in scenarios:
                # Pick commands relevant to the scenario
                relevant_commands = [
                    cmd for cmd in commands 
                    if any(keyword in ' '.join(cmd.keywords) for keyword in 
                          scenario["context"].split())
                ]
                
                if relevant_commands:
                    cmd = random.choice(relevant_commands)
                    success = random.choice([True, True, True, False])  # 75% success
                    exec_time = random.uniform(0.5, 3.0)
                    
                    self.learning_engine.learn_from_usage(
                        cmd.id, scenario, success, exec_time
                    )
        
        except Exception as e:
            self.logger.debug(f"Usage simulation failed: {e}")
    
    def _expand_exploration_targets(self):
        """Expand exploration targets based on discoveries."""
        try:
            # Look for new directories that might have been created or discovered
            for target in self.exploration_targets.copy():
                if target.exists():
                    for item in target.iterdir():
                        if (item.is_dir() and 
                            item not in self.exploration_targets and
                            not item.name.startswith('.')):
                            self.exploration_targets.append(item)
                            self.logger.debug(f"Added exploration target: {item}")
        
        except Exception as e:
            self.logger.debug(f"Failed to expand targets: {e}")
    
    def _create_workflow_command(self, category: str, commands: List) -> Optional:
        """Create a workflow command that orchestrates multiple commands."""
        try:
            from ................................................command_database import Command
            
            workflow_id = f"workflow_{category.replace('.', '_')}_{int(time.time())}"
            
            # Create a workflow that runs common commands in sequence
            sample_commands = commands[:3]  # Take first 3 commands
            
            workflow_command = Command(
                id=workflow_id,
                number=self._generate_discovery_number(),
                name=f"workflow {category}",
                description=f"Automated workflow for {category} commands",
                category=category,
                subcategory="Workflows",
                executable="python",
                args=["-c", f"# Workflow: {' -> '.join(cmd.name for cmd in sample_commands)}"],
                flags={"--dry-run": "Preview workflow steps"},
                examples=[f"workflow {category} --dry-run"],
                keywords=["workflow", "automation", category.replace('.', '_')],
                source_file="pattern_discovery",
                complexity="medium"
            )
            
            return workflow_command
            
        except Exception as e:
            self.logger.debug(f"Failed to create workflow command: {e}")
            return None
    
    def _generate_discovery_number(self) -> str:
        """Generate command number for discovered commands."""
        # Use category 6 for discovered commands
        subcategory = random.randint(1, 99)
        command_num = random.randint(1, 999)
        return f"6.{subcategory}.{command_num}"
    
    def _calculate_growth_metrics(self, initial_commands: int, final_commands: int, 
                                new_discoveries: int, cycle_results: Dict) -> GrowthMetrics:
        """Calculate comprehensive growth metrics."""
        
        # Calculate growth rate
        if initial_commands > 0:
            growth_rate = (final_commands - initial_commands) / initial_commands
        else:
            growth_rate = 1.0 if final_commands > 0 else 0.0
        
        # Calculate curiosity level (average agent energy)
        growth_stats = self.growth_engine.get_growth_stats()
        agent_energies = [stats['energy'] for stats in growth_stats['agents'].values()]
        curiosity_level = sum(agent_energies) / len(agent_energies) if agent_energies else 0.0
        
        # Calculate fitness improvement
        learning_stats = self.learning_engine.get_learning_stats()
        fitness_by_gen = learning_stats.get('avg_fitness_by_generation', {})
        
        if len(fitness_by_gen) > 1:
            generations = sorted(fitness_by_gen.keys())
            latest_fitness = fitness_by_gen[generations[-1]]
            previous_fitness = fitness_by_gen[generations[-2]]
            fitness_improvement = latest_fitness - previous_fitness
        else:
            fitness_improvement = 0.0
        
        return GrowthMetrics(
            timestamp=datetime.now().isoformat(),
            total_commands=final_commands,
            new_discoveries=new_discoveries,
            evolution_events=cycle_results.get('learning_evolutions', 0),
            learning_patterns=learning_stats.get('total_patterns', 0),
            growth_rate=growth_rate,
            curiosity_level=curiosity_level / 100.0,  # Normalize to 0-1
            exploration_breadth=len(self.exploration_targets),
            fitness_improvement=fitness_improvement
        )
    
    def _adapt_growth_parameters(self, metrics: GrowthMetrics):
        """Adapt growth parameters based on performance metrics."""
        
        # Increase exponential factor if growth is consistent
        if len(self.growth_metrics) >= 3:
            recent_growth_rates = [m.growth_rate for m in self.growth_metrics[-3:]]
            avg_growth_rate = sum(recent_growth_rates) / len(recent_growth_rates)
            
            if avg_growth_rate > 0.1:  # Good growth
                self.exponential_factor = min(2.0, self.exponential_factor * 1.1)
            elif avg_growth_rate < 0.05:  # Slow growth
                self.exponential_factor = max(1.1, self.exponential_factor * 0.9)
        
        # Boost curiosity if exploration is stagnating
        if metrics.curiosity_level < self.curiosity_boost_threshold:
            for agent_id in self.growth_engine.agents.keys():
                self.growth_engine.boost_curiosity(agent_id, 1.3)
            self.logger.info("Boosted all agents due to low curiosity")
        
        # Accelerate learning if patterns are rich but evolution is slow
        if (metrics.learning_patterns > 20 and 
            metrics.evolution_events == 0):
            self.learning_engine.boost_learning(self.learning_acceleration_factor)
            self.logger.info("Accelerated learning due to rich patterns")
    
    def _calculate_next_interval(self, discoveries: int) -> float:
        """Calculate the next cycle interval based on discoveries."""
        base_interval = self.cycle_interval
        
        # Shorter intervals if we're discovering lots of commands
        if discoveries > 10:
            multiplier = 0.5
        elif discoveries > 5:
            multiplier = 0.7
        elif discoveries > 0:
            multiplier = 1.0
        else:
            multiplier = min(2.0, 1.5 ** (self.growth_cycles / 10))  # Slow down if no discoveries
        
        next_interval = min(self.max_cycle_interval, base_interval * multiplier)
        
        self.logger.debug(f"Next cycle in {next_interval:.1f} seconds")
        return next_interval
    
    def get_exponential_status(self) -> Dict[str, Any]:
        """Get comprehensive status of exponential growth."""
        
        current_commands = len(self.db.search_commands(""))
        
        # Calculate exponential growth trend
        if len(self.growth_metrics) >= 2:
            initial_commands = self.growth_metrics[0].total_commands
            exponential_growth = current_commands / initial_commands if initial_commands > 0 else 1.0
        else:
            exponential_growth = 1.0
        
        # Get component statuses
        growth_stats = self.growth_engine.get_growth_stats()
        learning_stats = self.learning_engine.get_learning_stats()
        
        return {
            "is_active": self.is_active,
            "growth_cycles": self.growth_cycles,
            "total_discoveries": self.total_discoveries,
            "current_commands": current_commands,
            "exponential_factor": self.exponential_factor,
            "exponential_growth": exponential_growth,
            "exploration_targets": len(self.exploration_targets),
            "curious_agents": len(growth_stats.get('agents', {})),
            "learning_patterns": learning_stats.get('total_patterns', 0),
            "evolution_events": learning_stats.get('evolution_events', 0),
            "neuro_available": self.neuro_connector.is_available(),
            "recent_metrics": self.growth_metrics[-5:] if self.growth_metrics else []
        }
    
    def _log_growth_start(self):
        """Log the start of exponential growth."""
        initial_commands = len(self.db.search_commands(""))
        
        self.logger.info("🌱 EXPONENTIAL GROWTH INITIATED")
        self.logger.info(f"📊 Starting with {initial_commands} commands")
        self.logger.info(f"🎯 Target: Exponential factor {self.exponential_factor}")
        self.logger.info(f"🤖 Curious agents: {len(self.growth_engine.agents)}")
        self.logger.info(f"🧠 Neuro integration: {'✅' if self.neuro_connector.is_available() else '❌'}")
        self.logger.info(f"🔍 Exploration targets: {len(self.exploration_targets)}")
    
    def _log_growth_summary(self):
        """Log summary of exponential growth session."""
        final_commands = len(self.db.search_commands(""))
        
        if self.growth_metrics:
            initial_commands = self.growth_metrics[0].total_commands
            total_growth = final_commands - initial_commands
            growth_rate = total_growth / initial_commands if initial_commands > 0 else 0
        else:
            total_growth = 0
            growth_rate = 0
        
        self.logger.info("🏁 EXPONENTIAL GROWTH SESSION COMPLETE")
        self.logger.info(f"📈 Commands grew from {final_commands - total_growth} to {final_commands}")
        self.logger.info(f"🚀 Total discoveries: {self.total_discoveries}")
        self.logger.info(f"⚡ Growth rate: {growth_rate:.2%}")
        self.logger.info(f"🔄 Cycles completed: {self.growth_cycles}")
        
        if self.growth_metrics:
            avg_discovery_rate = sum(m.new_discoveries for m in self.growth_metrics) / len(self.growth_metrics)
            self.logger.info(f"📊 Average discoveries per cycle: {avg_discovery_rate:.1f}")

def test_exponential_orchestrator():
    """Test the exponential orchestrator."""
    print("🚀 Testing Exponential Growth Orchestrator")
    print("=" * 60)
    
    # Initialize orchestrator
    project_root = Path("/Users/camdouglas/quark")
    orchestrator = ExponentialOrchestrator(project_root)
    
    # Show initial status
    status = orchestrator.get_exponential_status()
    print(f"Initial commands: {status['current_commands']}")
    print(f"Curious agents: {status['curious_agents']}")
    print(f"Exploration targets: {status['exploration_targets']}")
    print(f"Neuro available: {status['neuro_available']}")
    
    # Run a few manual growth cycles
    print("\n🔄 Running manual growth cycles...")
    for i in range(3):
        print(f"\nCycle {i + 1}:")
        cycle_results = orchestrator._run_growth_cycle()
        
        for key, value in cycle_results.items():
            if value > 0:
                print(f"  {key}: {value}")
    
    # Show final status
    final_status = orchestrator.get_exponential_status()
    print(f"\nFinal commands: {final_status['current_commands']}")
    print(f"Total discoveries: {final_status['total_discoveries']}")
    
    # Cleanup
    orchestrator.db.close()

if __name__ == "__main__":
    test_exponential_orchestrator()
