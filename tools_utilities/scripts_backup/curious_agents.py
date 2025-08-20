#!/usr/bin/env python3
"""
Curious Agents - Naturally Curious Command Discovery System

This module implements agents that are naturally curious about finding new commands,
expanding the resource base exponentially through intelligent exploration and learning.
"""

import os, sys
import json
import time
import random
import asyncio
import threading
import subprocess
from typing import Dict, List, Optional, Any, Set, Tuple
from data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclasses import data_knowledge.datasets_knowledge.datasets_knowledge.datasetsclass, asdict
from pathlib import Path
import logging
from datetime import datetime, timedelta
import hashlib
import re
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

from .....................................................command_database import CommandDatabase, Command
from .....................................................neuro_integration import NeuroAgentConnector

@dataclass
class DiscoveryTarget:
    """Represents a target for command discovery."""
    path: str
    type: str  # file, directory, url, repository, package
    priority: float
    last_explored: Optional[str] = None
    discovery_count: int = 0
    curiosity_score: float = 1.0
    metadata: Dict[str, Any] = None

@dataclass
class CuriosityState:
    """Represents the curiosity state of an agent."""
    agent_id: str
    exploration_energy: float
    learning_rate: float
    discovery_count: int
    specialization: List[str]  # Areas of interest
    memory: Dict[str, Any]
    last_active: str
    
class CuriousAgent:
    """A naturally curious agent that discovers and learns about commands."""
    
    def __init__(self, agent_id: str, specializations: List[str] = None):
        self.agent_id = agent_id
        self.specializations = specializations or ["general"]
        self.curiosity_state = CuriosityState(
            agent_id=agent_id,
            exploration_energy=100.0,
            learning_rate=0.1,
            discovery_count=0,
            specialization=self.specializations,
            memory={},
            last_active=datetime.now().isoformat()
        )
        self.discovered_commands = []
        self.exploration_targets = []
        self.logger = logging.getLogger(f"curious_agent_{agent_id}")
        
    def get_curiosity_score(self, target: DiscoveryTarget) -> float:
        """Calculate curiosity score for a discovery target."""
        base_score = target.curiosity_score
        
        # Boost score based on specialization match
        specialization_boost = 0
        for spec in self.specializations:
            if spec.lower() in target.path.lower() or spec.lower() in target.type.lower():
                specialization_boost += 0.5
        
        # Reduce score if recently explored
        recency_penalty = 0
        if target.last_explored:
            last_explored = datetime.fromisoformat(target.last_explored)
            hours_since = (datetime.now() - last_explored).total_seconds() / 3600
            if hours_since < 24:
                recency_penalty = 0.5 * (24 - hours_since) / 24
        
        # Boost score for unexplored targets
        exploration_boost = 0.3 if target.discovery_count == 0 else 0
        
        # Random curiosity factor
        random_factor = random.uniform(0.8, 1.2)
        
        final_score = (base_score + specialization_boost + exploration_boost - recency_penalty) * random_factor
        return max(0.1, final_score)
    
    def discover_file_commands(self, file_path: Path) -> List[Command]:
        """Discover commands from a file through intelligent analysis."""
        commands = []
        
        try:
            if not file_path.exists() or file_path.is_dir():
                return commands
            
            # Read file content safely
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                return commands
            
            # Python script analysis
            if file_path.suffix == '.py':
                commands.extend(self._discover_python_commands(file_path, content))
            
            # Shell script analysis
            elif file_path.suffix in ['.sh', '.bash', '.zsh']:
                commands.extend(self._discover_shell_commands(file_path, content))
            
            # Makefile analysis
            elif file_path.name.lower() in ['makefile', 'makefile.mk']:
                commands.extend(self._discover_makefile_commands(file_path, content))
            
            # Package.json analysis
            elif file_path.name == 'package.json':
                commands.extend(self._discover_npm_commands(file_path, content))
            
            # Docker analysis
            elif file_path.name.lower() in ['dockerfile', 'docker-compose.yml']:
                commands.extend(self._discover_docker_commands(file_path, content))
            
            # Configuration files
            elif file_path.suffix in ['.yml', '.yaml', '.toml', '.ini']:
                commands.extend(self._discover_config_commands(file_path, content))
            
            # Jupyter notebooks
            elif file_path.suffix == '.ipynb':
                commands.extend(self._discover_notebook_commands(file_path, content))
            
            self.logger.info(f"Discovered {len(commands)} commands from {file_path}")
            
        except Exception as e:
            self.logger.debug(f"Error discovering commands from {file_path}: {e}")
        
        return commands
    
    def _discover_python_commands(self, file_path: Path, content: str) -> List[Command]:
        """Discover commands from Python files."""
        commands = []
        
        # Look for argparse-based CLIs
        if 'argparse' in content or 'ArgumentParser' in content:
            # Extract command line interface
            if 'def main(' in content or 'if __name__ == "__main__"' in content:
                command_id = f"python_{file_path.stem}_{int(time.time())}"
                commands.append(Command(
                    id=command_id,
                    number=self._generate_command_number(),
                    name=f"python {file_path.name}",
                    description=f"Execute Python script: {file_path.name}",
                    category="5.3",  # Development Tools -> Setup
                    subcategory="Scripts",
                    executable="python",
                    args=[str(file_path)],
                    flags=self._extract_python_flags(content),
                    examples=[f"python {file_path.name}"],
                    keywords=["python", "script", file_path.stem],
                    source_file=str(file_path),
                    complexity="low"
                ))
        
        # Look for click-based CLIs
        if '@click.command' in content or 'import click' in content:
            command_id = f"click_{file_path.stem}_{int(time.time())}"
            commands.append(Command(
                id=command_id,
                number=self._generate_command_number(),
                name=f"click {file_path.stem}",
                description=f"Click CLI: {file_path.name}",
                category="5.3",
                subcategory="CLI",
                executable="python",
                args=[str(file_path)],
                flags=self._extract_click_flags(content),
                examples=[f"python {file_path.name} --help"],
                keywords=["click", "cli", file_path.stem],
                source_file=str(file_path),
                complexity="medium"
            ))
        
        # Look for FastAPI apps
        if 'FastAPI' in content or '@app.get' in content:
            command_id = f"fastapi_{file_path.stem}_{int(time.time())}"
            commands.append(Command(
                id=command_id,
                number=self._generate_command_number(),
                name=f"run {file_path.stem} api",
                description=f"Start FastAPI server: {file_path.name}",
                category="4.2",  # Cloud Computing -> Deployment
                subcategory="API",
                executable="uvicorn",
                args=[f"{file_path.stem}:app", "--reload"],
                flags={"--host": "Host address", "--port": "Port number"},
                examples=[f"uvicorn {file_path.stem}:app --reload"],
                keywords=["fastapi", "api", "server", file_path.stem],
                source_file=str(file_path),
                complexity="medium"
            ))
        
        return commands
    
    def _discover_shell_commands(self, file_path: Path, content: str) -> List[Command]:
        """Discover commands from shell scripts."""
        commands = []
        
        # Basic shell script
        if content.startswith('#!') or 'bash' in content or 'sh' in content:
            command_id = f"shell_{file_path.stem}_{int(time.time())}"
            
            # Extract description from comments
            description = self._extract_shell_description(content)
            
            commands.append(Command(
                id=command_id,
                number=self._generate_command_number(),
                name=f"run {file_path.name}",
                description=description or f"Execute shell script: {file_path.name}",
                category="5.3",
                subcategory="Scripts",
                executable="bash",
                args=[str(file_path)],
                flags=self._extract_shell_flags(content),
                examples=[f"bash {file_path.name}"],
                keywords=["bash", "shell", "script", file_path.stem],
                source_file=str(file_path),
                requires_shell=True,
                complexity="medium"
            ))
        
        return commands
    
    def _discover_makefile_commands(self, file_path: Path, content: str) -> List[Command]:
        """Discover commands from Makefiles."""
        commands = []
        
        # Extract make targets
        targets = re.findall(r'^([a-zA-Z_][a-zA-Z0-9_-]*):(?!\=)', content, re.MULTILINE)
        
        for target in targets:
            if target not in ['all', '.PHONY']:  # Skip special targets
                command_id = f"make_{target}_{int(time.time())}"
                commands.append(Command(
                    id=command_id,
                    number=self._generate_command_number(),
                    name=f"make {target}",
                    description=f"Run make target: {target}",
                    category="5.1",  # Development Tools -> Testing
                    subcategory="Build",
                    executable="make",
                    args=[target],
                    flags={},
                    examples=[f"make {target}"],
                    keywords=["make", "build", target],
                    source_file=str(file_path),
                    complexity="low"
                ))
        
        return commands
    
    def _discover_npm_commands(self, file_path: Path, content: str) -> List[Command]:
        """Discover commands from package.json scripts."""
        commands = []
        
        try:
            package_data = json.loads(content)
            scripts = package_data.get('scripts', {})
            
            for script_name, script_command in scripts.items():
                command_id = f"npm_{script_name}_{int(time.time())}"
                commands.append(Command(
                    id=command_id,
                    number=self._generate_command_number(),
                    name=f"npm run {script_name}",
                    description=f"NPM script: {script_command}",
                    category="5.3",
                    subcategory="NPM",
                    executable="npm",
                    args=["run", script_name],
                    flags={},
                    examples=[f"npm run {script_name}"],
                    keywords=["npm", "script", script_name],
                    source_file=str(file_path),
                    complexity="low"
                ))
        except json.JSONDecodeError:
            pass
        
        return commands
    
    def _discover_docker_commands(self, file_path: Path, content: str) -> List[Command]:
        """Discover commands from Docker files."""
        commands = []
        
        if file_path.name.lower() == 'dockerfile':
            command_id = f"docker_build_{int(time.time())}"
            commands.append(Command(
                id=command_id,
                number=self._generate_command_number(),
                name="docker build",
                description="Build Docker image from Dockerfile",
                category="4.2",
                subcategory="Docker",
                executable="docker",
                args=["build", "-t", "app", "."],
                flags={"-t": "Tag name", "-f": "Dockerfile path"},
                examples=["docker build -t myapp ."],
                keywords=["docker", "build", "container"],
                source_file=str(file_path),
                complexity="medium"
            ))
        
        elif 'docker-compose' in file_path.name.lower():
            command_id = f"docker_compose_{int(time.time())}"
            commands.append(Command(
                id=command_id,
                number=self._generate_command_number(),
                name="docker-compose up",
                description="Start services with docker-compose",
                category="4.2",
                subcategory="Docker",
                executable="docker-compose",
                args=["up", "-d"],
                flags={"-d": "Detached mode", "--build": "Build images"},
                examples=["docker-compose up -d"],
                keywords=["docker", "compose", "services"],
                source_file=str(file_path),
                complexity="medium"
            ))
        
        return commands
    
    def _discover_config_commands(self, file_path: Path, content: str) -> List[Command]:
        """Discover commands from configuration files."""
        commands = []
        
        # Look for tool configurations that might have commands
        if 'pytest' in content.lower():
            command_id = f"pytest_{int(time.time())}"
            commands.append(Command(
                id=command_id,
                number=self._generate_command_number(),
                name="run tests",
                description="Run pytest test suite",
                category="5.1",
                subcategory="Testing",
                executable="pytest",
                args=[],
                flags={"-v": "Verbose", "--cov": "Coverage"},
                examples=["pytest -v", "pytest --cov=src"],
                keywords=["pytest", "test", "testing"],
                source_file=str(file_path),
                complexity="low"
            ))
        
        return commands
    
    def _discover_notebook_commands(self, file_path: Path, content: str) -> List[Command]:
        """Discover commands from Jupyter notebooks."""
        commands = []
        
        try:
            notebook_data = json.loads(content)
            
            # Check if notebook has executable content
            if 'cells' in notebook_data and len(notebook_data['cells']) > 0:
                command_id = f"jupyter_{file_path.stem}_{int(time.time())}"
                commands.append(Command(
                    id=command_id,
                    number=self._generate_command_number(),
                    name=f"run notebook {file_path.name}",
                    description=f"Execute Jupyter notebook: {file_path.name}",
                    category="3.1",  # Data Processing -> Analysis
                    subcategory="Notebooks",
                    executable="jupyter",
                    args=["nbconvert", "--execute", str(file_path)],
                    flags={"--to": "Output format"},
                    examples=[f"jupyter nbconvert --execute {file_path.name}"],
                    keywords=["jupyter", "notebook", file_path.stem],
                    source_file=str(file_path),
                    complexity="low"
                ))
        except json.JSONDecodeError:
            pass
        
        return commands
    
    def _extract_python_flags(self, content: str) -> Dict[str, str]:
        """Extract command line flags from Python argparse code."""
        flags = {}
        
        # Look for add_argument calls
        flag_patterns = [
            r'add_argument\(["\']([^"\']+)["\'][^)]*help=["\']([^"\']+)["\']',
            r'add_argument\(["\']([^"\']+)["\'].*?help=["\']([^"\']+)["\']'
        ]
        
        for pattern in flag_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for flag, help_text in matches:
                flags[flag] = help_text
        
        return flags
    
    def _extract_click_flags(self, content: str) -> Dict[str, str]:
        """Extract command line flags from Click decorators."""
        flags = {}
        
        # Look for @click.option decorators
        option_pattern = r'@click\.option\(["\']([^"\']+)["\'][^)]*help=["\']([^"\']+)["\']'
        matches = re.findall(option_pattern, content)
        
        for flag, help_text in matches:
            flags[flag] = help_text
        
        return flags
    
    def _extract_shell_description(self, content: str) -> Optional[str]:
        """Extract description from shell script comments."""
        lines = content.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('#') and not line.startswith('#!'):
                description = line[1:].strip()
                if len(description) > 10:  # Meaningful description
                    return description
        
        return None
    
    def _extract_shell_flags(self, content: str) -> Dict[str, str]:
        """Extract flags from shell script getopts or manual parsing."""
        flags = {}
        
        # Look for getopts usage
        getopts_pattern = r'getopts\s+["\']([^"\']+)["\']'
        matches = re.findall(getopts_pattern, content)
        
        for opts in matches:
            for opt in opts:
                if opt != ':':
                    flags[f"-{opt}"] = f"Option {opt}"
        
        # Look for manual flag parsing
        flag_pattern = r'\$\{?1\}?\s*==?\s*["\'](-[^"\']+)["\']'
        matches = re.findall(flag_pattern, content)
        
        for flag in matches:
            flags[flag] = f"Command option {flag}"
        
        return flags
    
    def _generate_command_number(self) -> str:
        """Generate a unique command number."""
        # This is a simplified version - in practice, you'd want to check existing numbers
        category = random.choice(["1.1", "2.1", "3.1", "4.1", "5.1"])
        subcmd = random.randint(100, 999)
        return f"{category}.{subcmd}"
    
    def explore_directory(self, directory: Path, max_depth: int = 3) -> List[Command]:
        """Explore a directory for potential commands."""
        discovered_commands = []
        
        if max_depth <= 0 or not directory.exists():
            return discovered_commands
        
        try:
            for item in directory.iterdir():
                if item.is_file():
                    # Skip hidden files and common non-executable files
                    if (not item.name.startswith('.') and 
                        item.suffix not in ['.pyc', '.pyo', '.log', '.tmp']):
                        commands = self.discover_file_commands(item)
                        discovered_commands.extend(commands)
                
                elif item.is_dir() and not item.name.startswith('.'):
                    # Recursively explore subdirectories
                    subcommands = self.explore_directory(item, max_depth - 1)
                    discovered_commands.extend(subcommands)
        
        except PermissionError:
            self.logger.debug(f"Permission denied accessing {directory}")
        except Exception as e:
            self.logger.debug(f"Error exploring {directory}: {e}")
        
        return discovered_commands
    
    def learn_from_usage(self, command_id: str, success: bool, execution_time: float):
        """Learn from command usage to improve future discoveries."""
        if command_id not in self.curiosity_state.memory:
            self.curiosity_state.memory[command_id] = {
                "usage_count": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0
            }
        
        memory = self.curiosity_state.memory[command_id]
        memory["usage_count"] += 1
        
        # Update success rate
        old_rate = memory["success_rate"]
        memory["success_rate"] = (old_rate * (memory["usage_count"] - 1) + (1 if success else 0)) / memory["usage_count"]
        
        # Update execution time
        old_time = memory["avg_execution_time"]
        memory["avg_execution_time"] = (old_time * (memory["usage_count"] - 1) + execution_time) / memory["usage_count"]
        
        # Adjust learning rate based on success
        if success:
            self.curiosity_state.learning_rate = min(1.0, self.curiosity_state.learning_rate + 0.01)
        else:
            self.curiosity_state.learning_rate = max(0.01, self.curiosity_state.learning_rate - 0.005)
    
    def get_exploration_targets(self, base_path: Path) -> List[DiscoveryTarget]:
        """Generate exploration targets based on curiosity."""
        targets = []
        
        # Explore project directories
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    curiosity_score = self._calculate_directory_curiosity(item)
                    targets.append(DiscoveryTarget(
                        path=str(item),
                        type="directory",
                        priority=curiosity_score,
                        curiosity_score=curiosity_score
                    ))
        
        # Add external sources based on specializations
        if "ml" in self.specializations or "ai" in self.specializations:
            targets.extend([
                DiscoveryTarget(
                    path="https://github.com/huggingface",
                    type="repository",
                    priority=0.8,
                    curiosity_score=0.8
                ),
                DiscoveryTarget(
                    path="https://pytorch.org/tutorials",
                    type="documentation",
                    priority=0.6,
                    curiosity_score=0.6
                )
            ])
        
        if "neuroscience" in self.specializations:
            targets.extend([
                DiscoveryTarget(
                    path="https://github.com/NeuralEnsemble",
                    type="repository",
                    priority=0.9,
                    curiosity_score=0.9
                ),
                DiscoveryTarget(
                    path="https://nest-simulator.org",
                    type="documentation",
                    priority=0.7,
                    curiosity_score=0.7
                )
            ])
        
        return sorted(targets, key=lambda t: self.get_curiosity_score(t), reverse=True)
    
    def _calculate_directory_curiosity(self, directory: Path) -> float:
        """Calculate curiosity score for a directory."""
        base_score = 0.5
        
        # Boost for interesting directory names
        interesting_patterns = [
            "script", "tool", "bin", "cli", "command", "util", "demo", "example",
            "neural", "brain", "ai", "ml", "model", "train", "simulation"
        ]
        
        dir_name_lower = directory.name.lower()
        for pattern in interesting_patterns:
            if pattern in dir_name_lower:
                base_score += 0.2
        
        # Boost for directories with many files
        try:
            file_count = len([f for f in directory.iterdir() if f.is_file()])
            if file_count > 10:
                base_score += 0.3
            elif file_count > 5:
                base_score += 0.1
        except PermissionError:
            pass
        
        return min(1.0, base_score)

class ExponentialGrowthEngine:
    """Engine that manages exponential growth of command discovery."""
    
    def __init__(self, command_database: CommandDatabase):
        self.db = command_database
        self.agents = {}
        self.growth_rate = 1.5  # Commands grow by 50% each cycle
        self.discovery_cycles = 0
        self.total_discoveries = 0
        self.logger = logging.getLogger("growth_engine")
        
        # Initialize curiosity agents
        self._initialize_agents()
        
        # Start background discovery
        self.discovery_thread = None
        self.running = False
    
    def _initialize_agents(self):
        """Initialize curious agents with different specializations."""
        agent_specs = [
            ("scout", ["general", "scripts", "tools"]),
            ("neural_explorer", ["neuroscience", "brain", "ai", "ml"]),
            ("dev_tools_hunter", ["development", "testing", "deployment"]),
            ("cloud_seeker", ["cloud", "aws", "docker", "api"]),
            ("data_miner", ["data", "analysis", "jupyter", "python"]),
            ("system_inspector", ["system", "admin", "monitoring", "logs"])
        ]
        
        for agent_id, specializations in agent_specs:
            self.agents[agent_id] = CuriousAgent(agent_id, specializations)
            self.logger.info(f"Initialized curious agent: {agent_id} with specializations: {specializations}")
    
    def start_exponential_discovery(self, base_path: Path):
        """Start exponential command discovery process."""
        if self.running:
            return
        
        self.running = True
        self.discovery_thread = threading.Thread(
            target=self._discovery_loop,
            args=(base_path,),
            daemon=True
        )
        self.discovery_thread.start()
        self.logger.info("Started exponential discovery engine")
    
    def stop_discovery(self):
        """Stop the discovery process."""
        self.running = False
        if self.discovery_thread:
            self.discovery_thread.join(timeout=5)
        self.logger.info("Stopped discovery engine")
    
    def _discovery_loop(self, base_path: Path):
        """Main discovery loop that runs continuously."""
        while self.running:
            try:
                cycle_start = time.time()
                self.discovery_cycles += 1
                
                self.logger.info(f"Starting discovery cycle {self.discovery_cycles}")
                
                # Each agent explores in parallel
                with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                    futures = []
                    
                    for agent in self.agents.values():
                        future = executor.submit(self._agent_exploration_cycle, agent, base_path)
                        futures.append(future)
                    
                    # Collect results
                    cycle_discoveries = 0
                    for future in as_completed(futures):
                        try:
                            discoveries = future.result()
                            cycle_discoveries += discoveries
                        except Exception as e:
                            self.logger.error(f"Agent exploration failed: {e}")
                
                self.total_discoveries += cycle_discoveries
                cycle_time = time.time() - cycle_start
                
                self.logger.info(f"Cycle {self.discovery_cycles} completed: {cycle_discoveries} new commands in {cycle_time:.2f}s")
                
                # Exponential backoff - longer waits as we discover more
                wait_time = min(300, 30 * (1.1 ** self.discovery_cycles))  # Cap at 5 minutes
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Discovery cycle error: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _agent_exploration_cycle(self, agent: CuriousAgent, base_path: Path) -> int:
        """Single exploration cycle for one agent."""
        discoveries = 0
        
        try:
            # Get exploration targets based on agent's curiosity
            targets = agent.get_exploration_targets(base_path)
            
            # Explore top targets based on energy level
            energy_budget = agent.curiosity_state.exploration_energy
            explored_count = 0
            
            for target in targets[:5]:  # Explore top 5 targets
                if energy_budget <= 0:
                    break
                
                curiosity_score = agent.get_curiosity_score(target)
                if curiosity_score < 0.3:  # Skip low-interest targets
                    continue
                
                # Explore the target
                if target.type == "directory":
                    commands = agent.explore_directory(Path(target.path), max_depth=2)
                elif target.type == "file":
                    commands = agent.discover_file_commands(Path(target.path))
                else:
                    commands = []  # External sources not implemented yet
                
                # Store discovered commands
                for command in commands:
                    try:
                        self.db.store_command(command)
                        agent.discovered_commands.append(command)
                        discoveries += 1
                        self.logger.debug(f"Agent {agent.agent_id} discovered: {command.name}")
                    except Exception as e:
                        self.logger.debug(f"Failed to store command: {e}")
                
                # Update target exploration status
                target.last_explored = datetime.now().isoformat()
                target.discovery_count += len(commands)
                
                # Consume energy
                energy_cost = 10 * curiosity_score
                energy_budget -= energy_cost
                explored_count += 1
            
            # Update agent state
            agent.curiosity_state.exploration_energy = max(10, energy_budget)
            agent.curiosity_state.discovery_count += discoveries
            agent.curiosity_state.last_active = datetime.now().isoformat()
            
            # Regenerate energy based on successful discoveries
            if discoveries > 0:
                agent.curiosity_state.exploration_energy = min(100, 
                    agent.curiosity_state.exploration_energy + discoveries * 5)
            
            self.logger.debug(f"Agent {agent.agent_id} explored {explored_count} targets, discovered {discoveries} commands")
            
        except Exception as e:
            self.logger.error(f"Agent {agent.agent_id} exploration failed: {e}")
        
        return discoveries
    
    def get_growth_stats(self) -> Dict[str, Any]:
        """Get statistics about exponential growth."""
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = {
                "discoveries": agent.curiosity_state.discovery_count,
                "energy": agent.curiosity_state.exploration_energy,
                "learning_rate": agent.curiosity_state.learning_rate,
                "specializations": agent.specializations
            }
        
        return {
            "discovery_cycles": self.discovery_cycles,
            "total_discoveries": self.total_discoveries,
            "growth_rate": self.growth_rate,
            "agents": agent_stats,
            "running": self.running
        }
    
    def boost_curiosity(self, agent_id: str, boost_factor: float = 1.5):
        """Boost a specific agent's curiosity and energy."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.curiosity_state.exploration_energy = min(100, 
                agent.curiosity_state.exploration_energy * boost_factor)
            agent.curiosity_state.learning_rate = min(1.0,
                agent.curiosity_state.learning_rate * boost_factor)
            self.logger.info(f"Boosted curiosity for agent {agent_id}")

def test_curious_agents():
    """Test the curious agents system."""
    print("🧠 Testing Curious Agents System")
    print("=" * 50)
    
    # Initialize database and growth engine
    from command_database import CommandDatabase
    db = CommandDatabase()
    growth_engine = ExponentialGrowthEngine(db)
    
    # Show initial stats
    stats = growth_engine.get_growth_stats()
    print(f"Initialized {len(stats['agents'])} curious agents:")
    for agent_id, agent_data in stats['agents'].items():
        print(f"  • {agent_id}: {agent_data['specializations']}")
    
    # Test single agent exploration
    agent = growth_engine.agents['scout']
    base_path = Path("/Users/camdouglas/quark")
    
    print(f"\n🔍 Testing agent exploration on {base_path}")
    discoveries = growth_engine._agent_exploration_cycle(agent, base_path)
    print(f"Agent discovered {discoveries} new commands")
    
    # Show final stats
    final_stats = growth_engine.get_growth_stats()
    print(f"\nFinal discoveries: {final_stats['total_discoveries']}")
    
    db.close()

if __name__ == "__main__":
    test_curious_agents()
