#!/usr/bin/env python3
"""
SmallMind Command Database System
A sophisticated command discovery, organization, and execution system with natural language processing.
"""

import json
import sqlite3
import re
import os
import subprocess
import argparse
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import fnmatch

@dataclass
class Command:
    """Represents a single command with all its metadata."""
    id: str
    number: str  # Hierarchical number like "1.2.3"
    name: str
    description: str
    category: str
    subcategory: str
    executable: str
    args: List[str]
    flags: Dict[str, str]
    examples: List[str]
    keywords: List[str]
    requires_shell: bool = False
    requires_sudo: bool = False
    safe_mode: bool = True
    complexity: str = "low"  # low, medium, high
    source_file: str = ""
    last_updated: str = ""

@dataclass
class Category:
    """Represents a command category hierarchy."""
    number: str
    name: str
    description: str
    parent: Optional[str] = None
    subcategories: List[str] = None

class CommandDatabase:
    """Central command database with discovery and natural language processing."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Path.home() / ".smallmind" / "commands.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
        self.load_commands()
    
    def init_database(self):
        """Initialize SQLite database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS categories (
                number TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                parent TEXT,
                FOREIGN KEY (parent) REFERENCES categories (number)
            );
            
            CREATE TABLE IF NOT EXISTS commands (
                id TEXT PRIMARY KEY,
                number TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                subcategory TEXT,
                executable TEXT NOT NULL,
                args TEXT,  -- JSON array
                flags TEXT, -- JSON object
                examples TEXT, -- JSON array
                keywords TEXT, -- JSON array
                requires_shell BOOLEAN DEFAULT 0,
                requires_sudo BOOLEAN DEFAULT 0,
                safe_mode BOOLEAN DEFAULT 1,
                complexity TEXT DEFAULT 'low',
                source_file TEXT,
                last_updated TEXT,
                FOREIGN KEY (category) REFERENCES categories (number)
            );
            
            CREATE TABLE IF NOT EXISTS command_usage (
                command_id TEXT,
                timestamp TEXT,
                success BOOLEAN,
                user_input TEXT,
                execution_time REAL,
                FOREIGN KEY (command_id) REFERENCES commands (id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_keywords ON commands (keywords);
            CREATE INDEX IF NOT EXISTS idx_category ON commands (category, subcategory);
            CREATE INDEX IF NOT EXISTS idx_complexity ON commands (complexity);
        """)
        self.conn.commit()
    
    def load_commands(self):
        """Load and discover all available commands from the codebase."""
        print("🔍 Discovering commands...")
        
        # Initialize command categories
        self._init_categories()
        
        # Discover commands from different sources
        commands = []
        
        # 1. Neuro commands
        commands.extend(self._discover_neuro_commands())
        
        # 2. SmallMind CLI commands
        commands.extend(self._discover_smallmind_cli())
        
        # 3. Agent Hub commands
        commands.extend(self._discover_agent_hub())
        
        # 4. Script commands
        commands.extend(self._discover_script_commands())
        
        # 5. AWS and cloud commands
        commands.extend(self._discover_cloud_commands())
        
        # Store all commands
        for cmd in commands:
            self.store_command(cmd)
        
        print(f"✅ Loaded {len(commands)} commands into database")
    
    def _init_categories(self):
        """Initialize command categories hierarchy."""
        categories = [
            # Main categories
            Category("1", "Brain Development", "Neural development and brain simulation commands"),
            Category("2", "AI Models", "AI model management, training, and inference commands"),
            Category("3", "Data Processing", "Data analysis, transformation, and pipeline commands"),
            Category("4", "Cloud Computing", "Cloud deployment, optimization, and management commands"),
            Category("5", "Development Tools", "Development utilities, debugging, and maintenance commands"),
            Category("6", "Visualization", "Data visualization and plotting commands"),
            Category("7", "System Administration", "System management and configuration commands"),
            
            # Brain Development subcategories
            Category("1.1", "Simulation", "Brain physics and development simulation", "1"),
            Category("1.2", "Morphogenesis", "Tissue and cellular morphogenesis", "1"),
            Category("1.3", "Connectome", "Neural connectivity and network analysis", "1"),
            Category("1.4", "Plasticity", "Learning and synaptic plasticity", "1"),
            
            # AI Models subcategories
            Category("2.1", "Training", "Model training and fine-tuning", "2"),
            Category("2.2", "Inference", "Model inference and generation", "2"),
            Category("2.3", "Management", "Model downloading, storage, and versioning", "2"),
            Category("2.4", "Routing", "Intelligent model selection and routing", "2"),
            
            # Data Processing subcategories
            Category("3.1", "Analysis", "Data analysis and exploration", "3"),
            Category("3.2", "Transformation", "Data cleaning and transformation", "3"),
            Category("3.3", "Pipeline", "Data pipeline management", "3"),
            Category("3.4", "Validation", "Data validation and quality checks", "3"),
            
            # Cloud Computing subcategories
            Category("4.1", "AWS", "Amazon Web Services commands", "4"),
            Category("4.2", "Deployment", "Application deployment commands", "4"),
            Category("4.3", "Optimization", "Performance optimization commands", "4"),
            Category("4.4", "Monitoring", "System monitoring and logging", "4"),
            
            # Development Tools subcategories
            Category("5.1", "Testing", "Testing and validation tools", "5"),
            Category("5.2", "Debugging", "Debugging and troubleshooting tools", "5"),
            Category("5.3", "Setup", "Environment setup and configuration", "5"),
            Category("5.4", "Integration", "System integration and CI/CD", "5"),
        ]
        
        for cat in categories:
            self.conn.execute("""
                INSERT OR REPLACE INTO categories (number, name, description, parent)
                VALUES (?, ?, ?, ?)
            """, (cat.number, cat.name, cat.description, cat.parent))
        self.conn.commit()
    
    def _discover_neuro_commands(self) -> List[Command]:
        """Discover commands from the neuro module."""
        commands = []
        
        # Neuro CLI commands
        commands.extend([
            Command(
                id="neuro_scan",
                number="1.3.1",
                name="neuro scan",
                description="Scan files and build connectivity analysis",
                category="1.3",
                subcategory="Analysis",
                executable="python",
                args=["-m", "neuro.cli", "scan"],
                flags={},
                examples=["neuro scan"],
                keywords=["scan", "files", "connectivity", "analysis"],
                source_file="multi_model_training/neuro/cli.py"
            ),
            Command(
                id="neuro_analyze",
                number="1.3.2",
                name="neuro analyze",
                description="Analyze files and extract metadata",
                category="1.3",
                subcategory="Analysis",
                executable="python",
                args=["-m", "neuro.cli", "analyze"],
                flags={"--titles": "Extract file titles"},
                examples=["neuro analyze", "neuro analyze --titles"],
                keywords=["analyze", "metadata", "files", "titles"],
                source_file="multi_model_training/neuro/cli.py"
            ),
            Command(
                id="neuro_connectome",
                number="1.3.3",
                name="neuro connectome",
                description="Build connectome and detect communities",
                category="1.3",
                subcategory="Network",
                executable="python",
                args=["-m", "neuro.cli", "connectome"],
                flags={},
                examples=["neuro connectome"],
                keywords=["connectome", "communities", "network", "graph"],
                source_file="multi_model_training/neuro/cli.py"
            ),
            Command(
                id="neuro_compose",
                number="1.3.4",
                name="neuro compose",
                description="Compose neural networks and agents",
                category="1.3",
                subcategory="Composition",
                executable="python",
                args=["-m", "neuro.cli", "compose"],
                flags={"--learn": "Enable learning mode"},
                examples=["neuro compose", "neuro compose --learn"],
                keywords=["compose", "neural", "networks", "agents", "learning"],
                source_file="multi_model_training/neuro/cli.py"
            ),
            Command(
                id="neuro_organize",
                number="5.3.1",
                name="neuro organize",
                description="Organize files based on connectivity analysis",
                category="5.3",
                subcategory="Organization",
                executable="python",
                args=["-m", "neuro.cli", "organize"],
                flags={"--dry-run": "Show suggestions without moving files", "--execute": "Actually organize the files"},
                examples=["neuro organize --dry-run", "neuro organize --execute"],
                keywords=["organize", "files", "structure", "cleanup"],
                source_file="multi_model_training/neuro/cli.py"
            )
        ])
        
        return commands
    
    def _discover_smallmind_cli(self) -> List[Command]:
        """Discover SmallMind CLI commands."""
        commands = []
        
        # MoE CLI commands
        commands.extend([
            Command(
                id="moe_download",
                number="2.3.1",
                name="moe download",
                description="Download MoE models with optimizations",
                category="2.3",
                subcategory="Download",
                executable="python",
                args=["-m", "smallmind.cli.moe_cli", "download"],
                flags={"--force": "Force re-download"},
                examples=["moe download deepseek-v2", "moe download qwen1.5-moe mix-tao-moe"],
                keywords=["download", "models", "moe", "optimize"],
                source_file="src/smallmind/cli/moe_cli.py"
            ),
            Command(
                id="moe_list",
                number="2.3.2",
                name="moe list",
                description="List all available MoE models",
                category="2.3",
                subcategory="Management",
                executable="python",
                args=["-m", "smallmind.cli.moe_cli", "list"],
                flags={},
                examples=["moe list"],
                keywords=["list", "models", "moe", "available"],
                source_file="src/smallmind/cli/moe_cli.py"
            ),
            Command(
                id="moe_route",
                number="2.4.1",
                name="moe route",
                description="Test intelligent model routing",
                category="2.4",
                subcategory="Routing",
                executable="python",
                args=["-m", "smallmind.cli.moe_cli", "route"],
                flags={},
                examples=["moe route 'Simulate a neural circuit'"],
                keywords=["route", "intelligence", "model", "selection"],
                source_file="src/smallmind/cli/moe_cli.py"
            ),
            Command(
                id="moe_execute",
                number="2.2.1",
                name="moe execute",
                description="Execute text generation with routing",
                category="2.2",
                subcategory="Generation",
                executable="python",
                args=["-m", "smallmind.cli.moe_cli", "execute"],
                flags={"--max-tokens": "Maximum tokens to generate"},
                examples=["moe execute 'Explain synaptic plasticity' --max-tokens 512"],
                keywords=["execute", "generate", "text", "routing"],
                source_file="src/smallmind/cli/moe_cli.py"
            ),
            Command(
                id="moe_status",
                number="2.4.2",
                name="moe status",
                description="Show comprehensive system status",
                category="2.4",
                subcategory="Monitoring",
                executable="python",
                args=["-m", "smallmind.cli.moe_cli", "status"],
                flags={},
                examples=["moe status"],
                keywords=["status", "system", "health", "monitoring"],
                source_file="src/smallmind/cli/moe_cli.py"
            )
        ])
        
        # Neuroscience CLI commands
        commands.extend([
            Command(
                id="neuro_cli_list",
                number="1.1.1",
                name="neuroscience list",
                description="List available neuroscience experts",
                category="1.1",
                subcategory="Management",
                executable="python",
                args=["-m", "smallmind.cli.neuroscience_cli", "list"],
                flags={"--verbose": "Show detailed information"},
                examples=["neuroscience list", "neuroscience list --verbose"],
                keywords=["neuroscience", "experts", "list", "available"],
                source_file="src/smallmind/cli/neuroscience_cli.py"
            ),
            Command(
                id="neuro_cli_execute",
                number="1.1.2",
                name="neuroscience execute",
                description="Execute neuroscience simulation tasks",
                category="1.1",
                subcategory="Simulation",
                executable="python",
                args=["-m", "smallmind.cli.neuroscience_cli", "execute"],
                flags={
                    "--max-length": "Maximum length for text generation",
                    "--duration": "Simulation duration",
                    "--num-neurons": "Number of neurons",
                    "--connection-prob": "Connection probability",
                    "--dt": "Time step"
                },
                examples=[
                    "neuroscience execute spiking_networks 'Simulate 100 neurons' --duration 2000 --num-neurons 100"
                ],
                keywords=["neuroscience", "execute", "simulation", "spiking", "networks"],
                source_file="src/smallmind/cli/neuroscience_cli.py"
            )
        ])
        
        # Advanced CLI commands
        commands.extend([
            Command(
                id="advanced_optimize",
                number="4.3.1",
                name="advanced optimize",
                description="Run advanced model optimization",
                category="4.3",
                subcategory="Model",
                executable="python",
                args=["-m", "smallmind.cli.advanced_cli", "optimize"],
                flags={
                    "--model": "Model name to optimize",
                    "--config": "Configuration file path",
                    "--report": "Generate and display report"
                },
                examples=["advanced optimize --model 'meta-llama/Meta-Llama-3-8B-Instruct' --report"],
                keywords=["optimize", "advanced", "model", "performance"],
                source_file="src/smallmind/cli/advanced_cli.py"
            ),
            Command(
                id="advanced_neural",
                number="2.1.1",
                name="advanced neural",
                description="Run neural network optimization",
                category="2.1",
                subcategory="Optimization",
                executable="python",
                args=["-m", "smallmind.cli.advanced_cli", "neural"],
                flags={
                    "--epochs": "Number of training epochs",
                    "--optimize-hyperparams": "Enable hyperparameter optimization",
                    "--input-size": "Input feature size",
                    "--hidden-size": "Hidden layer size"
                },
                examples=["advanced neural --epochs 200 --optimize-hyperparams"],
                keywords=["neural", "optimization", "training", "hyperparameters"],
                source_file="src/smallmind/cli/advanced_cli.py"
            ),
            Command(
                id="advanced_simulate",
                number="1.1.3",
                name="advanced simulate",
                description="Run brain development simulation",
                category="1.1",
                subcategory="Development",
                executable="python",
                args=["-m", "smallmind.cli.advanced_cli", "simulate"],
                flags={
                    "--steps": "Number of simulation steps",
                    "--physics": "Physics engine (pybullet/mujoco)"
                },
                examples=["advanced simulate --steps 2000 --physics pybullet"],
                keywords=["simulate", "brain", "development", "physics"],
                source_file="src/smallmind/cli/advanced_cli.py"
            )
        ])
        
        return commands
    
    def _discover_agent_hub(self) -> List[Command]:
        """Discover Agent Hub commands."""
        commands = []
        
        # Agent Hub CLI commands
        commands.extend([
            Command(
                id="smctl_list",
                number="2.4.3",
                name="smctl list",
                description="List available agent models",
                category="2.4",
                subcategory="Registry",
                executable="smctl",
                args=["list"],
                flags={},
                examples=["smctl list"],
                keywords=["agents", "models", "list", "registry"],
                source_file="models/agent_hub/cli.py"
            ),
            Command(
                id="smctl_ask",
                number="2.2.2",
                name="smctl ask",
                description="Ask a question with automatic model routing",
                category="2.2",
                subcategory="Query",
                executable="smctl",
                args=["ask"],
                flags={
                    "--tools": "Comma-separated tools list",
                    "--allow-shell": "Allow shell access",
                    "--sudo-ok": "Allow sudo operations",
                    "--show-run-dir": "Show run directory"
                },
                examples=[
                    "smctl ask 'How do I install Python packages?'",
                    "smctl ask 'Install numpy using pip' --tools shell,python --allow-shell"
                ],
                keywords=["ask", "query", "routing", "intelligent"],
                requires_shell=False,
                complexity="medium",
                source_file="models/agent_hub/cli.py"
            ),
            Command(
                id="smctl_describe",
                number="2.4.4",
                name="smctl describe",
                description="Describe a specific agent model",
                category="2.4",
                subcategory="Information",
                executable="smctl",
                args=["describe"],
                flags={},
                examples=["smctl describe hf.qwen2.5"],
                keywords=["describe", "model", "information", "details"],
                source_file="models/agent_hub/cli.py"
            ),
            Command(
                id="smctl_run",
                number="2.2.3",
                name="smctl run",
                description="Run a specific agent model",
                category="2.2",
                subcategory="Execution",
                executable="smctl",
                args=["run"],
                flags={
                    "--model": "Model ID to run",
                    "--allow-shell": "Allow shell access",
                    "--sudo-ok": "Allow sudo operations"
                },
                examples=["smctl run --model hf.qwen2.5 'Explain quantum computing'"],
                keywords=["run", "execute", "model", "specific"],
                source_file="models/agent_hub/cli.py"
            ),
            Command(
                id="smctl_plan",
                number="2.4.5",
                name="smctl plan",
                description="Plan a complex task breakdown",
                category="2.4",
                subcategory="Planning",
                executable="smctl",
                args=["plan"],
                flags={},
                examples=["smctl plan 'Build a web application with authentication'"],
                keywords=["plan", "task", "breakdown", "complex"],
                complexity="high",
                source_file="models/agent_hub/cli.py"
            )
        ])
        
        return commands
    
    def _discover_script_commands(self) -> List[Command]:
        """Discover commands from scripts directory."""
        commands = []
        
        # AWS and cloud scripts
        commands.extend([
            Command(
                id="aws_config",
                number="4.1.1",
                name="aws config",
                description="Check and configure AWS setup",
                category="4.1",
                subcategory="Configuration",
                executable="python",
                args=["scripts/aws_config.py"],
                flags={},
                examples=["python scripts/aws_config.py"],
                keywords=["aws", "config", "setup", "check"],
                source_file="scripts/aws_config.py"
            ),
            Command(
                id="aws_deploy",
                number="4.2.1",
                name="aws deploy",
                description="Deploy SmallMind to AWS",
                category="4.2",
                subcategory="AWS",
                executable="python",
                args=["scripts/aws_deploy.py"],
                flags={"--key-name": "AWS key pair name"},
                examples=["python scripts/aws_deploy.py --key-name YOUR_KEY_NAME"],
                keywords=["aws", "deploy", "cloud", "ec2"],
                requires_shell=True,
                complexity="high",
                source_file="scripts/aws_deploy.py"
            ),
            Command(
                id="aws_optimize",
                number="4.3.2",
                name="aws optimize",
                description="Optimize AWS performance",
                category="4.3",
                subcategory="Performance",
                executable="python",
                args=["scripts/aws_performance_optimization.py"],
                flags={},
                examples=["python scripts/aws_performance_optimization.py"],
                keywords=["aws", "optimize", "performance", "tuning"],
                requires_shell=True,
                complexity="medium",
                source_file="scripts/aws_performance_optimization.py"
            )
        ])
        
        # Setup and configuration scripts
        commands.extend([
            Command(
                id="setup_complete",
                number="5.3.2",
                name="setup complete",
                description="Complete SmallMind environment setup",
                category="5.3",
                subcategory="Installation",
                executable="python",
                args=["scripts/setup_smallmind_complete.py"],
                flags={},
                examples=["python scripts/setup_smallmind_complete.py"],
                keywords=["setup", "install", "environment", "complete"],
                requires_shell=True,
                complexity="medium",
                source_file="scripts/setup_smallmind_complete.py"
            ),
            Command(
                id="startup_learning",
                number="2.1.2",
                name="startup learning",
                description="Start continuous learning system",
                category="2.1",
                subcategory="Training",
                executable="bash",
                args=["scripts/startup_continuous_learning.sh"],
                flags={
                    "start": "Start the learning system",
                    "stop": "Stop the learning system",
                    "status": "Show current status"
                },
                examples=[
                    "bash scripts/startup_continuous_learning.sh start",
                    "bash scripts/startup_continuous_learning.sh status"
                ],
                keywords=["startup", "learning", "continuous", "training"],
                requires_shell=True,
                complexity="medium",
                source_file="scripts/startup_continuous_learning.sh"
            )
        ])
        
        return commands
    
    def _discover_cloud_commands(self) -> List[Command]:
        """Discover cloud computing and deployment commands."""
        commands = []
        
        commands.extend([
            Command(
                id="cursor_status",
                number="5.4.1",
                name="cursor status",
                description="Show Cursor context status",
                category="5.4",
                subcategory="Context",
                executable="bash",
                args=["scripts/cursor_context_manager.sh", "cursor-status"],
                flags={},
                examples=["cursor-status"],
                keywords=["cursor", "status", "context", "chat"],
                source_file="scripts/cursor_context_manager.sh"
            ),
            Command(
                id="cursor_clear",
                number="5.4.2",
                name="cursor clear",
                description="Clear all Cursor chat history",
                category="5.4",
                subcategory="Maintenance",
                executable="bash",
                args=["scripts/cursor_context_manager.sh", "cursor-clear"],
                flags={},
                examples=["cursor-clear"],
                keywords=["cursor", "clear", "history", "reset"],
                requires_shell=True,
                safe_mode=False,
                source_file="scripts/cursor_context_manager.sh"
            ),
            Command(
                id="universal_trainer",
                number="2.1.3",
                name="universal trainer",
                description="Universal model training across platforms",
                category="2.1",
                subcategory="Training",
                executable="python",
                args=["scripts/universal_model_trainer.py"],
                flags={
                    "train_all": "Train all available models",
                    "--model-type": "Specific model type to train"
                },
                examples=["python scripts/universal_model_trainer.py train_all"],
                keywords=["universal", "training", "models", "platforms"],
                complexity="high",
                source_file="scripts/universal_model_trainer.py"
            )
        ])
        
        return commands
    
    def store_command(self, command: Command):
        """Store a command in the database."""
        command.last_updated = datetime.now().isoformat()
        
        self.conn.execute("""
            INSERT OR REPLACE INTO commands 
            (id, number, name, description, category, subcategory, executable, args, flags, 
             examples, keywords, requires_shell, requires_sudo, safe_mode, complexity, 
             source_file, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            command.id, command.number, command.name, command.description,
            command.category, command.subcategory, command.executable,
            json.dumps(command.args), json.dumps(command.flags),
            json.dumps(command.examples), json.dumps(command.keywords),
            command.requires_shell, command.requires_sudo, command.safe_mode,
            command.complexity, command.source_file, command.last_updated
        ))
        self.conn.commit()
    
    def get_command(self, command_id: str) -> Optional[Command]:
        """Get a command by ID."""
        row = self.conn.execute(
            "SELECT * FROM commands WHERE id = ?", (command_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return Command(
            id=row['id'],
            number=row['number'],
            name=row['name'],
            description=row['description'],
            category=row['category'],
            subcategory=row['subcategory'],
            executable=row['executable'],
            args=json.loads(row['args']),
            flags=json.loads(row['flags']),
            examples=json.loads(row['examples']),
            keywords=json.loads(row['keywords']),
            requires_shell=bool(row['requires_shell']),
            requires_sudo=bool(row['requires_sudo']),
            safe_mode=bool(row['safe_mode']),
            complexity=row['complexity'],
            source_file=row['source_file'],
            last_updated=row['last_updated']
        )
    
    def search_commands(self, query: str, category: str = None) -> List[Command]:
        """Search commands by natural language query."""
        if not query.strip():
            # Return all commands if query is empty
            if category:
                sql = "SELECT * FROM commands WHERE category = ? ORDER BY number"
                params = [category]
            else:
                sql = "SELECT * FROM commands ORDER BY number"
                params = []
        else:
            # Simple keyword matching for now - will enhance with NLP
            words = query.lower().split()
            
            # Build WHERE clauses for each word
            where_clauses = []
            params = []
            
            for word in words:
                where_clauses.append("""(
                    LOWER(name) LIKE ? OR 
                    LOWER(description) LIKE ? OR
                    LOWER(keywords) LIKE ?
                )""")
                params.extend([f"%{word}%", f"%{word}%", f"%{word}%"])
            
            sql = f"""
                SELECT * FROM commands 
                WHERE {" OR ".join(where_clauses)}
            """
            
            if category:
                sql += " AND category = ?"
                params.append(category)
            
            sql += " ORDER BY complexity, number"
        
        rows = self.conn.execute(sql, params).fetchall()
        
        commands = []
        for row in rows:
            commands.append(Command(
                id=row['id'],
                number=row['number'],
                name=row['name'],
                description=row['description'],
                category=row['category'],
                subcategory=row['subcategory'],
                executable=row['executable'],
                args=json.loads(row['args']),
                flags=json.loads(row['flags']),
                examples=json.loads(row['examples']),
                keywords=json.loads(row['keywords']),
                requires_shell=bool(row['requires_shell']),
                requires_sudo=bool(row['requires_sudo']),
                safe_mode=bool(row['safe_mode']),
                complexity=row['complexity'],
                source_file=row['source_file'],
                last_updated=row['last_updated']
            ))
        
        return commands
    
    def get_commands_by_category(self, category: str) -> List[Command]:
        """Get all commands in a category."""
        rows = self.conn.execute("""
            SELECT * FROM commands 
            WHERE category = ? 
            ORDER BY number
        """, (category,)).fetchall()
        
        commands = []
        for row in rows:
            commands.append(Command(
                id=row['id'],
                number=row['number'],
                name=row['name'],
                description=row['description'],
                category=row['category'],
                subcategory=row['subcategory'],
                executable=row['executable'],
                args=json.loads(row['args']),
                flags=json.loads(row['flags']),
                examples=json.loads(row['examples']),
                keywords=json.loads(row['keywords']),
                requires_shell=bool(row['requires_shell']),
                requires_sudo=bool(row['requires_sudo']),
                safe_mode=bool(row['safe_mode']),
                complexity=row['complexity'],
                source_file=row['source_file'],
                last_updated=row['last_updated']
            ))
        
        return commands
    
    def get_categories(self) -> List[Category]:
        """Get all categories."""
        rows = self.conn.execute("""
            SELECT * FROM categories 
            ORDER BY number
        """).fetchall()
        
        categories = []
        for row in rows:
            categories.append(Category(
                number=row['number'],
                name=row['name'],
                description=row['description'],
                parent=row['parent']
            ))
        
        return categories
    
    def log_usage(self, command_id: str, user_input: str, success: bool, execution_time: float):
        """Log command usage for analytics."""
        self.conn.execute("""
            INSERT INTO command_usage 
            (command_id, timestamp, success, user_input, execution_time)
            VALUES (?, ?, ?, ?, ?)
        """, (command_id, datetime.now().isoformat(), success, user_input, execution_time))
        self.conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        # Command counts
        stats['total_commands'] = self.conn.execute("SELECT COUNT(*) FROM commands").fetchone()[0]
        stats['total_categories'] = self.conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        
        # Category breakdown
        category_counts = self.conn.execute("""
            SELECT c.name, COUNT(cmd.id) as count
            FROM categories c
            LEFT JOIN commands cmd ON c.number = cmd.category
            WHERE c.parent IS NULL
            GROUP BY c.number, c.name
            ORDER BY count DESC
        """).fetchall()
        
        stats['categories'] = [dict(row) for row in category_counts]
        
        # Complexity breakdown
        complexity_counts = self.conn.execute("""
            SELECT complexity, COUNT(*) as count
            FROM commands
            GROUP BY complexity
            ORDER BY count DESC
        """).fetchall()
        
        stats['complexity'] = [dict(row) for row in complexity_counts]
        
        return stats
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    # Initialize and test the database
    db = CommandDatabase()
    
    print("\n📊 Command Database Statistics:")
    stats = db.get_stats()
    print(f"Total Commands: {stats['total_commands']}")
    print(f"Total Categories: {stats['total_categories']}")
    
    print("\n📁 Categories:")
    for cat in stats['categories']:
        print(f"  {cat['name']}: {cat['count']} commands")
    
    print("\n🎯 Complexity Distribution:")
    for comp in stats['complexity']:
        print(f"  {comp['complexity']}: {comp['count']} commands")
    
    # Test search
    print("\n🔍 Search Results for 'neural':")
    results = db.search_commands("neural simulation")
    for cmd in results[:5]:
        print(f"  {cmd.number} - {cmd.name}: {cmd.description}")
    
    db.close()
