#!/usr/bin/env python3
"""
Organization Agent - Automatic File Structure Management
Purpose: Maintain clean directory structure by auto-organizing files into appropriate directories
Inputs: File system events, new file creations, directory changes
Outputs: Organized file structure, movement logs, structure validation
Seeds: N/A (organizational logic)
Dependencies: pathlib, shutil, typing, json, datetime
"""

import os
import shutil
import json
import re
import ast
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
import logging
from collections import defaultdict, Counter

class OrganizationAgent:
    """
    Automatic file organization agent that maintains clean directory structure
    in the quark workspace by routing files to appropriate locations.
    Integrates with connectome agent for intelligent clustering and categorization.
    """
    
    def __init__(self, root_path: str = "/Users/camdouglas/quark"):
        self.root_path = Path(root_path)
        self.setup_logging()
        self.movement_log = []
        
        # Connectome integration
        self.connectome_path = self.root_path / "brain_modules" / "connectome"
        self.knowledge_graph = nx.DiGraph()
        self.semantic_clusters = {}
        self.relationship_weights = defaultdict(float)
        
        # Code analysis for semantic clustering
        self.import_graph = nx.DiGraph()
        self.function_dependencies = defaultdict(set)
        self.class_hierarchies = defaultdict(set)
        
        # Define file classification rules
        self.file_patterns = {
            # Core brain simulation files
            "brain_modules": {
                "patterns": [
                    r".*brain.*\.py$", r".*neural.*\.py$", r".*consciousness.*\.py$",
                    r".*agent.*\.py$", r".*cognitive.*\.py$", r".*synapse.*\.py$"
                ],
                "exclude_if_contains": ["test", "demo", "example"]
            },
            
            # Expert domain files
            "expert_domains": {
                "patterns": [
                    r".*neuroscience.*\.py$", r".*ml_.*\.py$", r".*data_eng.*\.py$",
                    r".*ethics.*\.py$", r".*safety.*\.py$", r".*architecture.*\.py$"
                ],
                "exclude_if_contains": ["test", "demo"]
            },
            
            # Configuration files
            "configs": {
                "patterns": [
                    r".*config.*\.(json|yaml|yml|toml)$", r".*\.env$",
                    r"requirements.*\.txt$", r"pyproject\.toml$", r".*\.cfg$"
                ]
            },
            
            # Documentation
            "docs": {
                "patterns": [
                    r".*README.*\.md$", r".*GUIDE.*\.md$", r".*SUMMARY.*\.md$",
                    r".*\.md$", r".*\.rst$", r".*\.txt$"
                ],
                "exclude_if_contains": ["test", "temp"]
            },
            
            # Research and experiments
            "research_lab": {
                "patterns": [
                    r".*experiment.*\.py$", r".*research.*\.py$", r".*analysis.*\.py$",
                    r".*\.ipynb$", r".*study.*\.py$"
                ]
            },
            
            # Applications and demos
            "applications": {
                "patterns": [
                    r".*demo.*\.py$", r".*example.*\.py$", r".*quickstart.*\.py$",
                    r".*app.*\.py$", r".*application.*\.py$"
                ]
            },
            
            # Tests
            "tests": {
                "patterns": [
                    r"test_.*\.py$", r".*_test\.py$", r".*test.*\.py$",
                    r".*_tests\.py$", r".*\.test\..*$"
                ]
            },
            
            # Data files
            "data": {
                "patterns": [
                    r".*\.csv$", r".*\.json$", r".*\.jsonl$", r".*\.pkl$",
                    r".*\.npy$", r".*\.npz$", r".*\.h5$", r".*\.hdf5$"
                ],
                "exclude_if_contains": ["config", "manifest"]
            },
            
            # Models and weights
            "models": {
                "patterns": [
                    r".*\.gguf$", r".*\.pth$", r".*\.pt$", r".*\.onnx$",
                    r".*\.safetensors$", r".*\.bin$", r".*model.*\.py$"
                ]
            },
            
            # Scripts and utilities
            "tools_utilities/scripts": {
                "patterns": [
                    r".*script.*\.py$", r".*util.*\.py$", r".*tool.*\.py$",
                    r".*helper.*\.py$", r".*\.sh$", r".*\.bash$"
                ]
            },
            
            # Training related
            "training": {
                "patterns": [
                    r".*train.*\.py$", r".*training.*\.py$", r".*trainer.*\.py$",
                    r".*fine.*tune.*\.py$"
                ]
            },
            
            # Results and outputs
            "results": {
                "patterns": [
                    r".*result.*\.(json|csv|png|jpg|html)$", r".*output.*\.(json|csv|png|jpg|html)$",
                    r".*report.*\.(json|csv|png|jpg|html)$"
                ]
            },
            
            # Deployment
            "deployment": {
                "patterns": [
                    r".*deploy.*\.py$", r".*docker.*$", r".*\.dockerfile$",
                    r".*compose.*\.ya?ml$", r".*k8s.*\.ya?ml$"
                ]
            }
        }
        
        # Files that should stay in root
        self.root_files = {
            "__init__.py", "setup.py", "main.py", "README.md", 
            "requirements.txt", "pyproject.toml", ".gitignore",
            ".cursorrules", "LICENSE", "MANIFEST.in"
        }
        
        # Temporary files to auto-clean
        self.temp_patterns = [
            r".*\.tmp$", r".*\.temp$", r".*~$", r".*\.bak$",
            r".*\.log$", r".*\.cache$", r"\.DS_Store$"
        ]

    def setup_logging(self):
        """Setup logging for organization activities"""
        log_dir = self.root_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - OrganizationAgent - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "organization_agent.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def classify_file(self, file_path: Path) -> Optional[str]:
        """
        Classify a file and determine its appropriate directory
        Uses connectome-based semantic classification for enhanced organization
        
        Args:
            file_path: Path to the file to classify
            
        Returns:
            Target directory path or None if file should stay in root
        """
        file_name = file_path.name
        file_str = str(file_path)
        
        # Keep root files in root
        if file_name in self.root_files:
            return None
            
        # Check temporary files for deletion
        for pattern in self.temp_patterns:
            if re.match(pattern, file_name, re.IGNORECASE):
                return "DELETE"
        
        # Try connectome-based classification first for Python files
        if file_path.suffix == '.py':
            connectome_classification = self.classify_by_connectome_relationships(file_path)
            if connectome_classification:
                return connectome_classification
        
        # Classify based on patterns
        for target_dir, rules in self.file_patterns.items():
            patterns = rules["patterns"]
            exclude_patterns = rules.get("exclude_if_contains", [])
            
            # Check if file matches any pattern
            for pattern in patterns:
                if re.match(pattern, file_name, re.IGNORECASE):
                    # Check exclusions
                    excluded = False
                    for exclude in exclude_patterns:
                        if exclude.lower() in file_str.lower():
                            excluded = True
                            break
                    
                    if not excluded:
                        return target_dir
        
        # Default fallback based on extension
        extension = file_path.suffix.lower()
        if extension in ['.py']:
            return "tools_utilities/scripts"  # Unclassified Python files
        elif extension in ['.md', '.txt']:
            return "docs"
        elif extension in ['.json', '.csv']:
            return "data"
        elif extension in ['.yaml', '.yml']:
            return "configs"
            
        return None  # Stay in root if can't classify

    def organize_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """
        Organize a single file into appropriate directory
        
        Args:
            file_path: Path to file to organize
            dry_run: If True, only log what would be done
            
        Returns:
            True if file was moved or would be moved
        """
        if not file_path.exists() or file_path.is_dir():
            return False
            
        target_dir = self.classify_file(file_path)
        
        if target_dir == "DELETE":
            if dry_run:
                self.logger.info(f"[DRY RUN] Would delete temporary file: {file_path}")
            else:
                file_path.unlink()
                self.logger.info(f"Deleted temporary file: {file_path}")
            return True
            
        if target_dir is None:
            return False  # File should stay in current location
            
        target_path = self.root_path / target_dir
        target_file = target_path / file_path.name
        
        # Create target directory if it doesn't exist
        if not dry_run:
            target_path.mkdir(parents=True, exist_ok=True)
            
        # Check if target file already exists
        if target_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_file = target_path / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
        if dry_run:
            self.logger.info(f"[DRY RUN] Would move: {file_path} → {target_file}")
        else:
            shutil.move(str(file_path), str(target_file))
            self.movement_log.append({
                "timestamp": datetime.now().isoformat(),
                "source": str(file_path),
                "target": str(target_file),
                "action": "move"
            })
            self.logger.info(f"Moved: {file_path.name} → {target_dir}/")
            
        return True

    def scan_and_organize(self, target_path: Optional[Path] = None, dry_run: bool = False) -> Dict:
        """
        Scan directory and organize all misplaced files
        
        Args:
            target_path: Path to scan (defaults to root)
            dry_run: If True, only report what would be done
            
        Returns:
            Summary of organization actions
        """
        if target_path is None:
            target_path = self.root_path
            
        summary = {
            "scanned": 0,
            "moved": 0,
            "deleted": 0,
            "errors": 0,
            "actions": []
        }
        
        self.logger.info(f"Starting organization scan of: {target_path}")
        
        # Only scan files in the immediate directory (not subdirectories)
        for item in target_path.iterdir():
            if item.is_file():
                summary["scanned"] += 1
                try:
                    target_dir = self.classify_file(item)
                    
                    if target_dir == "DELETE":
                        summary["deleted"] += 1
                        summary["actions"].append(f"DELETE: {item.name}")
                    elif target_dir is not None:
                        summary["moved"] += 1
                        summary["actions"].append(f"MOVE: {item.name} → {target_dir}/")
                        
                    if self.organize_file(item, dry_run=dry_run):
                        if target_dir != "DELETE":
                            summary["moved"] += 1
                        else:
                            summary["deleted"] += 1
                            
                except Exception as e:
                    summary["errors"] += 1
                    self.logger.error(f"Error organizing {item}: {e}")
                    
        self.logger.info(f"Organization complete. Moved: {summary['moved']}, Deleted: {summary['deleted']}, Errors: {summary['errors']}")
        return summary

    def create_organization_rule(self) -> str:
        """
        Generate a workspace rule for automatic organization
        
        Returns:
            Rule string to be added to workspace rules
        """
        rule = """
# AUTOMATIC FILE ORGANIZATION RULE
- **File Organization Agent**: Always active in quark root directory
  - **Trigger**: Any file creation, modification, or directory change in root
  - **Action**: Automatically classify and move files to appropriate subdirectories
  - **Classifications**:
    - Brain simulation files → `brain_modules/`
    - Expert domain code → `expert_domains/`
    - Configuration files → `configs/`
    - Documentation → `docs/`
    - Research/experiments → `research_lab/`
    - Applications/demos → `applications/`
    - Tests → `tests/`
    - Data files → `data/`
    - Models → `models/`
    - Scripts/utilities → `tools_utilities/scripts/`
    - Training code → `training/`
    - Results/outputs → `results/`
    - Deployment → `deployment/`
  - **Root Protection**: Core files (README.md, setup.py, etc.) remain in root
  - **Auto-Clean**: Temporary files (.tmp, .bak, .log) automatically removed
  - **Logging**: All organization actions logged to `logs/organization_agent.log`
"""
        return rule

    def load_connectome_metadata(self) -> Dict[str, Any]:
        """
        Load connectome metadata for intelligent categorization
        
        Returns:
            Connectome metadata and module relationships
        """
        metadata = {}
        try:
            # Load connectome exports
            exports_path = self.connectome_path / "exports"
            if exports_path.exists():
                # Load connectome graph
                connectome_file = exports_path / "connectome.graphml"
                if connectome_file.exists():
                    self.knowledge_graph = nx.read_graphml(str(connectome_file))
                    
                # Load module manifests
                for manifest_file in exports_path.glob("*_manifest.json"):
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                        module_id = manifest.get("module_id")
                        if module_id:
                            metadata[module_id] = manifest
                            
                # Load build summary
                summary_file = exports_path / "build_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        metadata["build_summary"] = json.load(f)
                        
        except Exception as e:
            self.logger.error(f"Error loading connectome metadata: {e}")
            
        return metadata

    def analyze_code_semantics(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze Python file for semantic content and relationships
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Semantic analysis results
        """
        if not file_path.suffix == '.py':
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            analysis = {
                "imports": [],
                "classes": [],
                "functions": [],
                "brain_keywords": [],
                "neural_concepts": [],
                "dependencies": [],
                "complexity_score": 0
            }
            
            # Brain/neural concept keywords
            neural_keywords = {
                'neural', 'brain', 'neuron', 'synapse', 'connectome', 'plasticity',
                'cortex', 'hippocampus', 'thalamus', 'ganglia', 'cognitive', 'consciousness',
                'memory', 'learning', 'attention', 'perception', 'motor', 'sensory',
                'dopamine', 'serotonin', 'acetylcholine', 'gaba', 'glutamate',
                'spike', 'firing', 'potential', 'axon', 'dendrite', 'soma'
            }
            
            # Extract semantic information
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                        analysis["dependencies"].append(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis["imports"].append(node.module)
                        analysis["dependencies"].append(node.module)
                        
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append(node.name)
                    analysis["complexity_score"] += 2
                    
                elif isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(node.name)
                    analysis["complexity_score"] += 1
                    
            # Identify neural/brain concepts in content
            content_lower = content.lower()
            for keyword in neural_keywords:
                if keyword in content_lower:
                    analysis["brain_keywords"].append(keyword)
                    analysis["neural_concepts"].append(keyword)
                    
            # Calculate semantic similarity scores
            analysis["brain_relevance"] = len(analysis["brain_keywords"]) / max(1, len(content.split()))
            analysis["module_coupling"] = len(analysis["dependencies"])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing code semantics for {file_path}: {e}")
            return {}

    def build_semantic_clusters(self) -> Dict[str, List[str]]:
        """
        Build semantic clusters based on code analysis and connectome data
        
        Returns:
            Dictionary mapping cluster names to file lists
        """
        clusters = defaultdict(list)
        
        try:
            # Analyze all Python files in the workspace
            file_analyses = {}
            for py_file in self.root_path.rglob("*.py"):
                if py_file.is_file() and not str(py_file).startswith(str(self.root_path / "venv")):
                    analysis = self.analyze_code_semantics(py_file)
                    if analysis:
                        file_analyses[str(py_file)] = analysis
                        
            # Load connectome metadata for brain module relationships
            connectome_meta = self.load_connectome_metadata()
            
            # Cluster files by semantic similarity
            for file_path, analysis in file_analyses.items():
                file_name = Path(file_path).name
                
                # Brain module clustering
                if analysis.get("brain_relevance", 0) > 0.001:  # Has brain-related content
                    brain_concepts = analysis.get("brain_keywords", [])
                    
                    # Map to specific brain modules based on keywords
                    if any(kw in brain_concepts for kw in ['cortex', 'prefrontal', 'executive']):
                        clusters['brain_modules/prefrontal_cortex'].append(file_path)
                    elif any(kw in brain_concepts for kw in ['hippocampus', 'memory', 'learning']):
                        clusters['brain_modules/hippocampus'].append(file_path)
                    elif any(kw in brain_concepts for kw in ['thalamus', 'relay', 'routing']):
                        clusters['brain_modules/thalamus'].append(file_path)
                    elif any(kw in brain_concepts for kw in ['ganglia', 'basal', 'gating']):
                        clusters['brain_modules/basal_ganglia'].append(file_path)
                    elif any(kw in brain_concepts for kw in ['default', 'mode', 'network', 'dmn']):
                        clusters['brain_modules/default_mode_network'].append(file_path)
                    elif any(kw in brain_concepts for kw in ['salience', 'attention']):
                        clusters['brain_modules/salience_networks'].append(file_path)
                    elif any(kw in brain_concepts for kw in ['working', 'memory', 'buffer']):
                        clusters['brain_modules/working_memory'].append(file_path)
                    elif any(kw in brain_concepts for kw in ['conscious', 'agent', 'awareness']):
                        clusters['brain_modules/conscious_agent'].append(file_path)
                    else:
                        clusters['brain_modules'].append(file_path)
                        
                # Expert domain clustering based on imports and function names
                dependencies = analysis.get("dependencies", [])
                functions = analysis.get("functions", [])
                
                if any('torch' in dep or 'tensorflow' in dep or 'sklearn' in dep for dep in dependencies):
                    clusters['expert_domains/machine_learning'].append(file_path)
                elif any('scipy' in dep or 'numpy' in dep or 'pandas' in dep for dep in dependencies):
                    clusters['expert_domains/computational_neuroscience'].append(file_path)
                elif any('test' in func.lower() for func in functions):
                    clusters['tests'].append(file_path)
                elif any('train' in func.lower() for func in functions):
                    clusters['training'].append(file_path)
                elif analysis.get("complexity_score", 0) > 10:  # High complexity
                    clusters['expert_domains/systems_architecture'].append(file_path)
                    
            # Remove empty clusters and deduplicate
            final_clusters = {}
            for cluster_name, file_list in clusters.items():
                if file_list:
                    final_clusters[cluster_name] = list(set(file_list))
                    
            return final_clusters
            
        except Exception as e:
            self.logger.error(f"Error building semantic clusters: {e}")
            return {}

    def classify_by_connectome_relationships(self, file_path: Path) -> Optional[str]:
        """
        Classify file based on connectome module relationships and semantic analysis
        
        Args:
            file_path: Path to file to classify
            
        Returns:
            Target directory based on connectome relationships
        """
        try:
            # First run semantic analysis
            analysis = self.analyze_code_semantics(file_path)
            
            # Check if file is brain-related
            brain_relevance = analysis.get("brain_relevance", 0)
            brain_keywords = analysis.get("brain_keywords", [])
            
            if brain_relevance > 0.001:  # Has significant brain content
                # Use connectome metadata to determine best module placement
                connectome_meta = self.load_connectome_metadata()
                
                # Score potential modules based on keyword matching
                module_scores = defaultdict(float)
                
                for keyword in brain_keywords:
                    if keyword in ['cortex', 'prefrontal', 'executive', 'planning']:
                        module_scores['brain_modules/prefrontal_cortex'] += 1.0
                    elif keyword in ['hippocampus', 'memory', 'episodic', 'consolidation']:
                        module_scores['brain_modules/hippocampus'] += 1.0
                    elif keyword in ['thalamus', 'relay', 'routing', 'gating']:
                        module_scores['brain_modules/thalamus'] += 1.0
                    elif keyword in ['basal', 'ganglia', 'gating', 'selection']:
                        module_scores['brain_modules/basal_ganglia'] += 1.0
                    elif keyword in ['default', 'mode', 'network', 'introspection']:
                        module_scores['brain_modules/default_mode_network'] += 1.0
                    elif keyword in ['salience', 'attention', 'focus']:
                        module_scores['brain_modules/salience_networks'] += 1.0
                    elif keyword in ['working', 'memory', 'buffer', 'maintenance']:
                        module_scores['brain_modules/working_memory'] += 1.0
                    elif keyword in ['conscious', 'agent', 'awareness', 'global']:
                        module_scores['brain_modules/conscious_agent'] += 1.0
                        
                # Return highest scoring module
                if module_scores:
                    best_module = max(module_scores, key=module_scores.get)
                    return best_module
                else:
                    return 'brain_modules'  # General brain module
                    
            # Fall back to original classification logic
            return None
            
        except Exception as e:
            self.logger.error(f"Error in connectome-based classification: {e}")
            return None

    def organize_by_semantic_clusters(self, dry_run: bool = False) -> Dict:
        """
        Organize files using semantic clustering and connectome relationships
        
        Args:
            dry_run: If True, only show what would be done
            
        Returns:
            Organization summary with cluster information
        """
        summary = {
            "scanned": 0,
            "moved": 0,
            "clusters_created": 0,
            "semantic_moves": 0,
            "errors": 0,
            "cluster_details": {}
        }
        
        try:
            # Build semantic clusters
            clusters = self.build_semantic_clusters()
            summary["cluster_details"] = {k: len(v) for k, v in clusters.items()}
            summary["clusters_created"] = len(clusters)
            
            # Organize files based on clusters
            for cluster_path, file_list in clusters.items():
                target_dir = self.root_path / cluster_path
                
                for file_path in file_list:
                    file_obj = Path(file_path)
                    if file_obj.exists() and file_obj.is_file():
                        summary["scanned"] += 1
                        
                        # Check if file needs to be moved
                        current_dir = file_obj.parent
                        if str(current_dir) != str(target_dir):
                            if not dry_run:
                                target_dir.mkdir(parents=True, exist_ok=True)
                                target_file = target_dir / file_obj.name
                                
                                # Handle name conflicts
                                if target_file.exists():
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    target_file = target_dir / f"{file_obj.stem}_{timestamp}{file_obj.suffix}"
                                    
                                shutil.move(str(file_obj), str(target_file))
                                
                                self.movement_log.append({
                                    "timestamp": datetime.now().isoformat(),
                                    "source": str(file_obj),
                                    "target": str(target_file),
                                    "action": "semantic_cluster_move",
                                    "cluster": cluster_path
                                })
                                
                            summary["moved"] += 1
                            summary["semantic_moves"] += 1
                            self.logger.info(f"{'[DRY RUN] Would move' if dry_run else 'Moved'}: {file_obj.name} → {cluster_path}/")
                            
        except Exception as e:
            summary["errors"] += 1
            self.logger.error(f"Error in semantic clustering organization: {e}")
            
        return summary

    def monitor_and_organize(self):
        """
        Main monitoring function to be called periodically or on file system events
        """
        self.logger.info("Running periodic organization check...")
        summary = self.scan_and_organize()
        
        # Save movement log
        if self.movement_log:
            log_file = self.root_path / "logs" / f"file_movements_{datetime.now().strftime('%Y%m%d')}.json"
            with open(log_file, 'w') as f:
                json.dump(self.movement_log, f, indent=2)
                
        return summary

    def validate_structure(self) -> Dict:
        """
        Validate current directory structure and identify issues
        
        Returns:
            Validation report
        """
        report = {
            "valid": True,
            "issues": [],
            "recommendations": [],
            "structure_health": "good"
        }
        
        # Check for files in root that should be organized
        root_files = list(self.root_path.glob("*"))
        misplaced = []
        
        for item in root_files:
            if item.is_file() and item.name not in self.root_files:
                target = self.classify_file(item)
                if target and target != "DELETE":
                    misplaced.append(f"{item.name} → {target}/")
                    
        if misplaced:
            report["valid"] = False
            report["issues"].extend(misplaced)
            report["recommendations"].append("Run organization agent to move misplaced files")
            
        if len(misplaced) > 5:
            report["structure_health"] = "poor"
        elif len(misplaced) > 2:
            report["structure_health"] = "fair"
            
        return report


def main():
    """CLI interface for organization agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quark Directory Organization Agent with Connectome Integration")
    parser.add_argument("--scan", action="store_true", help="Scan and organize files using pattern matching")
    parser.add_argument("--semantic", action="store_true", help="Organize files using semantic clustering and connectome relationships")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--validate", action="store_true", help="Validate directory structure")
    parser.add_argument("--monitor", action="store_true", help="Run periodic monitoring")
    parser.add_argument("--analyze", metavar="FILE", help="Analyze semantic content of a specific file")
    parser.add_argument("--clusters", action="store_true", help="Show semantic clusters without organizing")
    
    args = parser.parse_args()
    
    agent = OrganizationAgent()
    
    if args.analyze:
        file_path = Path(args.analyze)
        if file_path.exists():
            analysis = agent.analyze_code_semantics(file_path)
            classification = agent.classify_by_connectome_relationships(file_path)
            
            print(f"\n=== Semantic Analysis: {file_path.name} ===")
            print(f"Brain relevance score: {analysis.get('brain_relevance', 0):.4f}")
            print(f"Complexity score: {analysis.get('complexity_score', 0)}")
            print(f"Dependencies: {len(analysis.get('dependencies', []))}")
            print(f"Brain keywords: {', '.join(analysis.get('brain_keywords', []))}")
            print(f"Functions: {len(analysis.get('functions', []))}")
            print(f"Classes: {len(analysis.get('classes', []))}")
            print(f"Recommended location: {classification or 'Root (no specific classification)'}")
        else:
            print(f"File not found: {args.analyze}")
            
    elif args.clusters:
        clusters = agent.build_semantic_clusters()
        print("\n=== Semantic Clusters ===")
        for cluster_name, files in clusters.items():
            print(f"\n{cluster_name} ({len(files)} files):")
            for file_path in files[:5]:  # Show first 5 files
                print(f"  • {Path(file_path).name}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
                
    elif args.validate:
        report = agent.validate_structure()
        print("\n=== Directory Structure Validation ===")
        print(f"Status: {'✅ VALID' if report['valid'] else '❌ NEEDS ORGANIZATION'}")
        print(f"Health: {report['structure_health'].upper()}")
        
        if report['issues']:
            print("\nMisplaced files:")
            for issue in report['issues']:
                print(f"  • {issue}")
                
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
                
    elif args.semantic:
        summary = agent.organize_by_semantic_clusters(dry_run=args.dry_run)
        print(f"\n=== Semantic Organization Summary ===")
        print(f"Files scanned: {summary['scanned']}")
        print(f"Files moved: {summary['moved']}")
        print(f"Semantic moves: {summary['semantic_moves']}")
        print(f"Clusters created: {summary['clusters_created']}")
        print(f"Errors: {summary['errors']}")
        
        if summary['cluster_details']:
            print("\nCluster breakdown:")
            for cluster, count in summary['cluster_details'].items():
                print(f"  • {cluster}: {count} files")
                
    elif args.scan:
        summary = agent.scan_and_organize(dry_run=args.dry_run)
        print(f"\n=== Pattern-Based Organization Summary ===")
        print(f"Files scanned: {summary['scanned']}")
        print(f"Files moved: {summary['moved']}")
        print(f"Files deleted: {summary['deleted']}")
        print(f"Errors: {summary['errors']}")
        
        if summary['actions']:
            print("\nActions taken:")
            for action in summary['actions'][:10]:  # Show first 10
                print(f"  • {action}")
            if len(summary['actions']) > 10:
                print(f"  ... and {len(summary['actions']) - 10} more")
                
    elif args.monitor:
        agent.monitor_and_organize()
        
    else:
        print("Available commands:")
        print("  --scan         : Pattern-based file organization")
        print("  --semantic     : Connectome-based semantic organization")
        print("  --clusters     : Show semantic clusters")
        print("  --analyze FILE : Analyze specific file semantics")
        print("  --validate     : Validate directory structure")
        print("  --monitor      : Run periodic monitoring")
        print("  --dry-run      : Preview changes without executing")
        print("\nUse --help for detailed usage information.")


if __name__ == "__main__":
    main()
