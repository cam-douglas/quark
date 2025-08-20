# connectome/advanced_monitoring_agent.py
# Advanced monitoring agent addressing versioning, compliance, energy budgets, and observability gaps

import os
import json
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Set, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import networkx as nx
import yaml

from schema import load_config, ConnectomeConfig
from connectome_manager import compile_connectome, validate_connectome
from runtime_bus import ConnectomeBus, read_telemetry_sleep_flag

import logging
logger = logging.getLogger('advanced_monitoring')

@dataclass
class ConnectomeDiff:
    """Represents changes between connectome versions."""
    timestamp: str
    version_from: str
    version_to: str
    nodes_added: int
    nodes_removed: int
    edges_added: int
    edges_removed: int
    modules_changed: List[str]
    topology_changes: Dict[str, float]
    energy_delta: float
    compliance_status: str

@dataclass
class EnergyBudget:
    """Energy budget tracking for connectome operations."""
    total_budget: float
    current_consumption: float
    edge_costs: Dict[str, float]
    module_costs: Dict[str, float]
    optimization_savings: float
    budget_violations: List[str]

@dataclass
class ComplianceCheck:
    """Compliance verification results."""
    timestamp: str
    authority_chain: List[str]
    violations: List[str]
    warnings: List[str]
    approved_modules: Set[str]
    denied_operations: List[str]
    security_score: float

class AdvancedMonitoringAgent:
    """Advanced monitoring with versioning, compliance, energy budgets, and observability."""
    
    def __init__(self, config_path: str = "connectome/connectome.yaml"):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.running = False
        
        # Versioning
        self.version_history: List[Dict[str, Any]] = []
        self.current_version = self._generate_version_id()
        
        # Energy budget
        self.energy_budget = EnergyBudget(
            total_budget=10000.0,  # Base energy units
            current_consumption=0.0,
            edge_costs={},
            module_costs={},
            optimization_savings=0.0,
            budget_violations=[]
        )
        
        # Compliance
        self.compliance_state = ComplianceCheck(
            timestamp=datetime.now().isoformat(),
            authority_chain=[],
            violations=[],
            warnings=[],
            approved_modules=set(),
            denied_operations=[],
            security_score=1.0
        )
        
        # Metrics (Prometheus-style)
        self.metrics = {
            "connectome_builds_total": 0,
            "connectome_validations_total": 0,
            "connections_pruned_total": 0,
            "energy_budget_violations_total": 0,
            "compliance_violations_total": 0,
            "sleep_wake_cycles_total": 0,
            "gating_activations_total": 0,
            "node_churn_rate": 0.0,
            "edge_churn_rate": 0.0,
            "last_metric_update": time.time()
        }
        
        # Critical periods and curriculum
        self.developmental_stage = "F"  # Fetal, N0, N1, etc.
        self.stage_constraints = self._load_stage_constraints()
        
        # Task-state routing (communication vs control)
        self.communication_graph = None
        self.control_graph = None
        
        # Observability
        self.event_log = []
        self.performance_history = []
        
    def start(self):
        """Start the advanced monitoring agent."""
        if self.running:
            logger.warning("Advanced monitoring agent already running")
            return
            
        self.running = True
        self._initialize_versioning()
        self._load_compliance_rules()
        self._setup_energy_budget()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Advanced monitoring agent started")
        
    def stop(self):
        """Stop the monitoring agent."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        logger.info("Advanced monitoring agent stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop with comprehensive checks."""
        while self.running:
            try:
                current_time = time.time()
                
                # Version management
                self._check_for_connectome_changes()
                
                # Compliance monitoring
                self._run_compliance_checks()
                
                # Energy budget monitoring
                self._monitor_energy_budget()
                
                # Critical period checks
                self._apply_developmental_constraints()
                
                # Observability metrics
                self._update_metrics()
                
                # Performance monitoring
                self._monitor_performance()
                
                # Emit metrics and logs
                self._emit_observability_data()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in advanced monitoring loop: {e}")
                time.sleep(60)
                
    def _initialize_versioning(self):
        """Initialize versioned connectome tracking."""
        exports_dir = Path(self.cfg.exports.dir)
        versions_dir = exports_dir / "versions"
        versions_dir.mkdir(exist_ok=True)
        
        # Load existing version history
        history_file = versions_dir / "version_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.version_history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load version history: {e}")
                
    def _generate_version_id(self) -> str:
        """Generate unique version identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"
        
    def _check_for_connectome_changes(self):
        """Check for connectome changes and create versioned snapshots."""
        try:
            graph_path = os.path.join(self.cfg.exports.dir, "connectome.graphml")
            summary_path = os.path.join(self.cfg.exports.dir, "build_summary.json")
            
            if not os.path.exists(graph_path) or not os.path.exists(summary_path):
                return
                
            # Check if files have changed
            current_hash = self._calculate_connectome_hash(graph_path, summary_path)
            last_version = self.version_history[-1] if self.version_history else None
            
            if not last_version or last_version.get("hash") != current_hash:
                self._create_versioned_snapshot(current_hash)
                
        except Exception as e:
            logger.error(f"Error checking connectome changes: {e}")
            
    def _calculate_connectome_hash(self, graph_path: str, summary_path: str) -> str:
        """Calculate hash of connectome files for change detection."""
        hasher = hashlib.sha256()
        
        with open(graph_path, 'rb') as f:
            hasher.update(f.read())
        with open(summary_path, 'rb') as f:
            hasher.update(f.read())
            
        return hasher.hexdigest()[:16]
        
    def _create_versioned_snapshot(self, current_hash: str):
        """Create versioned snapshot of current connectome."""
        try:
            version_id = self._generate_version_id()
            exports_dir = Path(self.cfg.exports.dir)
            versions_dir = exports_dir / "versions"
            version_dir = versions_dir / version_id
            version_dir.mkdir(exist_ok=True)
            
            # Copy current files to versioned directory
            import shutil
            files_to_version = [
                "connectome.graphml",
                "connectome.json", 
                "build_summary.json",
                "validation_report.json"
            ]
            
            for filename in files_to_version:
                src = exports_dir / filename
                if src.exists():
                    shutil.copy2(src, version_dir / filename)
                    
            # Copy manifests
            for manifest_file in exports_dir.glob("*_manifest.json"):
                shutil.copy2(manifest_file, version_dir / manifest_file.name)
                
            # Create versioned build summary
            timestamp = datetime.now().isoformat()
            versioned_summary_path = exports_dir / f"build_summary_{version_id}.json"
            
            with open(exports_dir / "build_summary.json", 'r') as f:
                summary = json.load(f)
                
            summary["version"] = version_id
            summary["timestamp"] = timestamp
            summary["hash"] = current_hash
            
            with open(versioned_summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Calculate diff if previous version exists
            diff = self._calculate_connectome_diff(version_id)
            
            # Update version history
            version_record = {
                "version": version_id,
                "timestamp": timestamp,
                "hash": current_hash,
                "summary": summary,
                "diff": asdict(diff) if diff else None
            }
            
            self.version_history.append(version_record)
            
            # Save version history
            with open(versions_dir / "version_history.json", 'w') as f:
                json.dump(self.version_history, f, indent=2)
                
            logger.info(f"Created versioned snapshot: {version_id}")
            self.metrics["connectome_builds_total"] += 1
            
        except Exception as e:
            logger.error(f"Error creating versioned snapshot: {e}")
            
    def _calculate_connectome_diff(self, current_version: str) -> Optional[ConnectomeDiff]:
        """Calculate differences between connectome versions."""
        if len(self.version_history) < 1:
            return None
            
        try:
            previous_version = self.version_history[-1]["version"]
            
            # Load both graphs
            current_graph = nx.read_graphml(os.path.join(self.cfg.exports.dir, "connectome.graphml"))
            previous_path = Path(self.cfg.exports.dir) / "versions" / previous_version / "connectome.graphml"
            
            if not previous_path.exists():
                return None
                
            previous_graph = nx.read_graphml(str(previous_path))
            
            # Calculate differences
            current_nodes = set(current_graph.nodes())
            previous_nodes = set(previous_graph.nodes())
            current_edges = set(current_graph.edges())
            previous_edges = set(previous_graph.edges())
            
            nodes_added = len(current_nodes - previous_nodes)
            nodes_removed = len(previous_nodes - current_nodes)
            edges_added = len(current_edges - previous_edges)
            edges_removed = len(previous_edges - current_edges)
            
            # Calculate topology changes
            topology_changes = {}
            try:
                current_cc = nx.average_clustering(current_graph)
                previous_cc = nx.average_clustering(previous_graph)
                topology_changes["clustering_delta"] = current_cc - previous_cc
            except:
                topology_changes["clustering_delta"] = 0.0
                
            # Calculate energy delta
            energy_delta = self._calculate_energy_delta(current_graph, previous_graph)
            
            # Check which modules changed
            modules_changed = self._find_changed_modules(current_graph, previous_graph)
            
            return ConnectomeDiff(
                timestamp=datetime.now().isoformat(),
                version_from=previous_version,
                version_to=current_version,
                nodes_added=nodes_added,
                nodes_removed=nodes_removed,
                edges_added=edges_added,
                edges_removed=edges_removed,
                modules_changed=modules_changed,
                topology_changes=topology_changes,
                energy_delta=energy_delta,
                compliance_status="pending"
            )
            
        except Exception as e:
            logger.error(f"Error calculating connectome diff: {e}")
            return None
            
    def _load_compliance_rules(self):
        """Load compliance rules from authority chain."""
        try:
            authority_order = self.cfg.metadata.get("authority_order", [])
            self.compliance_state.authority_chain = authority_order
            
            # Load compliance review rules
            compliance_file = ".cursor/rules/compliance_review.md"
            if os.path.exists(compliance_file):
                with open(compliance_file, 'r') as f:
                    compliance_content = f.read()
                    
                # Parse compliance rules (simplified)
                if "DENY" in compliance_content.upper():
                    self.compliance_state.security_score *= 0.8
                    
                if "RESTRICT" in compliance_content.upper():
                    self.compliance_state.warnings.append("Restrictions detected in compliance review")
                    
            # Load approved modules from config
            for module in self.cfg.modules:
                self.compliance_state.approved_modules.add(module.id)
                
        except Exception as e:
            logger.error(f"Error loading compliance rules: {e}")
            
    def _run_compliance_checks(self):
        """Run compliance checks against current connectome."""
        try:
            violations = []
            warnings = []
            
            # Check if all modules are approved
            current_modules = {m.id for m in self.cfg.modules}
            for module_id in current_modules:
                if module_id not in self.compliance_state.approved_modules:
                    violations.append(f"Unauthorized module: {module_id}")
                    
            # Check for unauthorized connections
            graph_path = os.path.join(self.cfg.exports.dir, "connectome.graphml")
            if os.path.exists(graph_path):
                G = nx.read_graphml(graph_path)
                
                # Check for full-mesh violations
                possible_edges = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
                if G.number_of_edges() > 0.5 * possible_edges:
                    violations.append("Potential full-mesh topology detected")
                    
                # Check energy budget violations
                if self.energy_budget.current_consumption > self.energy_budget.total_budget:
                    violations.append("Energy budget exceeded")
                    
            # Update compliance state
            self.compliance_state.violations = violations
            self.compliance_state.warnings = warnings
            self.compliance_state.timestamp = datetime.now().isoformat()
            
            if violations:
                self.metrics["compliance_violations_total"] += len(violations)
                logger.warning(f"Compliance violations detected: {violations}")
                
        except Exception as e:
            logger.error(f"Error running compliance checks: {e}")
            
    def _setup_energy_budget(self):
        """Setup energy budget tracking."""
        try:
            # Initialize module energy costs
            for module in self.cfg.modules:
                base_cost = module.population * 0.1  # Base cost per neuron
                complexity_multiplier = len(module.required_links) * 0.2
                self.energy_budget.module_costs[module.id] = base_cost * (1 + complexity_multiplier)
                
            # Initialize edge costs (will be updated as graph changes)
            self._update_edge_energy_costs()
            
        except Exception as e:
            logger.error(f"Error setting up energy budget: {e}")
            
    def _update_edge_energy_costs(self):
        """Update energy costs for all edges."""
        try:
            graph_path = os.path.join(self.cfg.exports.dir, "connectome.graphml")
            if not os.path.exists(graph_path):
                return
                
            G = nx.read_graphml(graph_path)
            total_cost = 0.0
            
            for u, v in G.edges():
                # Cost based on node degrees and modules
                u_degree = G.degree(u)
                v_degree = G.degree(v)
                u_module = G.nodes[u].get('module', '')
                v_module = G.nodes[v].get('module', '')
                
                # Base edge cost
                edge_cost = 0.01 * (u_degree + v_degree) / 2
                
                # Inter-module connections cost more
                if u_module != v_module:
                    edge_cost *= 1.5
                    
                self.energy_budget.edge_costs[f"{u}-{v}"] = edge_cost
                total_cost += edge_cost
                
            self.energy_budget.current_consumption = total_cost
            
            # Check for budget violations
            if total_cost > self.energy_budget.total_budget:
                violation = f"Energy budget exceeded: {total_cost:.2f} > {self.energy_budget.total_budget}"
                self.energy_budget.budget_violations.append(violation)
                self.metrics["energy_budget_violations_total"] += 1
                
        except Exception as e:
            logger.error(f"Error updating edge energy costs: {e}")
            
    def _monitor_energy_budget(self):
        """Monitor and enforce energy budget."""
        self._update_edge_energy_costs()
        
        # Suggest optimizations if budget is tight
        utilization = self.energy_budget.current_consumption / self.energy_budget.total_budget
        if utilization > 0.9:
            logger.warning(f"Energy budget at {utilization*100:.1f}% utilization")
            
            # Find most expensive edges for potential pruning
            expensive_edges = sorted(
                self.energy_budget.edge_costs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            logger.info(f"Most expensive edges: {expensive_edges}")
            
    def _load_stage_constraints(self) -> Dict[str, Any]:
        """Load developmental stage constraints."""
        return {
            "F": {  # Fetal
                "projection_density_scale": 0.5,
                "gating_enabled": False,
                "sensory_inputs": False
            },
            "N0": {  # Neonate
                "projection_density_scale": 0.7,
                "gating_enabled": True,
                "sensory_inputs": True,
                "sleep_cycles": True
            },
            "N1": {  # Early Postnatal
                "projection_density_scale": 1.0,
                "gating_enabled": True,
                "sensory_inputs": True,
                "sleep_cycles": True,
                "cerebellar_modulation": True
            }
        }
        
    def _apply_developmental_constraints(self):
        """Apply stage-dependent constraints."""
        try:
            constraints = self.stage_constraints.get(self.developmental_stage, {})
            
            # Adjust projection density if needed
            if "projection_density_scale" in constraints:
                scale = constraints["projection_density_scale"]
                # This would trigger a recompilation with adjusted parameters
                # For now, just log the constraint
                logger.debug(f"Stage {self.developmental_stage}: projection density scale = {scale}")
                
            # Check gating status
            if not constraints.get("gating_enabled", True):
                logger.debug(f"Stage {self.developmental_stage}: gating disabled")
                
        except Exception as e:
            logger.error(f"Error applying developmental constraints: {e}")
            
    def _update_metrics(self):
        """Update Prometheus-style metrics."""
        current_time = time.time()
        
        # Update rates and counters
        if current_time - self.metrics["last_metric_update"] > 60:  # Update every minute
            # Calculate churn rates
            if len(self.version_history) >= 2:
                last_diff = self.version_history[-1].get("diff")
                if last_diff:
                    time_delta = 60.0  # 1 minute in seconds
                    self.metrics["node_churn_rate"] = (last_diff["nodes_added"] + last_diff["nodes_removed"]) / time_delta
                    self.metrics["edge_churn_rate"] = (last_diff["edges_added"] + last_diff["edges_removed"]) / time_delta
                    
            self.metrics["last_metric_update"] = current_time
            
    def _emit_observability_data(self):
        """Emit observability data in Prometheus format."""
        try:
            metrics_file = os.path.join(self.cfg.exports.dir, "metrics.json")
            
            # Add timestamps and additional context
            enriched_metrics = self.metrics.copy()
            enriched_metrics.update({
                "timestamp": datetime.now().isoformat(),
                "developmental_stage": self.developmental_stage,
                "energy_utilization": self.energy_budget.current_consumption / self.energy_budget.total_budget,
                "compliance_score": self.compliance_state.security_score,
                "active_violations": len(self.compliance_state.violations),
                "version_count": len(self.version_history)
            })
            
            with open(metrics_file, 'w') as f:
                json.dump(enriched_metrics, f, indent=2)
                
            # Also emit in ND-JSON format for streaming
            ndjson_file = os.path.join(self.cfg.exports.dir, "metrics.ndjson")
            with open(ndjson_file, 'a') as f:
                f.write(json.dumps(enriched_metrics) + '\n')
                
        except Exception as e:
            logger.error(f"Error emitting observability data: {e}")
            
    def _monitor_performance(self):
        """Monitor system performance."""
        try:
            # Collect performance metrics
            perf_data = {
                "timestamp": datetime.now().isoformat(),
                "memory_usage": self._get_memory_usage(),
                "cpu_usage": self._get_cpu_usage(),
                "disk_usage": self._get_disk_usage(),
                "connectome_size": self._get_connectome_size()
            }
            
            self.performance_history.append(perf_data)
            
            # Keep only last 100 measurements
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0.0
            
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
            
    def _get_disk_usage(self) -> float:
        """Get disk usage for exports directory."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.cfg.exports.dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # MB
        except:
            return 0.0
            
    def _get_connectome_size(self) -> Dict[str, int]:
        """Get current connectome size metrics."""
        try:
            graph_path = os.path.join(self.cfg.exports.dir, "connectome.graphml")
            if os.path.exists(graph_path):
                G = nx.read_graphml(graph_path)
                return {
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges()
                }
        except:
            pass
        return {"nodes": 0, "edges": 0}
        
    def _calculate_energy_delta(self, current_graph: nx.Graph, previous_graph: nx.Graph) -> float:
        """Calculate energy consumption change between graphs."""
        try:
            current_energy = len(current_graph.edges()) * 0.01  # Simplified
            previous_energy = len(previous_graph.edges()) * 0.01
            return current_energy - previous_energy
        except:
            return 0.0
            
    def _find_changed_modules(self, current_graph: nx.Graph, previous_graph: nx.Graph) -> List[str]:
        """Find modules that changed between graph versions."""
        changed = []
        
        try:
            # Group nodes by module
            current_modules = defaultdict(list)
            previous_modules = defaultdict(list)
            
            for node in current_graph.nodes():
                module = current_graph.nodes[node].get('module', 'unknown')
                current_modules[module].append(node)
                
            for node in previous_graph.nodes():
                module = previous_graph.nodes[node].get('module', 'unknown')
                previous_modules[module].append(node)
                
            # Find changed modules
            all_modules = set(current_modules.keys()) | set(previous_modules.keys())
            for module in all_modules:
                if len(current_modules[module]) != len(previous_modules[module]):
                    changed.append(module)
                    
        except Exception as e:
            logger.error(f"Error finding changed modules: {e}")
            
        return changed
        
    def rollback_to_version(self, version_id: str) -> bool:
        """Rollback connectome to a previous version."""
        try:
            versions_dir = Path(self.cfg.exports.dir) / "versions"
            version_dir = versions_dir / version_id
            
            if not version_dir.exists():
                logger.error(f"Version {version_id} not found")
                return False
                
            exports_dir = Path(self.cfg.exports.dir)
            
            # Backup current version first
            backup_id = f"backup_{self._generate_version_id()}"
            self._create_versioned_snapshot(backup_id)
            
            # Copy files from version directory
            import shutil
            for version_file in version_dir.glob("*"):
                if version_file.is_file():
                    target = exports_dir / version_file.name
                    shutil.copy2(version_file, target)
                    
            logger.info(f"Rolled back to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back to version {version_id}: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        # Convert sets to lists for JSON serialization
        compliance_state_dict = asdict(self.compliance_state)
        compliance_state_dict["approved_modules"] = list(compliance_state_dict["approved_modules"])
        
        return {
            "running": self.running,
            "current_version": self.current_version,
            "version_count": len(self.version_history),
            "developmental_stage": self.developmental_stage,
            "energy_budget": asdict(self.energy_budget),
            "compliance_state": compliance_state_dict,
            "metrics": self.metrics,
            "recent_performance": self.performance_history[-5:] if self.performance_history else []
        }

# Singleton instance
advanced_monitoring_agent = AdvancedMonitoringAgent()

def start_advanced_monitoring():
    """Start the advanced monitoring agent."""
    advanced_monitoring_agent.start()
    
def stop_advanced_monitoring():
    """Stop the advanced monitoring agent."""
    advanced_monitoring_agent.stop()
    
def get_advanced_monitoring_status():
    """Get advanced monitoring status."""
    return advanced_monitoring_agent.get_status()
    
def rollback_connectome(version_id: str):
    """Rollback connectome to specified version."""
    return advanced_monitoring_agent.rollback_to_version(version_id)
