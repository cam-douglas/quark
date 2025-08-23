# connectome/maintenance_agent.py
# Autonomous agent for connectome maintenance, optimization, and directory management.

import os
import json
import time
import shutil
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Set
from pathlib import Path

import networkx as nx
import yaml

from schema import load_config
from connectome_manager import compile_connectome, validate_connectome
from runtime_bus import ConnectomeBus, read_telemetry_sleep_flag

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('connectome_maintenance')

class ConnectomeMaintenanceAgent:
    """Autonomous agent for connectome optimization and maintenance."""
    
    def __init__(self, config_path: str = "connectome/connectome.yaml", 
                 maintenance_config_path: str = "connectome/maintenance_config.yaml"):
        self.config_path = config_path
        self.maintenance_config_path = maintenance_config_path
        self.cfg = load_config(config_path)
        self.maintenance_cfg = self._load_maintenance_config()
        self.bus = ConnectomeBus()
        self.running = False
        self.last_optimization = time.time()
        self.last_cleanup = time.time()
        
        # Load intervals from config
        intervals = self.maintenance_cfg.get("maintenance", {}).get("intervals", {})
        self.optimization_interval = intervals.get("optimization_seconds", 300)
        self.cleanup_interval = intervals.get("cleanup_seconds", 1800)
        self.health_check_interval = intervals.get("health_check_seconds", 30)
        self.statistics = {
            "connections_pruned": 0,
            "redundant_paths_removed": 0,
            "files_cleaned": 0,
            "optimizations_performed": 0,
            "last_maintenance": None
        }
        
    def _load_maintenance_config(self) -> Dict[str, Any]:
        """Load maintenance configuration from YAML file."""
        try:
            if os.path.exists(self.maintenance_config_path):
                with open(self.maintenance_config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load maintenance config: {e}")
        
        # Return default config
        return {
            "maintenance": {
                "intervals": {
                    "optimization_seconds": 300,
                    "cleanup_seconds": 1800,
                    "health_check_seconds": 30
                },
                "connection_optimization": {
                    "enable_pruning": True,
                    "prune_threshold": 0.05,
                    "max_prune_percentage": 25
                },
                "directory_cleanup": {
                    "enable_temp_cleanup": True,
                    "enable_log_rotation": True,
                    "max_log_size_mb": 10
                }
            }
        }
        
    def start(self):
        """Start the maintenance agent in a background thread."""
        if self.running:
            logger.warning("Maintenance agent already running")
            return
            
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        logger.info("Connectome maintenance agent started")
        
    def stop(self):
        """Stop the maintenance agent."""
        self.running = False
        if hasattr(self, 'maintenance_thread'):
            self.maintenance_thread.join(timeout=5)
        logger.info("Connectome maintenance agent stopped")
        
    def _maintenance_loop(self):
        """Main maintenance loop."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if optimization is needed
                if current_time - self.last_optimization > self.optimization_interval:
                    self._optimize_connections()
                    self.last_optimization = current_time
                
                # Check if cleanup is needed
                if current_time - self.last_cleanup > self.cleanup_interval:
                    self._cleanup_directories()
                    self.last_cleanup = current_time
                
                # Monitor system health
                self._monitor_system_health()
                
                # Save statistics
                self._save_statistics()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                time.sleep(60)  # Wait longer on error
                
    def _optimize_connections(self):
        """Optimize connectome connections by pruning unnecessary ones."""
        try:
            logger.info("Starting connection optimization...")
            
            # Load current connectome
            graph_path = os.path.join(self.cfg.exports.dir, "connectome.graphml")
            if not os.path.exists(graph_path):
                logger.warning("No connectome graph found for optimization")
                return
                
            G = nx.read_graphml(graph_path)
            original_edges = G.number_of_edges()
            
            # Prune weak connections
            pruned_count = self._prune_weak_connections(G)
            
            # Remove redundant paths
            redundant_removed = self._remove_redundant_paths(G)
            
            # Optimize small-world properties
            self._optimize_small_world_properties(G)
            
            # Save optimized graph
            nx.write_graphml(G, graph_path)
            
            # Update statistics
            self.statistics["connections_pruned"] += pruned_count
            self.statistics["redundant_paths_removed"] += redundant_removed
            self.statistics["optimizations_performed"] += 1
            
            final_edges = G.number_of_edges()
            logger.info(f"Optimization complete: {original_edges} → {final_edges} edges "
                       f"({pruned_count} pruned, {redundant_removed} redundant removed)")
                       
        except Exception as e:
            logger.error(f"Error during connection optimization: {e}")
            
    def _prune_weak_connections(self, G: nx.Graph) -> int:
        """Remove connections below threshold strength."""
        edges_to_remove = []
        
        # Calculate edge weights based on node degrees
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        for u, v in G.edges():
            # Calculate connection strength
            u_strength = degrees.get(u, 1) / max_degree
            v_strength = degrees.get(v, 1) / max_degree
            connection_strength = (u_strength + v_strength) / 2
            
            # Prune very weak connections (bottom 5%)
            if connection_strength < 0.05:
                # But preserve required module connections
                u_module = G.nodes[u].get('module', '')
                v_module = G.nodes[v].get('module', '')
                
                if not self._is_required_connection(u_module, v_module):
                    edges_to_remove.append((u, v))
        
        # Remove edges
        G.remove_edges_from(edges_to_remove[:len(edges_to_remove)//4])  # Only remove 25% of weak edges
        return len(edges_to_remove[:len(edges_to_remove)//4])
        
    def _remove_redundant_paths(self, G: nx.Graph) -> int:
        """Remove redundant paths while preserving connectivity."""
        removed_count = 0
        
        # Find triangles (3-cycles) and remove one edge from some of them
        triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
        
        for triangle in triangles[:len(triangles)//10]:  # Only process 10% to avoid over-pruning
            if len(triangle) == 3:
                u, v, w = triangle
                # Remove the edge with lowest combined degree
                edges = [(u, v), (v, w), (u, w)]
                degrees = [(G.degree(a) + G.degree(b), a, b) for a, b in edges if G.has_edge(a, b)]
                
                if degrees:
                    _, a, b = min(degrees)
                    if G.has_edge(a, b):
                        # Check if removing this edge keeps graph connected
                        G.remove_edge(a, b)
                        if nx.is_connected(G):
                            removed_count += 1
                        else:
                            G.add_edge(a, b)  # Restore if it breaks connectivity
                            
        return removed_count
        
    def _optimize_small_world_properties(self, G: nx.Graph):
        """Optimize small-world properties of the network."""
        try:
            # Calculate current metrics
            clustering = nx.average_clustering(G)
            
            # If clustering is too low, add some triangle-closing edges
            if clustering < 0.1:
                self._add_triangle_closing_edges(G, target_count=10)
                
        except Exception as e:
            logger.warning(f"Error optimizing small-world properties: {e}")
            
    def _add_triangle_closing_edges(self, G: nx.Graph, target_count: int):
        """Add edges that close triangles to improve clustering."""
        added = 0
        nodes = list(G.nodes())
        
        for _ in range(target_count * 10):  # Try up to 10x target
            if added >= target_count:
                break
                
            # Pick a random node
            u = nx.utils.misc.choice(nodes)
            neighbors = list(G.neighbors(u))
            
            if len(neighbors) >= 2:
                # Find two neighbors that aren't connected
                for i, v in enumerate(neighbors):
                    for w in neighbors[i+1:]:
                        if not G.has_edge(v, w):
                            G.add_edge(v, w)
                            added += 1
                            if added >= target_count:
                                return
                                
    def _is_required_connection(self, module_a: str, module_b: str) -> bool:
        """Check if connection between modules is required."""
        if not module_a or not module_b:
            return False
            
        # Find module specs
        module_map = {m.id: m for m in self.cfg.modules}
        
        if module_a in module_map and module_b in module_map.required_links:
            return True
        if module_b in module_map and module_a in module_map.required_links:
            return True
            
        return False
        
    def _cleanup_directories(self):
        """Clean up and organize directories."""
        try:
            logger.info("Starting directory cleanup...")
            
            cleanup_stats = {
                "temp_files_removed": 0,
                "old_exports_archived": 0,
                "logs_rotated": 0
            }
            
            # Clean temporary files
            cleanup_stats["temp_files_removed"] = self._clean_temp_files()
            
            # Archive old exports
            cleanup_stats["old_exports_archived"] = self._archive_old_exports()
            
            # Rotate logs
            cleanup_stats["logs_rotated"] = self._rotate_logs()
            
            # Organize exports directory
            self._organize_exports_directory()
            
            self.statistics["files_cleaned"] += sum(cleanup_stats.values())
            
            logger.info(f"Directory cleanup complete: {cleanup_stats}")
            
        except Exception as e:
            logger.error(f"Error during directory cleanup: {e}")
            
    def _clean_temp_files(self) -> int:
        """Remove temporary and cache files."""
        temp_patterns = [
            "**/*.tmp",
            "**/*.cache",
            "**/__pycache__",
            "**/.*_temp",
            "**/temp_*"
        ]
        
        removed_count = 0
        for pattern in temp_patterns:
            for path in Path(".").glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                        removed_count += 1
                    elif path.is_dir():
                        shutil.rmtree(path)
                        removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {path}: {e}")
                    
        return removed_count
        
    def _archive_old_exports(self) -> int:
        """Archive old export files."""
        exports_dir = Path(self.cfg.exports.dir)
        if not exports_dir.exists():
            return 0
            
        archive_dir = exports_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        archived_count = 0
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago
        
        for file_path in exports_dir.glob("*.json"):
            if file_path.name.startswith(("build_", "validation_", "state_")):
                if file_path.stat().st_mtime < cutoff_time:
                    archive_path = archive_dir / f"{datetime.now().strftime('%Y%m%d')}_{file_path.name}"
                    try:
                        shutil.move(str(file_path), str(archive_path))
                        archived_count += 1
                    except Exception as e:
                        logger.warning(f"Could not archive {file_path}: {e}")
                        
        return archived_count
        
    def _rotate_logs(self) -> int:
        """Rotate log files if they get too large."""
        log_files = list(Path(".").glob("**/*.log"))
        rotated_count = 0
        
        for log_file in log_files:
            try:
                if log_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
                    backup_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d')}.log"
                    backup_path = log_file.parent / backup_name
                    shutil.move(str(log_file), str(backup_path))
                    rotated_count += 1
            except Exception as e:
                logger.warning(f"Could not rotate log {log_file}: {e}")
                
        return rotated_count
        
    def _organize_exports_directory(self):
        """Organize exports directory structure."""
        exports_dir = Path(self.cfg.exports.dir)
        if not exports_dir.exists():
            return
            
        # Create subdirectories for different types of exports
        subdirs = ["graphs", "manifests", "reports", "archive"]
        for subdir in subdirs:
            (exports_dir / subdir).mkdir(exist_ok=True)
            
        # Move files to appropriate subdirectories
        try:
            # Move graph files
            for graph_file in exports_dir.glob("connectome.*"):
                if graph_file.is_file():
                    target = exports_dir / "graphs" / graph_file.name
                    if not target.exists():
                        shutil.move(str(graph_file), str(target))
                        
            # Move manifest files
            for manifest_file in exports_dir.glob("*_manifest.json"):
                if manifest_file.is_file():
                    target = exports_dir / "manifests" / manifest_file.name
                    if not target.exists():
                        shutil.move(str(manifest_file), str(target))
                        
            # Move report files
            for report_file in exports_dir.glob("*_report.json"):
                if report_file.is_file():
                    target = exports_dir / "reports" / report_file.name
                    if not target.exists():
                        shutil.move(str(report_file), str(target))
                        
        except Exception as e:
            logger.warning(f"Error organizing exports directory: {e}")
            
    def _monitor_system_health(self):
        """Monitor system health and adjust maintenance intervals."""
        try:
            # Check system load
            if read_telemetry_sleep_flag():
                # System under stress, reduce maintenance frequency
                self.optimization_interval = max(600, self.optimization_interval * 1.2)
                self.cleanup_interval = max(3600, self.cleanup_interval * 1.2)
            else:
                # System healthy, can increase maintenance frequency
                self.optimization_interval = max(300, self.optimization_interval * 0.9)
                self.cleanup_interval = max(1800, self.cleanup_interval * 0.9)
                
            # Check export directory size
            exports_size = self._get_directory_size(self.cfg.exports.dir)
            if exports_size > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"Exports directory large ({exports_size / 1024 / 1024:.1f}MB), scheduling cleanup")
                self._cleanup_directories()
                
        except Exception as e:
            logger.warning(f"Error monitoring system health: {e}")
            
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except Exception:
            pass
        return total_size
        
    def _save_statistics(self):
        """Save maintenance statistics."""
        self.statistics["last_maintenance"] = datetime.now().isoformat()
        
        stats_path = os.path.join(self.cfg.exports.dir, "maintenance_stats.json")
        try:
            with open(stats_path, "w") as f:
                json.dump(self.statistics, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save statistics: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get current maintenance agent status."""
        return {
            "running": self.running,
            "last_optimization": datetime.fromtimestamp(self.last_optimization).isoformat(),
            "last_cleanup": datetime.fromtimestamp(self.last_cleanup).isoformat(),
            "optimization_interval": self.optimization_interval,
            "cleanup_interval": self.cleanup_interval,
            "statistics": self.statistics.copy()
        }

# Singleton instance
maintenance_agent = ConnectomeMaintenanceAgent()

def start_maintenance_agent():
    """Start the global maintenance agent."""
    maintenance_agent.start()
    
def stop_maintenance_agent():
    """Stop the global maintenance agent."""
    maintenance_agent.stop()
    
def get_maintenance_status():
    """Get maintenance agent status."""
    return maintenance_agent.get_status()
