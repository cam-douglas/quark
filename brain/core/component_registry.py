"""
Component Registry
==================
Registry and discovery system for brain components.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib.util
import inspect


class ComponentRegistry:
    """
    Dynamic registry for brain components.
    
    This registry can discover and catalog brain managers and components
    without hardcoding their locations.
    """
    
    def __init__(self, brain_dir: Path):
        self.brain_dir = brain_dir
        self.components = {}
        self.managers = {}
        self._discover_components()
    
    def _discover_components(self) -> None:
        """Discover all brain components dynamically."""
        # Patterns to identify managers
        manager_patterns = [
            "**/manager*.py",
            "**/*_manager.py",
            "**/controller*.py", 
            "**/*_controller.py",
            "**/*_hub.py",
            "**/orchestrator*.py"
        ]
        
        # Search for managers
        for pattern in manager_patterns:
            for file_path in self.brain_dir.rglob(pattern):
                if "__pycache__" not in str(file_path) and "test" not in file_path.name:
                    self._register_component(file_path)
    
    def _register_component(self, file_path: Path) -> None:
        """Register a discovered component."""
        relative_path = file_path.relative_to(self.brain_dir)
        component_id = str(relative_path).replace("/", ".").replace(".py", "")
        
        # Determine category based on path
        category = self._determine_category(relative_path)
        
        self.components[component_id] = {
            "id": component_id,
            "name": self._extract_component_name(file_path),
            "path": str(relative_path),
            "absolute_path": str(file_path),
            "category": category,
            "type": self._determine_type(file_path.name),
            "module": None,  # Lazy loaded
            "instance": None,  # Lazy instantiated
        }
        
        # Add to managers if it's a manager type
        if self._is_manager(file_path.name):
            self.managers[component_id] = self.components[component_id]
    
    def _determine_category(self, path: Path) -> str:
        """Determine component category from its path."""
        path_str = str(path)
        
        if "cognitive_systems" in path_str:
            return "cognitive"
        elif "memory" in path_str or "hippocampus" in path_str:
            return "memory"
        elif "motor" in path_str or "basal_ganglia" in path_str:
            return "motor"
        elif "sensory" in path_str or "visual" in path_str or "auditory" in path_str:
            return "sensory"
        elif "prefrontal" in path_str or "executive" in path_str:
            return "executive"
        elif "learning" in path_str or "training" in path_str:
            return "learning"
        elif "language" in path_str:
            return "language"
        elif "simulation_engine" in path_str:
            return "simulation"
        else:
            return "other"
    
    def _determine_type(self, filename: str) -> str:
        """Determine component type from filename."""
        if "manager" in filename:
            return "manager"
        elif "controller" in filename:
            return "controller"
        elif "hub" in filename:
            return "hub"
        elif "orchestrator" in filename:
            return "orchestrator"
        else:
            return "component"
    
    def _is_manager(self, filename: str) -> bool:
        """Check if a file is a manager type."""
        manager_keywords = ["manager", "controller", "hub", "orchestrator"]
        return any(keyword in filename.lower() for keyword in manager_keywords)
    
    def _extract_component_name(self, file_path: Path) -> str:
        """Extract a human-readable component name."""
        name = file_path.stem
        # Convert snake_case to Title Case
        name = name.replace("_", " ").title()
        return name
    
    def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific component by ID."""
        return self.components.get(component_id)
    
    def get_managers_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all managers in a specific category."""
        return [
            comp for comp in self.managers.values()
            if comp["category"] == category
        ]
    
    def load_component(self, component_id: str) -> Optional[Any]:
        """Dynamically load a component module."""
        component = self.components.get(component_id)
        if not component:
            return None
        
        if component["module"] is None:
            try:
                spec = importlib.util.spec_from_file_location(
                    component_id,
                    component["absolute_path"]
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                component["module"] = module
            except Exception as e:
                print(f"Failed to load {component_id}: {e}")
                return None
        
        return component["module"]
    
    def instantiate_component(self, component_id: str, *args, **kwargs) -> Optional[Any]:
        """Instantiate a component class."""
        module = self.load_component(component_id)
        if not module:
            return None
        
        # Find the main class in the module
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                obj.__module__ == module.__name__ and
                not name.startswith("_")):
                try:
                    instance = obj(*args, **kwargs)
                    self.components[component_id]["instance"] = instance
                    return instance
                except Exception as e:
                    print(f"Failed to instantiate {name}: {e}")
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all registered components."""
        summary = {
            "total_components": len(self.components),
            "total_managers": len(self.managers),
            "by_category": {},
            "by_type": {}
        }
        
        for comp in self.components.values():
            # By category
            cat = comp["category"]
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            
            # By type
            typ = comp["type"]
            summary["by_type"][typ] = summary["by_type"].get(typ, 0) + 1
        
        return summary
