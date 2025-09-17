#!/usr/bin/env python3
"""Dataset Management Module - Unified interface for dataset integration and management.

Provides streamlined interface to dataset discovery and loading with preserved functionality.

Integration: Main dataset management interface for neural learning workflows.
Rationale: Simplified dataset management with consolidated functionality.
"""

from typing import Dict, List, Any
from .dataset_discovery import DatasetDiscovery

class DatasetIntegration:
    """Simplified dataset integration interface."""

    def __init__(self, base_path: str = "/Users/camdouglas/quark"):
        self.discovery = DatasetDiscovery(base_path)
        print("📚 Dataset Integration System initialized")
        print(f"   Base path: {base_path}")
        print(f"   LLM-IK path: {self.discovery.ik_path}")
        print(f"   Manipulation path: {self.discovery.manipulation_path}")

    def load_ik_training_data(self, robot_name: str = "UR5") -> Dict[str, Any]:
        """Load IK training data for specified robot."""
        robot_solutions = {}

        for key, solution in self.discovery.ik_solutions.items():
            if robot_name.lower() in key.lower():
                robot_solutions[key] = solution

        return {
            "robot_name": robot_name,
            "solutions_found": len(robot_solutions),
            "solutions": robot_solutions,
            "total_size": sum(sol["file_size"] for sol in robot_solutions.values())
        }

    def load_manipulation_training_data(self, task_types: List[str] = None) -> Dict[str, Any]:
        """Load manipulation training data for specified task types."""
        filtered_demos = {}

        for demo_name, demo_data in self.discovery.manipulation_demos.items():
            if task_types is None or demo_data["task_type"] in task_types:
                filtered_demos[demo_name] = demo_data

        return {
            "task_types": task_types,
            "demos_found": len(filtered_demos),
            "demos": filtered_demos,
            "total_trajectories": sum(demo["demo_length"] for demo in filtered_demos.values())
        }

    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of dataset integration status."""
        return self.discovery.get_discovery_summary()

# Export for backward compatibility
__all__ = ['DatasetIntegration', 'DatasetDiscovery']
