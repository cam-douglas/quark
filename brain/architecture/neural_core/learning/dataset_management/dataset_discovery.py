#!/usr/bin/env python3
"""Dataset Discovery Module - Discovery and cataloging of available datasets.

Handles discovery of IK solutions, manipulation demos, and training datasets.

Integration: Dataset discovery for neural learning and training workflows.
Rationale: Specialized dataset discovery logic separate from data loading.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DatasetDiscovery:
    """Discovers and catalogs available training datasets."""

    def __init__(self, base_path: str = "/Users/camdouglas/quark"):
        self.base_path = Path(base_path)
        self.ik_path = self.base_path / "external" / "llm-ik"
        self.manipulation_path = self.base_path / "external" / "llm-articulated-manipulation"

        # Dataset catalogs
        self.ik_solutions = {}
        self.manipulation_demos = {}
        self.prompt_templates = {}

        logger.info("🔍 Discovering available datasets...")
        self._discover_all_datasets()
        logger.info("✅ Dataset discovery complete")

    def _discover_all_datasets(self):
        """Discover all available datasets."""
        self._discover_ik_solutions()
        self._discover_manipulation_demos()
        self._discover_prompt_templates()

        logger.info(f"   IK solutions: {len(self.ik_solutions)}")
        logger.info(f"   Manipulation demos: {len(self.manipulation_demos)}")
        logger.info(f"   Prompt templates: {len(self.prompt_templates)}")

    def _discover_ik_solutions(self):
        """Discover available IK solution datasets."""
        ik_data_path = self.ik_path / "data"

        if not ik_data_path.exists():
            return

        for solution_file in ik_data_path.glob("**/*.json"):
            try:
                solution_data = self._parse_ik_solution_filename(solution_file.name)
                if solution_data:
                    key = f"{solution_data['robot']}_{solution_data['config']}"
                    self.ik_solutions[key] = {
                        "file_path": solution_file,
                        "robot": solution_data["robot"],
                        "config": solution_data["config"],
                        "file_size": solution_file.stat().st_size,
                        "last_modified": solution_file.stat().st_mtime
                    }
            except Exception as e:
                logger.warning(f"Error parsing IK solution {solution_file}: {e}")

    def _discover_manipulation_demos(self):
        """Discover available manipulation demonstration datasets."""
        demo_path = self.manipulation_path / "data" / "demos"

        if not demo_path.exists():
            return

        for demo_file in demo_path.glob("**/*.json"):
            try:
                demo_data = self._parse_manipulation_demo(demo_file)
                if demo_data:
                    self.manipulation_demos[demo_file.stem] = demo_data
            except Exception as e:
                logger.warning(f"Error parsing manipulation demo {demo_file}: {e}")

    def _discover_prompt_templates(self):
        """Discover available prompt templates."""
        template_paths = [
            self.ik_path / "prompts",
            self.manipulation_path / "prompts"
        ]

        for template_path in template_paths:
            if not template_path.exists():
                continue

            for template_file in template_path.glob("**/*.txt"):
                try:
                    self.prompt_templates[template_file.stem] = {
                        "file_path": template_file,
                        "category": template_file.parent.name,
                        "file_size": template_file.stat().st_size
                    }
                except Exception as e:
                    logger.warning(f"Error cataloging template {template_file}: {e}")

    def _parse_ik_solution_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse IK solution filename to extract metadata."""
        # Example: "ur5_6dof_solutions.json"
        try:
            name_parts = filename.replace('.json', '').split('_')
            if len(name_parts) >= 2:
                return {
                    "robot": name_parts[0].upper(),
                    "config": '_'.join(name_parts[1:]),
                    "filename": filename
                }
        except Exception as e:
            logger.warning(f"Error parsing IK filename {filename}: {e}")

        return None

    def _parse_manipulation_demo(self, demo_file: Path) -> Optional[Dict[str, Any]]:
        """Parse manipulation demonstration file."""
        try:
            with open(demo_file, 'r') as f:
                demo_data = json.load(f)

            return {
                "file_path": demo_file,
                "task_type": demo_data.get("task_type", "unknown"),
                "robot_type": demo_data.get("robot_type", "unknown"),
                "demo_length": len(demo_data.get("trajectory", [])),
                "success_rate": demo_data.get("success_rate", 0.0),
                "file_size": demo_file.stat().st_size
            }

        except Exception as e:
            logger.warning(f"Error parsing demo {demo_file}: {e}")
            return None

    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovered datasets."""
        return {
            "ik_solutions": len(self.ik_solutions),
            "manipulation_demos": len(self.manipulation_demos),
            "prompt_templates": len(self.prompt_templates),
            "total_datasets": len(self.ik_solutions) + len(self.manipulation_demos) + len(self.prompt_templates),
            "base_path": str(self.base_path)
        }
