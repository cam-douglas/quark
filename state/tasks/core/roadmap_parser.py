"""
Roadmap Parser Module
=====================
Parses roadmap documents to extract milestones, KPIs, and task information.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class RoadmapParser:
    """Parser for roadmap markdown files."""
    
    def __init__(self, roadmap_dir: Path):
        self.roadmap_dir = roadmap_dir
        
        # Stage definitions
        self.stages = {
            1: "embryonic",
            2: "fetal", 
            3: "early_postnatal",
            4: "childhood",
            5: "adolescence",
            6: "adult"
        }
    
    def parse_roadmap_file(self, filepath: Path) -> Tuple[List[Dict], List[Dict]]:
        """Parse a roadmap markdown file to extract milestones and KPIs."""
        milestones = []
        kpis = []
        
        if not filepath.exists():
            return milestones, kpis
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Parse milestones (engineering milestones section)
        in_milestones = False
        in_kpis = False
        
        for line in lines:
            # Detect sections
            if "Engineering Milestones" in line:
                in_milestones = True
                in_kpis = False
                continue
            elif "kpis:" in line.lower():
                in_milestones = False
                in_kpis = True
                continue
            elif line.startswith("##") and in_milestones:
                in_milestones = False
            
            # Parse milestone lines
            if in_milestones and line.strip().startswith("*"):
                milestone = self._parse_milestone_line(line)
                if milestone:
                    milestones.append(milestone)
            
            # Parse KPIs from YAML section
            if in_kpis:
                kpi = self._parse_kpi_line(line)
                if kpi:
                    kpis.append(kpi)
        
        return milestones, kpis
    
    def _parse_milestone_line(self, line: str) -> Optional[Dict]:
        """Parse a single milestone line."""
        match = re.match(r'\* \[([^\]]+)\]\s*(✅|🚨|⏳)?\s*\*?\*?([^*]+)\*?\*?(.*)$', line)
        if match:
            category = match.group(1)
            status = match.group(2) or ""
            status_text = self._get_status_text(status)
            description = match.group(3).strip()
            details = match.group(4).strip() if match.group(4) else ""
            
            # Extract KPI references
            kpi_match = re.search(r'\*\*\(KPI: ([^)]+)\)\*\*', details)
            kpi_refs = kpi_match.group(1) if kpi_match else ""
            
            return {
                "category": category,
                "status": status_text,
                "description": description,
                "details": details,
                "kpi_refs": kpi_refs
            }
        return None
    
    def _parse_kpi_line(self, line: str) -> Optional[Dict]:
        """Parse a KPI line from YAML section."""
        kpi_match = re.match(r'\s+(\w+):\s*"([^"]+)"', line)
        if kpi_match:
            return {
                "name": kpi_match.group(1),
                "target": kpi_match.group(2)
            }
        return None
    
    def _get_status_text(self, status_icon: str) -> str:
        """Convert status icon to text."""
        if "✅" in status_icon:
            return "COMPLETED"
        elif "🚨" in status_icon:
            return "PENDING"
        elif "⏳" in status_icon:
            return "IN_PROGRESS"
        else:
            return "PENDING"
    
    def get_roadmap_file(self, stage: Optional[int] = None, category: Optional[str] = None) -> Path:
        """Get the appropriate roadmap file based on stage or category."""
        if stage:
            stage_name = self.stages.get(stage, 'unknown')
            return self.roadmap_dir / f"stage{stage}_{stage_name}_rules.md"
        elif category:
            # Map category to appropriate roadmap
            category_map = {
                "cerebellum": "stage1_embryonic_rules.md",
                "foundation": "stage1_embryonic_rules.md",
                "developmental": "stage1_embryonic_rules.md",
                "brainstem": "stage1_embryonic_rules.md"
            }
            return self.roadmap_dir / category_map.get(category, "stage1_embryonic_rules.md")
        else:
            return self.roadmap_dir / "stage1_embryonic_rules.md"
    
    def get_available_kpis(self) -> List[Dict]:
        """Get list of available KPIs from roadmaps."""
        return [
            {"name": "segmentation_dice", "target": "≥ 0.80", 
             "description": "Regional accuracy vs Allen Atlas"},
            {"name": "neuron_count_error_pct", "target": "≤ 5%", 
             "description": "Neuron population accuracy"},
            {"name": "laminar_accuracy", "target": "≥ 0.85", 
             "description": "Layer specification accuracy"},
            {"name": "cell_count_variance", "target": "≤ 5%", 
             "description": "Cell population stability"},
            {"name": "experimental_accuracy", "target": "≥ 0.70", 
             "description": "Human data validation"},
            {"name": "grid_resolution_mm", "target": "≤ 0.001", 
             "description": "Spatial precision"},
            {"name": "meninges_mesh_integrity", "target": "valid", 
             "description": "Structural validation"},
            {"name": "computational_efficiency", "target": "<2s", 
             "description": "Per timestep performance"}
        ]
