# management/rules/roadmaps/roadmap_controller.py
"""Roadmap Controller

Centralised loader that aggregates every roadmap YAML/Markdown file under
``management/rules/roadmaps/`` and exposes a single helper for the state system.
"""
from pathlib import Path
from typing import List, Dict
import yaml

ROOT_DIR = Path(__file__).resolve().parents[0]

SUPPORTED_EXTS = {'.yaml', '.yml', '.md'}


def _load_md(path: Path) -> Dict:
    """Very naive front-matter extractor for Markdown roadmaps."""
    title = path.stem.replace('_', ' ').title()
    return {"path": str(path), "title": title, "format": "markdown"}


def _load_yaml(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text()) or {}
    data.setdefault("path", str(path))
    data.setdefault("title", path.stem)
    data["format"] = "yaml"
    return data


def get_all_roadmaps() -> List[Dict]:
    roadmaps = []
    for fp in ROOT_DIR.rglob('*'):
        if fp.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if fp.suffix.lower() in {'.yaml', '.yml'}:
            roadmaps.append(_load_yaml(fp))
        else:
            roadmaps.append(_load_md(fp))
    return roadmaps


if __name__ == "__main__":
    import json, sys
    json.dump(get_all_roadmaps(), sys.stdout, indent=2)
