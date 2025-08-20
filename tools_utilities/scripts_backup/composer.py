from __future__ import annotations
import pathlib, yaml
from typing import Dict, List
from .....................................................config import ROOT, DEFAULT_MODELS
from .....................................................connectome import build_connectome, detect_communities

SMALLMIND_SIGNS = ["def main(", "if __name__ == '__main__'"]

def _read(p: pathlib.Path) -> str:
    try: 
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception: 
        return ""

def infer_agents_from_cluster(paths: List[pathlib.Path]) -> List[Dict]:
    agents = []
    for p in paths:
        if p.suffix == ".py" and ("agents/" in str(p) or any(s in _read(p) for s in SMALLMIND_SIGNS)):
            entry = f"{p.relative_to(ROOT).as_posix().rstrip('.py')}:main"
            agents.append({
                "id": f"sm.{p.stem}",
                "type": "smallmind",
                "entry": entry,
                "capabilities": ["reasoning", "planning", "python", "fs"],
                "concurrency": 1
            })
    return agents

def merge_models_yaml(new_models: List[Dict]):
    if DEFAULT_MODELS.exists():
        base = yaml.safe_load(DEFAULT_MODELS.read_text()) or {}
    else:
        base = {}
    base.setdefault("smallmind", [])
    known_ids = {m["id"] for m in base["smallmind"]}
    for m in new_models:
        if m["id"] not in known_ids:
            base["smallmind"].append(m)
            known_ids.add(m["id"])
    base.setdefault("routing", [])
    try:
        DEFAULT_MODELS.write_text(yaml.dump(base, sort_keys=False))
    except Exception:
        # Fallback if yaml not available
        import json
        DEFAULT_MODELS.with_suffix('.json').write_text(json.dumps(base, indent=2))

def compose_and_write() -> Dict:
    G = build_connectome()
    comms = detect_communities(G)
    clusters = [sorted(list(c)) for c in comms if len(c) > 0]
    new_agents: List[Dict] = []
    for cl in clusters:
        paths = [ROOT / p for p in cl]
        new_agents.extend(infer_agents_from_cluster(paths))
    merge_models_yaml(new_agents)
    return {"clusters": clusters, "new_agents": new_agents}
