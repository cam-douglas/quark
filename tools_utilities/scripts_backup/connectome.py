from __future__ import annotations
import pathlib, itertools
from typing import Dict, List, Tuple
from ................................................config import ROOT, SETTINGS, load_hebbian, save_hebbian
from ................................................scanners import scan_files, read_text_safe
from ................................................analyzers import analyze_file
from ................................................embedder import TextEmbedder, cosine

def build_connectome():
    """Build a basic connectome using simple heuristics."""
    files = scan_files()
    texts = [read_text_safe(p) for p in files]
    meta  = [analyze_file(p, t) for p,t in zip(files, texts)]
    
    # Simple graph representation
    nodes = []
    edges = []
    
    for i, (p, m) in enumerate(zip(files, meta)):
        nodes.append({
            "id": str(p), 
            "kind": m.get("kind", "other"), 
            "symbols": m.get("symbols", []), 
            "imports": m.get("imports", [])
        })
    
    # Find import connections
    for i, (p, m) in enumerate(zip(files, meta)):
        imports = set(m.get("imports", []))
        if not imports: 
            continue
        for j, q in enumerate(files):
            if i != j and q.suffix == ".py" and q.stem in imports:
                edges.append({
                    "u": str(p), 
                    "v": str(q), 
                    "weight": 0.85, 
                    "cause": "import"
                })
    
    return {"nodes": nodes, "edges": edges}

def detect_communities(G):
    """Simple community detection based on file paths."""
    nodes = G.get("nodes", [])
    communities = {}
    
    for node in nodes:
        path = pathlib.Path(node["id"])
        # Group by parent directory
        parent = str(path.parent) if path.parent != path else "root"
        if parent not in communities:
            communities[parent] = []
        communities[parent].append(node["id"])
    
    return list(communities.values())

def record_coactivation(paths: List[str]):
    """Record file co-activation for Hebbian learning."""
    hebb = load_hebbian()
    edges = hebb.setdefault("edges", {})
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            key = f"{paths[i]}|||{paths[j]}"
            edges[key] = edges.get(key, 0) + 1
    save_hebbian(hebb)
