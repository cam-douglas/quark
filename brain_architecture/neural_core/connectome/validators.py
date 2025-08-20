# connectome/validators.py
# Graph validators & repair utilities to enforce your architectural + biological constraints.

import networkx as nx
from typing import Dict, Any, Tuple, List
import random
import math

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ensure_small_world(G: nx.Graph, target_cc: float, max_path: float) -> Tuple[float, float]:
    """Try simple rewiring to approach target clustering and path length."""
    # Compute baseline
    try:
        cc = nx.average_clustering(G)
    except Exception:
        cc = 0.0
    try:
        apl = nx.average_shortest_path_length(max(nx.connected_components(G), key=len).subgraph(G))  # type: ignore
    except Exception:
        apl = math.inf

    # Light-touch: random edge swaps to increase CC if very low
    attempts = 200
    if cc < target_cc and G.number_of_edges() > 0:
        for _ in range(attempts):
            u, v = random.choice(list(G.edges()))
            x, y = random.sample(list(G.nodes()), 2)
            if not G.has_edge(x, y) and x != y:
                G.remove_edge(u, v)
                G.add_edge(x, y)
                new_cc = nx.average_clustering(G)
                if new_cc < cc:  # revert if worse
                    G.remove_edge(x, y)
                    G.add_edge(u, v)
                else:
                    cc = new_cc

    # If path too long, add a few random shortcuts
    if (apl == math.inf or apl > max_path) and G.number_of_nodes() > 3:
        add_shortcuts = max(1, G.number_of_nodes() // 50)
        for _ in range(add_shortcuts):
            a, b = random.sample(list(G.nodes()), 2)
            if not G.has_edge(a, b):
                G.add_edge(a, b)

    # Recompute
    try:
        cc = nx.average_clustering(G)
    except Exception:
        cc = 0.0
    try:
        apl = nx.average_shortest_path_length(max(nx.connected_components(G), key=len).subgraph(G))  # type: ignore
    except Exception:
        apl = math.inf
    return cc, apl

def enforce_ei_balance(node_attrs: Dict[str, Dict[str, Any]], min_inh: float, max_inh: float):
    """Assign/repair neuron types to meet E/I constraints per module."""
    for module_id, attrs in node_attrs.items():
        pop = attrs["population"]
        e_ratio = attrs.get("excitatory_ratio", 0.8)
        i_ratio = 1.0 - e_ratio
        i_ratio = clamp(i_ratio, min_inh, max_inh)
        e_ratio = 1.0 - i_ratio
        attrs["excitatory_ratio"] = e_ratio
        attrs["inhibitory_ratio"] = i_ratio
        attrs["e_count"] = int(round(pop * e_ratio))
        attrs["i_count"] = pop - attrs["e_count"]

def require_links(module_ids: List[str], required_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Return edges that must exist between modules (undirected)."""
    edges = set()
    for src in module_ids:
        for dst in required_map.get(src, []):
            if src != dst:
                a, b = sorted([src, dst])
                edges.add((a, b))
    return sorted(list(edges))
