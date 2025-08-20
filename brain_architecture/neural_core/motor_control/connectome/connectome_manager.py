# connectome/connectome_manager.py
# Build, validate, export, and watch the connectome. This is the core engine.

import os
import json
import time
import random
from typing import Dict, Any, List, Tuple
import networkx as nx
from schema import load_config, ConnectomeConfig
from validators import ensure_small_world, enforce_ei_balance, require_links, clamp
from runtime_bus import read_telemetry_sleep_flag

import yaml

EXPORT_MANIFEST_DIR = "connectome/exports"

def _rng():
    random.seed(42)

def build_module_populations(cfg: ConnectomeConfig) -> Dict[str, List[str]]:
    """Create per-module neuron ids, excluding dormant modules."""
    pop_map: Dict[str, List[str]] = {}
    for m in cfg.modules:
        # Only include active modules
        if getattr(m, 'activation', True):
            pop_map[m.id] = [f"{m.id}:n{idx}" for idx in range(m.population)]
    return pop_map

def ring_lattice(nodes: List[str], k: int) -> nx.Graph:
    """Even k ring lattice."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    n = len(nodes)
    k = max(2, k + (k % 2))  # ensure even
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(nodes[i], nodes[(i + j) % n])
            G.add_edge(nodes[i], nodes[(i - j) % n])
    return G

def rewire_ws(G: nx.Graph, p: float):
    """Watts–Strogatz style rewiring on existing ring lattice."""
    nodes = list(G.nodes())
    edges = list(G.edges())
    for (u, v) in edges:
        if random.random() < p:
            G.remove_edge(u, v)
            w = random.choice(nodes)
            tries = 0
            while (w == u or G.has_edge(u, w)) and tries < 10:
                w = random.choice(nodes)
                tries += 1
            if w != u and not G.has_edge(u, w):
                G.add_edge(u, w)

def wire_intra_module(cfg: ConnectomeConfig,
                      pop_map: Dict[str, List[str]]) -> Dict[str, nx.Graph]:
    """Build small-world subgraph for each module."""
    sw = cfg.defaults.small_world
    k = int(sw.get("k_nearest", 6))
    p = float(sw.get("rewiring_p", 0.08))
    module_graphs: Dict[str, nx.Graph] = {}
    for m in cfg.modules:
        # Skip dormant modules
        if not getattr(m, 'activation', True):
            continue
        if m.id not in pop_map:
            continue
        nodes = pop_map[m.id]
        if len(nodes) < 4:
            g = nx.Graph()
            g.add_nodes_from(nodes)
            module_graphs[m.id] = g
            continue
        g = ring_lattice(nodes, k=min(k, max(2, (len(nodes)//8)*2)))
        rewire_ws(g, p)
        module_graphs[m.id] = g
    return module_graphs

def wire_inter_module(cfg: ConnectomeConfig,
                      pop_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Project sparse inter-module edges honoring required_links and density."""
    defaults = cfg.defaults.inter_module
    density_pct = float(defaults.get("projection_density_pct", 1.5))
    edges: set = set()

    # Required module-level edges (symmetric) - only for active modules
    active_modules = [m for m in cfg.modules if getattr(m, 'activation', True)]
    required_map = {m.id: [link for link in m.required_links 
                          if any(am.id == link and getattr(am, 'activation', True) for am in active_modules)] 
                   for m in active_modules}
    mod_edges = require_links([m.id for m in active_modules], required_map)
    for (a, b) in mod_edges:
        A, B = pop_map[a], pop_map[b]
        possible = len(A) * len(B)
        budget = max(1, int(possible * (density_pct / 100.0)))
        for _ in range(budget):
            u = random.choice(A)
            v = random.choice(B)
            edges.add(tuple(sorted([u, v])))

    return list(edges)

def assemble_graph(cfg: ConnectomeConfig) -> nx.Graph:
    """Assemble whole-brain graph."""
    _rng()
    pop_map = build_module_populations(cfg)
    intra = wire_intra_module(cfg, pop_map)
    G = nx.Graph()
    for g in intra.values():
        G = nx.compose(G, g)
    inter_edges = wire_inter_module(cfg, pop_map)
    G.add_edges_from(inter_edges)

    # Annotate node attributes per module (only active modules)
    node_attrs: Dict[str, Dict[str, Any]] = {}
    for m in cfg.modules:
        if getattr(m, 'activation', True) and m.id in pop_map:
            node_attrs[m.id] = dict(
                module=m.id,
                population=len(pop_map[m.id]),
                excitatory_ratio=m.excitatory_ratio
            )
    # Enforce E/I balance at module level (metadata)
    ei = cfg.policies.e_i_balance
    enforce_ei_balance(node_attrs,
                       min_inh=float(ei.get("min_inhibitory_ratio", 0.15)),
                       max_inh=float(ei.get("max_inhibitory_ratio", 0.30)))
    # Write node attrs down to individual neurons as tags (probabilistic assignment)
    for m in cfg.modules:
        if getattr(m, 'activation', True) and m.id in node_attrs:
            attrs = node_attrs[m.id]
            e_count = attrs["e_count"]
            nodes = [n for n in G.nodes() if str(n).startswith(f"{m.id}:")]
            random.shuffle(nodes)
            for i, nid in enumerate(nodes):
                t = "E" if i < e_count else "I"
                G.nodes[nid]["cell_type"] = t
                G.nodes[nid]["module"] = m.id

    # Topology repair nudges
    topo = cfg.policies.topology
    if topo.get("enforce_small_world", True):
        cc, apl = ensure_small_world(G,
                                     target_cc=float(topo.get("target_clustering_coeff", 0.15)),
                                     max_path=float(topo.get("max_avg_path_length", 4.5)))
        G.graph["avg_clustering"] = cc
        G.graph["avg_path_length"] = apl

    return G

def export_graph(cfg: ConnectomeConfig, G: nx.Graph):
    out_dir = cfg.exports.dir
    if "graphml" in cfg.exports.formats:
        nx.write_graphml(G, os.path.join(out_dir, "connectome.graphml"))
    if "json" in cfg.exports.formats:
        data = nx.node_link_data(G)
        with open(os.path.join(out_dir, "connectome.json"), "w") as f:
            json.dump(data, f, indent=2)

def export_manifests(cfg: ConnectomeConfig, G: nx.Graph):
    """Emit per-module IO manifests + gating and thalamic relay hints."""
    out_dir = cfg.exports.dir
    weights_cfg = cfg.policies.weights
    clamp_min = float(weights_cfg.get("clamp_min", 0.01))
    clamp_max = float(weights_cfg.get("clamp_max", 3.0))

    # Simple degree→weight mapping
    deg = dict(G.degree())
    max_deg = max(deg.values()) if deg else 1

    for m in cfg.modules:
        # Only generate manifests for active modules
        if not getattr(m, 'activation', True):
            continue
        nodes = [n for n in G.nodes() if str(n).startswith(f"{m.id}:")]
        outgoing = {}
        incoming = {}
        for n in nodes:
            for nb in G.neighbors(n):
                w = (deg.get(n, 1) / max_deg) ** 0.5
                w = clamp(w, clamp_min, clamp_max)
                src_mod = G.nodes[n].get("module")
                dst_mod = G.nodes[nb].get("module")
                if src_mod == m.id:
                    outgoing.setdefault(dst_mod, 0.0)
                    outgoing[dst_mod] += w
                if dst_mod == m.id:
                    incoming.setdefault(src_mod, 0.0)
                    incoming[src_mod] += w

        manifest = {
            "module_id": m.id,
            "label": m.label,
            "role": m.role,
            "population": len(nodes),
            "io_summary": {
                "incoming_by_module": incoming,
                "outgoing_by_module": outgoing
            },
            "gating": {
                "controlled_by_bg": m.id in [t for mm in cfg.modules for t in mm.gate_targets],
                "thalamic_relay": "THA" in m.required_links
            },
            "neuromodulators": cfg.defaults.neuromodulators,
            "sleep_triggers": cfg.metadata.get("sleep_triggers", {}),
            "routing_hints": {
                "must_link": m.required_links
            },
            "graph_files": {
                "graphml": "connectome/exports/connectome.graphml",
                "json": "connectome/exports/connectome.json"
            },
        }
        with open(os.path.join(out_dir, f"{m.id.lower()}_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

def compile_connectome(config_path: str = "connectome/connectome.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    G = assemble_graph(cfg)
    export_graph(cfg, G)
    export_manifests(cfg, G)
    summary = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_clustering": float(G.graph.get("avg_clustering", 0.0)),
        "avg_path_length": float(G.graph.get("avg_path_length", 0.0)),
        "exports_dir": cfg.exports.dir
    }
    with open(os.path.join(cfg.exports.dir, "build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def validate_connectome(config_path: str = "connectome/connectome.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    G = nx.read_graphml(os.path.join(cfg.exports.dir, "connectome.graphml"))
    # Basic checks
    problems: List[str] = []
    # Prevent full mesh
    if cfg.policies.topology.get("prevent_full_mesh", True):
        possible = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
        if G.number_of_edges() > 0.25 * possible:
            problems.append("Graph too dense; likely approaching full mesh.")
    # Required module links
    mod_nodes = {m.id: [n for n in G.nodes() if str(n).startswith(f"{m.id}:")] for m in cfg.modules}
    for m in cfg.modules:
        for req in m.required_links:
            a = m.id; b = req
            # Ensure at least one inter-module edge
            found = False
            for u in mod_nodes[a]:
                if found: break
                for v in mod_nodes.get(b, []):
                    if G.has_edge(u, v):
                        found = True
                        break
            if not found:
                problems.append(f"Missing required projection between modules {a}<->{b}")

    report = {"valid": len(problems) == 0, "problems": problems}
    with open(os.path.join(cfg.exports.dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report

def apply_sleep_gating(config_path: str = "connectome/connectome.yaml"):
    cfg = load_config(config_path)
    st = cfg.metadata.get("sleep_triggers", {})
    asleep = read_telemetry_sleep_flag(
        cog=st.get("cognitive_load_pct", 85),
        mem=st.get("memory_usage_pct", 80),
        err=st.get("error_rate_pct", 15),
    )
    with open(os.path.join(cfg.exports.dir, "state.json"), "w") as f:
        json.dump({"sleeping": asleep, "ts": time.time()}, f, indent=2)
