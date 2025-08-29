#!/usr/bin/env python3
"""
Dependency & Reference Graph Builder (Phase 0 of Quark re-org)

Scans the repository (excluding heavy directories) to collect:
  • Python imports → directed edges (from_file → imported_module)
  • Markdown relative links → reference edges (from_doc → target_path)
Outputs two artifacts:
  1. JSON adjacency list (dependency_graph.json)
  2. Mermaid diagram file (dependency_graph_mermaid.md)

Run from repo root:
    python tools_utilities/audit/dependency_graph_builder.py

The heavy directories are excluded by default but can be overridden with
command-line flags.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

RE_HEAVY_DIRS = re.compile(r"^(datasets|external|models|mlruns|state_snapshots|venv)(/|$)")
PY_IMPORT_RE = re.compile(r"^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.MULTILINE)
MD_LINK_RE = re.compile(r"\]\(([^)]+)\)")

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # /quark
OUTPUT_DIR = PROJECT_ROOT / "audit_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

JsonGraph = Dict[str, List[str]]


def should_skip(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return bool(RE_HEAVY_DIRS.match(rel))


def scan_python_file(path: Path, graph: JsonGraph) -> None:
    src = path.read_text(encoding="utf-8", errors="ignore")
    for match in PY_IMPORT_RE.finditer(src):
        mod = match.group(1) or match.group(2)
        if mod:
            graph.setdefault(str(path), []).append(mod)


def scan_markdown_file(path: Path, graph: JsonGraph) -> None:
    src = path.read_text(encoding="utf-8", errors="ignore")
    for link in MD_LINK_RE.findall(src):
        if link.startswith("http"):
            continue  # external link
        graph.setdefault(str(path), []).append(link)


def build_graph() -> JsonGraph:
    graph: JsonGraph = {}
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_dir() and should_skip(path):
            # fast-skip entire heavy dir
            dirs_to_skip: Set[Path] = {path}
            # prune generator by using .rglob filtering side effect – we can't easily prune,
            # so we just continue; skip will apply to contained files too.
        if path.is_file():
            if should_skip(path):
                continue
            if path.suffix == ".py":
                scan_python_file(path, graph)
            elif path.suffix.lower() in {".md", ".rst"}:
                scan_markdown_file(path, graph)
    return graph


def export_graph(graph: JsonGraph) -> None:
    json_path = OUTPUT_DIR / "dependency_graph.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    print(f"✅ JSON dependency graph written to {json_path.relative_to(PROJECT_ROOT)}")

    # Mermaid
    mermaid_lines = ["```mermaid", "graph TD;"]
    for src, targets in graph.items():
        src_node = src.replace("/", "_")
        for tgt in targets:
            tgt_node = tgt.replace("/", "_")
            mermaid_lines.append(f"    {src_node} --> {tgt_node}")
    mermaid_lines.append("```\n")
    mermaid_path = OUTPUT_DIR / "dependency_graph_mermaid.md"
    mermaid_path.write_text("\n".join(mermaid_lines), encoding="utf-8")
    print(f"✅ Mermaid diagram saved to {mermaid_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build code/documentation dependency graph")
    parser.parse_args()  # no flags yet, placeholder for future extension
    graph = build_graph()
    export_graph(graph)


if __name__ == "__main__":
    main()
