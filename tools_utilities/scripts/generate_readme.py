#!/usr/bin/env python3
"""Generate a technical whitepaper-style README.md from the Quark State System
and repository structure.

Sections:
- Executive summary: What Quark is
- Current state (from state/quark_state_system/QUARK_STATE.md if present)
- Benchmarks & capabilities (tests and docs references)
- Architecture overview (brain & ML modules)
- Detailed project summary & links

Behavior:
- Safe to run locally and in CI. No external calls.
- If QUARK_STATE.md is missing, proceeds with placeholder.

Integration: Support utilities used by brain/state; indirectly integrated where imported.
Rationale: Shared helpers (performance, IO, streaming) used across runtime components.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / "state/quark_state_system/quark_state_system.md"
# Prefer the user's absolute README path if it exists; otherwise, use repo root.
ABS_README = Path("/Users/camdouglas/quark/README.md")
README_FILE = ABS_README if ABS_README.exists() else (REPO_ROOT / "README.md")


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def extract_state_summary(state_md: str) -> str:
    if not state_md:
        return (
            "Current Development Stage: (not found)\n"
            "Overall Progress: (unknown)\n"
            "Next Major Milestone: (unknown)\n"
        )
    # Lightweight extraction of key headings/lines
    lines = state_md.splitlines()
    keep: List[str] = []
    for ln in lines:
        if ln.strip().startswith("**Current Development Stage**") or ln.strip().startswith("**Overall Progress**"):
            keep.append(ln)
        if ln.strip().startswith("### **Phase 4") or ln.strip().startswith("### **Roadmap Phase**"):
            keep.append(ln)
        if ln.strip().startswith("#### **Roadmap Target Deliverables**"):
            keep.append(ln)
    return "\n".join(keep) if keep else state_md[:1000]


def summarize_benchmarks() -> str:
    # Point to canonical test locations. We avoid executing tests here.
    return (
        "- tests/cognitive_benchmarks/: working memory, decision-making, reversal learning\n"
        "- tests/testing_frameworks/: automated validation, live streaming, system health\n"
        "- tests/simulations/: embodiment and physics hooks\n"
        "- CI: .github/workflows/ci.yml (runs smoke + benchmark stubs)\n"
    )


def summarize_architecture() -> str:
    parts: List[str] = []
    parts.append("- brain/: multi-scale brain architecture (embodiment, learning, neural_core)\n")
    parts.append("- brain/modules/: integration modules (alphagenome, mathematical, etc.)\n")
    parts.append("- brain/simulator/: runtime simulators and entrypoints\n")
    parts.append("- data/: models, datasets, and simulation frameworks (heavy artifacts ignored)\n")
    parts.append("- state/quark_state_system/: autonomous agent, task loader, state orchestration\n")
    parts.append("- docs/: specifications, roadmaps, historical and integration guides\n")
    parts.append("- tests/: benchmarks, visualizations, and CI-friendly checks\n")
    return "".join(parts)


def build_readme(state_summary: str) -> str:
    return f"""
# 🚀 QUARK: Technical Whitepaper

## Overview

Quark is a modular brain-simulation and machine-learning platform spanning molecular genetics → tissue morphogenesis → micro-circuits → cognition. It emphasizes biological fidelity, rigorous testing, and reproducible workflows governed by the Quark State System.

## Current State (via Quark State System)

{state_summary}

### Operate the State System

```bash
python QUARK_STATE_SYSTEM.py status
python QUARK_STATE_SYSTEM.py recommendations
python QUARK_STATE_SYSTEM.py sync
```

## Benchmarks & Capabilities

{summarize_benchmarks()}

## Architecture

{summarize_architecture()}

## Development & Reproducibility

- Python 3.11, pinned deps in requirements.txt
- CI: GitHub Actions in .github/workflows/ (tests and artifacts)
- Heavy artifacts are ignored via .gitignore (models, large outputs)
- Deterministic runs preferred; seeds where applicable

## Documentation Index

See docs/INDEX.md for a curated map of specifications, guides, and historical reports.

"""


def main() -> None:
    state_md = read_text_safe(STATE_FILE)
    state_summary = extract_state_summary(state_md)
    content = build_readme(state_summary)
    README_FILE.write_text(content, encoding="utf-8")
    print(f"Wrote README: {README_FILE}")


if __name__ == "__main__":
    main()

