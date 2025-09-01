
# 🚀 QUARK: Technical Whitepaper

## Overview

Quark is a modular brain-simulation and machine-learning platform spanning molecular genetics → tissue morphogenesis → micro-circuits → cognition. It emphasizes biological fidelity, rigorous testing, and reproducible workflows governed by the Quark State System.

## Current State (via Quark State System)

**Current Development Stage**: STAGE N4 IN PROGRESS
**Overall Progress**: 5% (Stage N4 In Progress)
### **Phase 4: AGI Capabilities & Full Validation** 🚀
#### **Roadmap Target Deliverables**:
### **Phase 4: AGI Capabilities & Full Validation** 🚀 READY TO START

### Operate the State System

```bash
python QUARK_STATE_SYSTEM.py status
python QUARK_STATE_SYSTEM.py recommendations
python QUARK_STATE_SYSTEM.py sync
```

## Benchmarks & Capabilities

- tests/cognitive_benchmarks/: working memory, decision-making, reversal learning
- tests/testing_frameworks/: automated validation, live streaming, system health
- tests/simulations/: embodiment and physics hooks
- CI: .github/workflows/ci.yml (runs smoke + benchmark stubs)


## Architecture

- brain/: multi-scale brain architecture (embodiment, learning, neural_core)
- brain/modules/: integration modules (alphagenome, mathematical, etc.)
- brain/simulator/: runtime simulators and entrypoints
- data/: models, datasets, and simulation frameworks (heavy artifacts ignored)
- state/quark_state_system/: autonomous agent, task loader, state orchestration
- docs/: specifications, roadmaps, historical and integration guides
- tests/: benchmarks, visualizations, and CI-friendly checks


## Development & Reproducibility

- Python 3.11, pinned deps in requirements.txt
- CI: GitHub Actions in .github/workflows/ (tests and artifacts)
- Heavy artifacts are ignored via .gitignore (models, large outputs)
- Deterministic runs preferred; seeds where applicable

## Documentation Index

See docs/INDEX.md for a curated map of specifications, guides, and historical reports.

## Contributing

## Memory Snapshots & Persistence

Quark now preserves its long-term and episodic memories across restarts.

* Snapshots live in `state/memory/` as gzip-compressed JSON files (`episodic.json.gz`, `ltm.json.gz`).
* Every run of `BrainSimulator` loads any existing snapshots on start-up.
* During execution the simulator automatically calls `MemoryPersistenceManager.save_all()` every **N** simulation steps.  Configure the cadence with the environment variable:

```bash
export QUARK_PERSIST_EVERY=200   # save every 200 steps
```

* An atomic "write-then-rename" strategy ensures snapshots are never partially written; each file also carries a CRC-32 checksum and schema version for integrity checking.

If you need a clean slate, simply remove the files in `state/memory/` before launching Quark.

## Resource Buckets

The `ResourceManager` now exposes simple *buckets* that gate concurrent heavy tasks:

| Bucket | Default Tokens | Use-cases |
|--------|---------------|-----------|
| `nlp` | 2 | TF-IDF computation, LLM calls |
| `io`  | 4 | Snapshot save/load, large dataset copy |
| `background` | 8 | Low-priority jobs |

Adjust limits with environment variables before launching Quark:

```bash
export QUARK_RM_NLP_LIMIT=1   # allow only one NLP-heavy task at a time
export QUARK_RM_IO_LIMIT=2
export QUARK_RM_BG_LIMIT=6
```

For instrumentation:

```python
from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager
print(ResourceManager.get_default().get_stats())
```

Output:

```json
{"nlp": {"capacity": 2, "available": 1}, "io": {"capacity": 4, "available": 3}, ...}
```

### Natural-language training / fine-tuning

Quark can now launch its own streaming training pipeline with plain English:

```
train quark          # interactive bucket/prefix selection, streams data from S3
fine-tune quark      # same, launches fine-tuning
```

Behind the scenes:

1. BrainSimulator → KnowledgeHub detects the phrase.
2. KnowledgeHub delegates to ResourceManager → `quark_cli.py`.
3. CLI prompts for bucket alias (e.g. “tokyo bucket”), lists available dataset prefixes, shows size & ETA, then starts the streaming trainer with live progress.

Check `docs/STREAMING_DATA_README.md` for architecture, env-vars, and troubleshooting tips.

