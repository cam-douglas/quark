# connectome/README.md
# Connectome Manager

A declarative → compiled connectome pipeline for your Cursor brain-sim. It keeps your model
"connected to itself" with biologically-inspired constraints and **auto-rebuilds** on any repo change.

## What it does
- Reads `connectome/connectome.yaml` (single source of truth)
- Builds **small-world** intra-module populations + sparse inter-module projections
- Enforces required module links (PFC↔THA, BG→(WM,THA,PFC), PFC↔WM/DMN/HIP/ATT, etc.)
- Writes:
  - `connectome/exports/connectome.graphml` and `connectome/exports/connectome.json`
  - Per-module I/O manifests: `connectome/exports/<mod>_manifest.json`
  - Validation report + build summary
  - `state.json` reflecting sleep triggers (read from `runtime/telemetry.json`)

## Install
```bash
pip install -r connectome/requirements.txt
