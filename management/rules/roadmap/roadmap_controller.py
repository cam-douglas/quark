

"""
Integration: Not simulator-integrated; governance, rules, and roadmap processing.
Rationale: Consumed by state system to guide behavior; no direct simulator hooks.
"""
# management/rules/roadmaps/roadmap_controller.py
"""Roadmap Controller

Centralised loader that aggregates every roadmap YAML/Markdown file under
``management/rules/roadmaps/`` and exposes a single helper for the state system.
"""
from pathlib import Path
from typing import List, Dict
import yaml

ROOT_DIR = Path(__file__).resolve().parents[0]

# --- New constants ---------------------------------------------------------
MASTER_ROADMAP = ROOT_DIR / "MASTER_ROADMAP.md"
INDEX_FILE = ROOT_DIR / "ROADMAPS_INDEX.md"

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

# ---------------------------------------------------------------------------
# New helpers (Phase-2 additions)
# ---------------------------------------------------------------------------


def _extract_status(line: str) -> str:
    """Return status emoji / word from a line (✅, 🚧, 📋 Planned …)."""
    # crude: look for emoji or keywords
    if "✅" in line:
        return "done"
    if "🚧" in line:
        return "progress"
    if "📋" in line or "Planned" in line:
        return "planned"
    return "unknown"


def _normalize_status(value: str) -> str:
    """Normalize free-form status strings from the index to canonical labels.

    Canonical set: {"done", "progress", "planned", "—", "unknown"}
    """
    s = (value or "").strip()
    # Map em-dash variants and ASCII dashes to em dash
    if s in {"—", "-", "--", "---"}:
        return "—"
    low = s.lower()
    if "✅" in s or "done" in low or "integration status" in low:
        return "done"
    if "🚧" in s or "progress" in low or "in progress" in low:
        return "progress"
    if "📋" in s or "planned" in low or "preparation" in low:
        return "planned"
    return "unknown"


def get_master_roadmap() -> Dict:
    """Parse the *master_roadmap.md* for quick metadata (title, statuses)."""
    if not MASTER_ROADMAP.exists():
        return {}
    title = "Master Roadmap"
    statuses = []
    with MASTER_ROADMAP.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.lstrip().startswith("#"):
                title = ln.lstrip("#").strip()
                continue
            if any(tok in ln for tok in ("✅", "🚧", "📋")):
                statuses.append({"line": ln.strip(), "status": _extract_status(ln)})
    return {"path": str(MASTER_ROADMAP), "title": title, "statuses": statuses}


def get_index() -> List[Dict]:
    """Return the table rows from ROADMAPS_INDEX.md as dicts."""
    if not INDEX_FILE.exists():
        return []
    rows = []
    in_table = False
    with INDEX_FILE.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("| # "):
                in_table = True
                continue
            if in_table and ln.startswith("|"):
                parts = [c.strip() for c in ln.strip().strip('|').split('|')]
                if len(parts) >= 4:
                    # Skip the markdown separator row like "| 0 | --- | --- | --- |"
                    if parts[1] == "---":
                        continue
                    idx = int(parts[0]) if parts[0].isdigit() else 0
                    path = parts[2]
                    title = parts[1]
                    # Ensure a canonical title for the master roadmap row
                    if path.endswith("master_roadmap.md"):
                        title = "Master Roadmap"
                    rows.append({
                        "index": idx,
                        "title": title,
                        "path": path,
                        "status": _normalize_status(parts[3]),
                    })
            elif in_table and not ln.startswith("|"):
                break
    return rows


def status_snapshot() -> Dict[str, str]:
    """Return mapping title → status from index (fallback to master)."""
    snapshot = {r["title"]: r["status"] for r in get_index()}
    if not snapshot and MASTER_ROADMAP.exists():
        mr = get_master_roadmap()
        snapshot[mr.get("title", "Master Roadmap")] = ";".join(
            _normalize_status(s["status"]) for s in mr.get("statuses", [])
        ) or "unknown"
    return snapshot

# ---------------------------------------------------------------------------
# Back-compat: legacy helpers
# ---------------------------------------------------------------------------

def get_roadmap_status_map() -> Dict[str, str]:  # pragma: no cover
    """Legacy alias – maps to status_snapshot() for older callers."""
    return status_snapshot()


if __name__ == "__main__":
    import json
    import sys
    json.dump(get_all_roadmaps(), sys.stdout, indent=2)
