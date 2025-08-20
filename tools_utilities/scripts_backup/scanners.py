from __future__ import annotations
import fnmatch, os, pathlib
from typing import Dict, List, Tuple
from ................................................config import ROOT, SETTINGS

def _match_any(path: str, globs: List[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)

def scan_files() -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    for p, _, files in os.walk(ROOT):
        for f in files:
            rel = os.path.relpath(os.path.join(p,f), ROOT)
            if _match_any(rel, SETTINGS["ignore_globs"]): continue
            if _match_any(rel, SETTINGS["scan_globs"]):
                paths.append((ROOT/rel).resolve())
    return sorted(paths)

def read_text_safe(p: pathlib.Path, limit_bytes: int = 200_000) -> str:
    try:
        data = p.read_bytes()[:limit_bytes]
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
