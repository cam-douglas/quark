"""Rules Orchestrator

Watches the repository for file changes and enforces Cursor/Quark rules
discovered via `cursor_rules_loader`. Each rule file (*.mdc) is treated as a
policy bundle that may declare simple metadata in YAML front matter.

This module provides:
  - Rule discovery and caching
  - File change filtering against per-rule glob patterns
  - Fast validators (regex/keywords) and script hooks (optional)
  - Diagnostics emission for CI or editor integration

CLI entry point: see `rules_orchestrator:main`.
"""
from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .cursor_rules_loader import discover_rule_files


@dataclasses.dataclass
class Diagnostic:
    path: str
    line: int
    column: int
    severity: str  # advice | warn | error
    message: str
    rule_id: str


@dataclasses.dataclass
class LoadedRule:
    rule_id: str
    description: str
    severity: str
    globs: List[str]
    patterns: List[Tuple[str, str]]  # (name, regex)


def _parse_front_matter(text: str) -> Tuple[Dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    # Strip leading '---' block
    _, _, rest = text.partition("---\n")
    meta_raw, sep, content = rest.partition("---\n")
    if sep == "":
        # No closing delimiter; treat as no-metadata
        return {}, text
    try:
        meta = yaml.safe_load(meta_raw) or {}
    except Exception:
        meta = {}
    return meta, content


def load_rules(repo_root: Path) -> List[LoadedRule]:
    rules: List[LoadedRule] = []
    for path in discover_rule_files(repo_root):
        text = path.read_text(encoding="utf-8")
        meta, _ = _parse_front_matter(text)

        rule_id = meta.get("id") or path.stem
        description = meta.get("description") or f"Rules from {path.name}"
        severity = meta.get("severity", "warn")
        globs = list(meta.get("globs", ["**/*"]))

        # Heuristic: extract simple anti-overconfidence patterns if declared
        patterns_cfg = meta.get("patterns", [])
        patterns: List[Tuple[str, str]] = []
        for item in patterns_cfg:
            if not isinstance(item, dict):
                continue
            name = item.get("name", rule_id)
            regex = item.get("regex")
            if isinstance(regex, str):
                patterns.append((name, regex))

        rules.append(
            LoadedRule(
                rule_id=rule_id,
                description=description,
                severity=severity,
                globs=globs,
                patterns=patterns,
            )
        )
    return rules


def _file_matches_globs(file_path: str, globs: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(file_path, g) for g in globs)


def run_validators(
    repo_root: Path, changed_files: Sequence[str], rules: Sequence[LoadedRule]
) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    for file_rel in changed_files:
        full_path = repo_root / file_rel
        if not full_path.is_file():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for rule in rules:
            if not _file_matches_globs(file_rel, rule.globs):
                continue

            # Pattern validators
            for pat_name, regex in rule.patterns:
                try:
                    compiled = re.compile(regex, flags=re.MULTILINE)
                except re.error:
                    continue
                for m in compiled.finditer(content):
                    # Rough line/column calculation
                    line = content.count("\n", 0, m.start()) + 1
                    last_nl = content.rfind("\n", 0, m.start())
                    column = m.start() - (0 if last_nl < 0 else last_nl + 1) + 1
                    diagnostics.append(
                        Diagnostic(
                            path=file_rel,
                            line=line,
                            column=column,
                            severity=rule.severity,
                            message=f"{pat_name} matched disallowed pattern: {m.group(0)[:80]}",
                            rule_id=rule.rule_id,
                        )
                    )

    return diagnostics


def _list_all_repo_files(repo_root: Path) -> List[str]:
    files: List[str] = []
    for p in repo_root.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            files.append(str(p.relative_to(repo_root)))
    return files


def _print_diagnostics(diagnostics: Sequence[Diagnostic], as_json: bool) -> int:
    if as_json:
        print(json.dumps([dataclasses.asdict(d) for d in diagnostics], indent=2))
    else:
        for d in diagnostics:
            print(f"{d.path}:{d.line}:{d.column}: {d.severity.upper()} {d.rule_id}: {d.message}")
    # Non-zero exit on error severity
    return 1 if any(d.severity == "error" for d in diagnostics) else 0


def _watch_loop(repo_root: Path, rules: Sequence[LoadedRule], interval_s: float) -> int:
    mtimes: Dict[str, float] = {}
    try:
        while True:
            all_files = _list_all_repo_files(repo_root)
            changed: List[str] = []
            for rel in all_files:
                p = repo_root / rel
                try:
                    mtime = p.stat().st_mtime
                except FileNotFoundError:
                    continue
                prev = mtimes.get(rel)
                if prev is None or mtime > prev:
                    changed.append(rel)
                mtimes[rel] = mtime

            if changed:
                diags = run_validators(repo_root, changed, rules)
                _print_diagnostics(diags, as_json=False)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quark Rules Orchestrator")
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--changed", nargs="*", help="Changed files (default: scan all)")
    parser.add_argument("--json", action="store_true", help="Output diagnostics as JSON")
    parser.add_argument("--advisory", action="store_true", help="Never exit non-zero")
    parser.add_argument("--watch", action="store_true", help="Watch for file changes and re-run")
    parser.add_argument("--interval", type=float, default=0.8, help="Watch poll interval seconds")

    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)

    rules = load_rules(repo_root)
    if not rules:
        # No rules found; nothing to enforce
        return 0

    if args.watch:
        # Always advisory in watch mode
        return _watch_loop(repo_root, rules, interval_s=args.interval)

    changed = args.changed or _list_all_repo_files(repo_root)
    diags = run_validators(repo_root, changed, rules)
    code = _print_diagnostics(diags, as_json=args.json)
    if args.advisory:
        return 0
    return code


if __name__ == "__main__":
    sys.exit(main())

