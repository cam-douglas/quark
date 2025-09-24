#!/usr/bin/env python3
"""
Rules Index Validator Module
============================
Validates the rules index and ensures all rules are properly configured.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class RulesValidator:
    """Validates rules index and configuration."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.index_json = self.workspace_root / "repo_indexes" / "RULES_INDEX.json"
        self.cursor_dir = self.workspace_root / ".cursor" / "rules"
        self.quark_dir = self.workspace_root / ".quark" / "rules"
        self.validation_anchor = self.workspace_root / "state" / "tasks" / "validation"
        self.failures: List[str] = []
    
    def validate_rules(self) -> Dict[str, Any]:
        """Run complete rules validation."""
        self.failures = []
        
        # Load index
        data = self._load_index()
        rules = data.get("rules", []) if isinstance(data, dict) else []
        
        if not rules:
            self._fail("No rules found in index")
        else:
            self._check_paths(rules)
            self._check_metadata(rules)
            self._check_validation_anchoring(rules)
        
        return {
            "success": len(self.failures) == 0,
            "failures": self.failures,
            "rules_count": len(rules),
            "message": "Rules validation passed" if not self.failures else "Rules validation failed"
        }
    
    def _fail(self, msg: str) -> None:
        """Record a validation failure."""
        self.failures.append(msg)
    
    def _load_index(self) -> Dict:
        """Load the rules index JSON."""
        if not self.index_json.exists():
            self._fail(f"Missing index: {self.index_json}")
            return {}
        
        try:
            return json.loads(self.index_json.read_text())
        except Exception as e:
            self._fail(f"Invalid JSON in {self.index_json}: {e}")
            return {}
    
    def _check_paths(self, entries: List[Dict]) -> None:
        """Check that all indexed paths exist."""
        for r in entries:
            p = self.workspace_root / r.get("path", "")
            if not p.exists():
                self._fail(f"Indexed path missing: {p}")
            
            if r.get("type") == "cursor" and self.cursor_dir not in p.parents:
                self._fail(f"Cursor rule not under .cursor/rules: {p}")
            
            if r.get("type") == "quark" and self.quark_dir not in p.parents:
                self._fail(f"Quark rule not under .quark/rules: {p}")
    
    def _check_metadata(self, entries: List[Dict]) -> None:
        """Check rule metadata."""
        for r in entries:
            # Type validation
            if r.get("type") not in {"cursor", "quark"}:
                self._fail(f"Unknown rule type: {r}")
    
    def _check_validation_anchoring(self, entries: List[Dict]) -> None:
        """Check that validation rules reference the anchor directory."""
        keywords = ("VALIDATION", "GOLDEN RULE", "GLOBAL PRIORITY")
        
        for r in entries:
            title = (r.get("title") or "").upper()
            
            if any(k in title for k in keywords):
                p = self.workspace_root / r["path"]
                
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    self._fail(f"Cannot read rule {p}: {e}")
                    continue
                
                anchor_str = str(self.validation_anchor)
                if anchor_str not in text:
                    self._fail(f"Validation rule missing anchor reference: {p} (expected path: {anchor_str})")
    
    def get_rules_summary(self) -> Dict[str, Any]:
        """Get a summary of all rules."""
        data = self._load_index()
        rules = data.get("rules", []) if isinstance(data, dict) else []
        
        cursor_rules = [r for r in rules if r.get("type") == "cursor"]
        quark_rules = [r for r in rules if r.get("type") == "quark"]
        
        return {
            "total_rules": len(rules),
            "cursor_rules": len(cursor_rules),
            "quark_rules": len(quark_rules),
            "validation_rules": len([r for r in rules if "VALIDATION" in (r.get("title", "").upper())]),
            "always_apply": len([r for r in rules if r.get("alwaysApply", False)])
        }
    
    def sync_rules(self) -> Dict[str, Any]:
        """Ensure cursor and quark rules are in sync."""
        synced = []
        failed = []
        
        # Check each cursor rule has a quark counterpart
        if self.cursor_dir.exists():
            for cursor_file in self.cursor_dir.glob("*.mdc"):
                quark_file = self.quark_dir / cursor_file.name
                
                if not quark_file.exists():
                    # Copy cursor to quark
                    try:
                        quark_file.parent.mkdir(parents=True, exist_ok=True)
                        quark_file.write_text(cursor_file.read_text())
                        synced.append(f"Copied {cursor_file.name} to .quark/rules/")
                    except Exception as e:
                        failed.append(f"Failed to sync {cursor_file.name}: {e}")
                else:
                    # Check if content matches
                    if cursor_file.read_text() != quark_file.read_text():
                        try:
                            quark_file.write_text(cursor_file.read_text())
                            synced.append(f"Updated {cursor_file.name} in .quark/rules/")
                        except Exception as e:
                            failed.append(f"Failed to update {cursor_file.name}: {e}")
        
        return {
            "synced": synced,
            "failed": failed,
            "success": len(failed) == 0
        }
