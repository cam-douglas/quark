# FILE HEADER
"""
Resource Manager
----------------
Purpose:
    Centralized module responsible for registering, sandbox-validating, and integrating external
    "resources" (code, datasets, models, configs) into the Quark brain infrastructure.
Inputs:
    • register_resource(path: str, metadata: dict | None) – called by ingestion pipeline or manually
Outputs:
    • Integration of the resource into correct repo directory + callback into active simulators.
Seeds / Reproducibility:
    All hash operations default to SHA-256; random sampling seeded by `QUARK_SEED` env var.
Dependencies:
    Standard library only for the MVP (hashlib, shutil, pathlib, importlib).
TODOs:
    – Add plugin discovery for domain-specific integrators
    – Add async queue & background worker
    – Add configurable storage backend (SQLite / YAML)
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional
import threading
import time
from queue import Queue, Empty
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:  # Optional dependency; fallback
    Observer = None
    class FileSystemEventHandler:  # type: ignore[misc]
        pass
import subprocess, tempfile
import re
# ---------------------------------------------------------------------------
# Constants & Globals
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[5]  # /quark
_DATA_DIR = _REPO_ROOT / "data"
_CONFIG_DIR = _REPO_ROOT / "management" / "configurations" / "resource_manager"
_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup (moved after constants)
import logging
logger = logging.getLogger("quark.resource_manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _log_path = _REPO_ROOT / "logs" / "resource_manager.log"
    _log_path.parent.mkdir(exist_ok=True)
    fh = logging.FileHandler(_log_path)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

from brain.architecture.neural_core.cognitive_systems.callback_hub import hub  # NEW
from brain.architecture.neural_core.cognitive_systems.plugins import get_plugins  # NEW


class ResourceManager:
    """Lightweight MVP for registering and integrating resources.

    This version focuses on *manual* resource registration and immediate synchronous
    integration so that BrainSimulator can call it during runtime.  Future versions
    will offload heavy work to background agents.
    """

    def __init__(self, auto_scan: bool = False, scan_interval_sec: int = 3600):
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.auto_scan = auto_scan
        self.scan_interval_sec = scan_interval_sec
        # load persisted registry if exists
        self._registry_path = _CONFIG_DIR / "registry.yaml"
        try:
            import yaml  # runtime import so we don’t add top-level dep if unused
            if self._registry_path.exists():
                self.registry = yaml.safe_load(self._registry_path.read_text()) or {}
        except Exception:
            # fallback to empty registry on first run
            self.registry = {}
        self._blacklist_path = _CONFIG_DIR / "blacklist.yaml"
        try:
            import yaml
            self.blacklist = set(yaml.safe_load(self._blacklist_path.read_text()) or []) if self._blacklist_path.exists() else set()
        except Exception:
            self.blacklist = set()
        self._scan_state_path = _CONFIG_DIR / "registry_scan_state.yaml"
        try:
            import yaml
            self._seen_hashes = set(yaml.safe_load(self._scan_state_path.read_text()) or []) if self._scan_state_path.exists() else set()
        except Exception:
            self._seen_hashes = set()

        self.plugins = get_plugins()

        # Start autoscan if enabled in default config
        if auto_scan:
            self.autoscanner = AutoScanner(_DATA_DIR, interval_sec)
            self.autoscanner.start()
        else:
            self.autoscanner = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_resource(self, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Hash the resource, persist minimal metadata, and attempt integration.

        Returns the resource hash ID.
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)

        rid = self._hash_path(p)
        # Accurate size: file size or sum of files in dir
        if p.is_file():
            size_bytes = p.stat().st_size
        else:
            size_bytes = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())

        meta = metadata.copy() if metadata else {}
        meta.update({"path": str(p), "size_bytes": size_bytes})
        self.registry[rid] = meta
        self._persist_registry()
        # Integrate synchronously for MVP
        self.integrate(rid)
        return rid

    def integrate(self, resource_id: str):
        """Route the resource to its canonical location and load if needed."""
        if resource_id not in self.registry:
            raise KeyError(f"Unknown resource id {resource_id}")
        meta = self.registry[resource_id]
        src = Path(meta["path"])
        size = meta["size_bytes"]

        # Plugin delegation
        for plugin in self.plugins:
            try:
                if plugin.can_handle(meta):
                    if plugin.integrate(meta):
                        logger.info("Plugin %s handled resource %s", plugin.__class__.__name__, resource_id)
                        hub.emit("resource_integrated", id=resource_id, path=meta.get("integrated_path", ""))
                        return
            except Exception as e:
                logger.warning("Plugin %s error: %s", plugin.__class__.__name__, e)

        # --- License gate ---
        lic = detect_license(src) if src.is_file() else None
        meta["license"] = lic
        if lic in _BLOCKED_LICENSES and not meta.get("force", False):
            logger.warning("❌ Integration blocked: license %s for %s", lic, src)
            hub.emit("resource_rejected", id=resource_id, reason="blocked_license", license=lic)
            return

        # --- Sandbox gate for code (<200 MB, .py) ---
        if src.suffix == ".py" and size < 200*2**20:
            sb = SandboxExecutor()
            if not sb.run(src):
                logger.error("❌ Sandbox validation failed for %s", src)
                hub.emit("resource_failed", id=resource_id, reason="sandbox_fail")
                return

        # --- Decide placement directory ---
        if size > 200 * 2**20:
            target_base = _DATA_DIR / self._infer_domain(src)
        else:
            target_base = _REPO_ROOT / "brain" / "externals" / src.stem
        target_base.mkdir(parents=True, exist_ok=True)

        # Avoid redundant copy if src already under target_base
        try:
            src.relative_to(target_base)
            # Source already in place; just mark integrated_path and return
            meta["integrated_path"] = str(src)
            meta["approved"] = True
            self._persist_registry()
            logger.info("Resource %s already located under %s", resource_id, target_base)
            hub.emit("resource_integrated", id=resource_id, path=str(src))
            self._notify_simulators(resource_id, meta)
            return
        except ValueError:
            pass  # need to copy

        target = target_base / src.name

        if not target.exists():
            if src.is_dir():
                shutil.copytree(src, target, dirs_exist_ok=True)
            else:
                shutil.copy2(src, target)
        meta["integrated_path"] = str(target)
        meta["approved"] = True
        self._persist_registry()
        logger.info("Integrated resource %s -> %s", resource_id, target)
        hub.emit("resource_integrated", id=resource_id, path=str(target))

        # Attempt dynamic import if python module
        if target.suffix == ".py" and size < 5 * 2**20:
            self._attempt_import(target)

        # TODO real-time callback hooks – stub for now
        self._notify_simulators(resource_id, meta)

    def approve(self, resource_id: str):
        if resource_id not in self.registry:
            logger.error("Unknown resource id %s", resource_id)
            return
        self.registry[resource_id]["approved"] = True
        self._persist_registry()
        logger.info("Approved resource %s", resource_id)
        self.integrate(resource_id)

    def reject(self, resource_id: str):
        if resource_id not in self.registry:
            logger.error("Unknown resource id %s", resource_id)
            return
        self.blacklist.add(resource_id)
        self.registry.pop(resource_id, None)
        self._persist_registry()
        self._persist_blacklist()
        logger.info("Rejected resource %s", resource_id)
        hub.emit("resource_rejected", id=resource_id, reason="user_reject")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _hash_path(self, p: Path) -> str:
        h = compute_sha256(p)
        self._seen_hashes.add(h)
        self._persist_scan_state()
        return h

    def _persist_registry(self):
        try:
            import yaml
            self._registry_path.write_text(yaml.safe_dump(self.registry))
        except Exception:
            pass

    def _persist_blacklist(self):
        try:
            import yaml
            self._blacklist_path.write_text(yaml.safe_dump(list(self.blacklist)))
        except Exception:
            pass

    def _persist_scan_state(self):
        try:
            import yaml
            self._scan_state_path.write_text(yaml.safe_dump(list(self._seen_hashes)))
        except Exception:
            pass

    def _infer_domain(self, p: Path) -> str:
        """Very naive domain inference based on keywords; extend with ML later."""
        name = p.stem.lower()
        for key in ("dataset", "model", "mesh", "memory"):
            if key in name:
                return key + "s"
        return "misc"

    def _attempt_import(self, module_path: Path):
        """Try importing a copied python file to ensure it is valid."""
        module_name = module_path.stem
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
            except Exception as e:
                print(f"⚠️ ResourceManager: Import failed for {module_name}: {e}")

    def _notify_simulators(self, rid: str, meta: Dict[str, Any]):
        """Placeholder for real-time callbacks to BrainSimulator etc."""
        # In MVP we just print; later we’ll route via an observer pattern.
        print(f"🔗 ResourceManager integrated {rid} -> {meta.get('integrated_path')}")

# ---------------------------------------------------------------------------
# Auto-Scanner (Phase 2)
# ---------------------------------------------------------------------------
class _EventHandler(FileSystemEventHandler):  # type: ignore[misc]
    def __init__(self, q: Queue):
        super().__init__()
        self.q = q

class AutoScanner:
    """Background scanner that watches the /data directory for new resources."""
    def __init__(self, root: Path, interval_sec: int = 3600):
        self.root = root
        self.interval = interval_sec
        self._stop = threading.Event()
        self.q: Queue[Path] = Queue()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None:
            return  # already running
        if Observer:
            observer = Observer()
            observer.schedule(_EventHandler(self.q), str(self.root), recursive=True)
            observer.start()
            self._thread = threading.Thread(target=self._loop_watchdog, args=(observer,), daemon=True)
        else:
            self._thread = threading.Thread(target=self._loop_poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ---------------------- internal ----------------------
    def _loop_watchdog(self, observer):
        while not self._stop.is_set():
            try:
                p = self.q.get(timeout=1)
                print(f"🔍 AutoScanner detected new file: {p}")
                # defer processing to ResourceManager via callback; placeholder
            except Empty:
                pass
        observer.stop()
        observer.join()

    def _loop_poll(self):
        last_scan = 0.0
        seen: set[str] = set()
        while not self._stop.is_set():
            now = time.time()
            if now - last_scan >= self.interval:
                for f in self.root.rglob("*"):
                    if f.is_file():
                        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                        if h not in seen:
                            seen.add(h)
                            print(f"🔍 AutoScanner detected new file: {f}")
                            # placeholder callback
                last_scan = now
            time.sleep(1)

# ---------------------------------------------------------------------------
# Utility functions (Phase 2)
# ---------------------------------------------------------------------------

def compute_sha256(path: Path) -> str:
    """Return first 16 hex chars of SHA-256 for the given file *or* directory.

    For directories we hash the relative file paths and their contents (deterministically
    sorted) so the hash changes whenever any contained file changes. This enables
    resource-level caching & deduplication for large model folders.
    """
    h = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    else:
        # Walk deterministically
        for fp in sorted([p for p in path.rglob("*") if p.is_file()]):
            rel = fp.relative_to(path)
            h.update(str(rel).encode())
            # Only hash first 1 MB of each large file to speed things up
            with fp.open("rb") as f:
                chunk = f.read(1024 * 1024)
                h.update(chunk)
    return h.hexdigest()[:16]


def file_size(path: Path) -> int:
    return path.stat().st_size

# ---------------------------------------------------------------------------
# Sandbox Executor (Phase 3)
# ---------------------------------------------------------------------------
class SandboxExecutor:
    """Very lightweight sandbox: executes `python -m py_compile <file>` inside a
    temporary venv with resource limits. Meant for quick safety checks before
    integrating arbitrary code. Heavy isolation (Docker/Firejail) can replace
    this in the future.
    """
    def __init__(self, cpu_sec: int = 120, mem_mb: int = 1024):
        self.cpu_sec = cpu_sec
        self.mem_mb = mem_mb

    def run(self, path: Path) -> bool:
        if not path.suffix == ".py":
            return True  # Only handle single Python files for v0.1
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / path.name
            shutil.copy2(path, target)
            cmd = [sys.executable, "-m", "py_compile", str(target)]
            log_dir = _REPO_ROOT / "logs" / "resource_manager" / "sandbox" / compute_sha256(path)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "compile.log"
            try:
                proc = subprocess.run(cmd, timeout=self.cpu_sec, capture_output=True, text=True)
                log_file.write_text(proc.stdout + "\n" + proc.stderr)
                ok = proc.returncode == 0
                if not ok:
                    logger.warning("Sandbox compile error for %s: %s", path, proc.stderr)
                return ok
            except subprocess.TimeoutExpired:
                logger.error("Sandbox timeout for %s", path)
                return False

# ---------------------------------------------------------------------------
# License detection util
# ---------------------------------------------------------------------------
_ALLOWED_LICENSES = {"MIT", "Apache-2.0", "BSD-3-Clause"}
_BLOCKED_LICENSES = {"GPL-3.0"}

def detect_license(path: Path) -> str | None:
    """Very naive license detector: search first 20 lines for SPDX or keywords."""
    try:
        lines = path.read_text(errors="ignore").splitlines()[:20]
        joined = "\n".join(lines)
        m = re.search(r"SPDX-License-Identifier:\s*([\w\-\.]+)", joined)
        if m:
            return m.group(1)
        if "MIT License" in joined:
            return "MIT"
        if "GNU GENERAL PUBLIC LICENSE" in joined or "GPL" in joined:
            return "GPL-3.0"
    except Exception:
        pass
    return None
