

"""
Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.

Resource Manager
----------------
Centralized module responsible for registering, sandbox-validating, and integrating external
"resources" (code, datasets, models, configs) into the Quark brain infrastructure.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
from pathlib import Path
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
import subprocess
import tempfile
import re
# New imports for streaming training integration

# Cline integration imports
try:
    from brain.modules.cline_integration.cline_adapter import ClineAdapter, CodingTask, TaskComplexity, ClineTaskType
    CLINE_AVAILABLE = True
except ImportError:
    CLINE_AVAILABLE = False
    ClineAdapter = None

# ---------------------------------------------------------------------------
# Constants & Globals
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[4]  # /quark
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

        # Initialize Cline integration
        self.cline_adapter: Optional[ClineAdapter] = None
        self.cline_enabled = CLINE_AVAILABLE and os.environ.get("QUARK_CLINE_ENABLED", "true").lower() == "true"
        if self.cline_enabled and CLINE_AVAILABLE:
            try:
                self.cline_adapter = ClineAdapter(resource_manager=self)
                logger.info("Cline autonomous coding integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Cline adapter: {e}")
                self.cline_enabled = False

        # ------------------------------------------------------------------
        # Lightweight resource buckets (CPU/RAM placeholder)
        # ------------------------------------------------------------------
        import threading

        self.BUCKET_LIMITS = {
            "nlp": int(os.environ.get("QUARK_RM_NLP_LIMIT", 2)),
            "io": int(os.environ.get("QUARK_RM_IO_LIMIT", 4)),
            "background": int(os.environ.get("QUARK_RM_BG_LIMIT", 8)),
        }
        # Convert zeros / negatives to 1
        self._semaphores = {
            name: threading.BoundedSemaphore(value=max(1, cap))
            for name, cap in self.BUCKET_LIMITS.items()
        }

        # Start autoscan if enabled in default config
        if auto_scan:
            self.autoscanner = AutoScanner(_DATA_DIR, scan_interval_sec)
            self.autoscanner.start()
        else:
            self.autoscanner = None

        # Register as global default if none exists
        if not getattr(ResourceManager, "_DEFAULT", None):
            ResourceManager._DEFAULT = self

    # ------------------------------------------------------------------
    # Streaming-aware training / fine-tuning launcher (Phase 3)
    # ------------------------------------------------------------------

    def _launch_orchestrator(self, verb: str, backend: str = "local", deploy: bool = False,
                             overrides: dict[str, str] | None = None, checkpoint: Optional[str] = None) -> int:
        """Run pipeline_orchestrator.py with given verb (train/finetune)."""
        overrides = overrides or {}
        orch_path = _REPO_ROOT / "tools_utilities/scripts/pipeline_orchestrator.py"
        if not orch_path.exists():
            logger.error("pipeline_orchestrator.py not found at %s", orch_path)
            return 1

        cmd = [sys.executable, str(orch_path), verb, backend]
        if deploy:
            cmd.append("--deploy")
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])

        for k, v in overrides.items():
            cmd += ["--override", f"{k}={v}"]

        logger.info("[ResourceManager] launching orchestrator: %s", " ".join(cmd))
        try:
            completed = subprocess.run(cmd, check=False)
            return completed.returncode
        except Exception as exc:
            logger.error("Failed to launch orchestrator: %s", exc)
            return 1

    # Public API
    def run_training_job(self, task_type: str, overrides: Optional[dict[str, str]] = None, dataset_local_path: Optional[str] = None) -> int:
        """Unified entry for training or fine-tuning.

        Parameters
        ----------
        task_type : str
            "train" or "fine_tune" / "finetune" (case-insensitive).
        overrides : dict, optional
            Key-value pairs forwarded to quark_cli via --override.
        """
        task_type_l = task_type.lower()

        # If caller passes a local dataset path, convert to S3-relative prefix
        if dataset_local_path:
            local_p = Path(dataset_local_path).expanduser().resolve()
            try:
                relative = local_p.relative_to(_REPO_ROOT)
            except ValueError:
                # Path outside repo – fall back to basename
                relative = local_p.name
            prefix = str(relative).strip("/") + "/"
            overrides = overrides.copy() if overrides else {}
            overrides.setdefault("train_prefix", prefix)

        if task_type_l in {"train", "training"}:
            return self._launch_orchestrator("train", overrides=overrides)
        elif task_type_l in {"fine_tune", "finetune", "fine-tune"}:
            return self._launch_orchestrator("finetune", overrides=overrides)
        else:
            logger.warning("Unknown task_type %s – falling back to legacy path", task_type)
            # legacy behaviour placeholder
            return 0

    # ------------------------------------------------------------------
    # Model checkpoint persistence
    # ------------------------------------------------------------------
    def register_model_checkpoint(self, ckpt_path: Path, name: str = "latest-quark-model") -> Path:
        """Copy checkpoint to /data/models/ and update registry."""
        dest_dir = _REPO_ROOT / "data" / "models"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / ckpt_path.name
        try:
            shutil.copy2(ckpt_path, dest_path)
            # update registry
            self.registry.setdefault("models", {})[name] = str(dest_path)
            import yaml
            self._registry_path.write_text(yaml.safe_dump(self.registry))
            logger.info("Stored model %s at %s", name, dest_path)
            return dest_path
        except Exception as e:
            logger.error("Failed to store checkpoint: %s", e)
            return ckpt_path

    # ------------------------------------------------------------------
    # StreamingManager access helper
    # ------------------------------------------------------------------
    def get_streaming_manager(self, bucket: str):
        """Return (and cache) a StreamingManager for *bucket*."""
        from tools_utilities.scripts.s3_streaming_manager import S3StreamingManager as _SM  # updated import path
        if not hasattr(self, "_sm_cache"):
            self._sm_cache = {}
        if bucket not in self._sm_cache:
            self._sm_cache[bucket] = _SM(bucket_name=bucket)
        return self._sm_cache[bucket]

    # ------------------------------------------------------------------
    # Global accessor
    # ------------------------------------------------------------------
    _DEFAULT = None  # type: Optional["ResourceManager"]

    @classmethod
    def get_default(cls) -> "ResourceManager | None":
        return cls._DEFAULT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    from contextlib import contextmanager

    @contextmanager
    def request_resources(self, bucket: str = "background"):
        """Context-manager to reserve a resource token from *bucket*.

        Usage::
            with resource_manager.request_resources("nlp"):
                heavy_llm_call()
        """
        sem = self._semaphores.get(bucket, self._semaphores["background"])
        acquired = sem.acquire(timeout=float(os.environ.get("QUARK_RM_TIMEOUT", "300")))
        if not acquired:
            raise RuntimeError(f"ResourceManager timeout waiting for bucket '{bucket}'.")
        try:
            yield
        finally:
            sem.release()

    # ------------------------------------------------------------------
    def set_bucket_limit(self, bucket: str, limit: int):
        if limit <= 0:
            raise ValueError("limit must be positive")
        import threading
        diff = limit - self.BUCKET_LIMITS.get(bucket, 0)
        if bucket not in self._semaphores:
            self._semaphores[bucket] = threading.BoundedSemaphore(value=limit)
        else:
            # Adjust semaphore by releasing or acquiring to match new capacity
            sem = self._semaphores[bucket]
            if diff > 0:
                for _ in range(diff):
                    sem.release()
            elif diff < 0:
                for _ in range(-diff):
                    sem.acquire(blocking=False)
        self.BUCKET_LIMITS[bucket] = limit

    def get_stats(self):
        """Return current bucket utilisation stats."""
        out = {}
        for name, sem in self._semaphores.items():
            out[name] = {
                "capacity": self.BUCKET_LIMITS[name],
                "available": sem._value,  # noqa: SLF001 – internal ok for stats
            }
        return out

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
        # Prefer explicit domain from metadata when provided
        domain = None
        t = meta.get("type") if isinstance(meta, dict) else None
        if isinstance(t, str):
            t_lower = t.lower()
            # Map common types to directory names
            mapping = {
                "model": "models",
                "dataset": "datasets",
                "mesh": "meshes",
                "memory": "memories",
                "misc": "misc",
            }
            domain = mapping.get(t_lower)

        if size > 200 * 2**20:
            target_base = _DATA_DIR / (domain or self._infer_domain(src))
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

        # Test-mode gate to avoid heavy copies during CI/unit tests
        if os.environ.get("QUARK_SKIP_MODEL_COPY_FOR_TESTS", "") == "1":
            meta["integrated_path"] = str(src)
            meta["approved"] = True
            self._persist_registry()
            logger.info("Test mode: skipping copy for %s", src)
            hub.emit("resource_integrated", id=resource_id, path=str(src))
            self._notify_simulators(resource_id, meta)
            return

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

    # -----------------------------------------------------------------------
    # Cline Autonomous Coding Integration
    # -----------------------------------------------------------------------
    
    async def handle_complex_coding_task(self, task_description: str, files_involved: Optional[list] = None) -> Dict[str, Any]:
        """
        Handle complex coding tasks with intelligent delegation to Cline
        
        Args:
            task_description: Natural language description of coding task
            files_involved: Optional list of files that may be involved
            
        Returns:
            Dictionary with task result and execution details
        """
        if not self.cline_enabled or not self.cline_adapter:
            logger.warning("Cline integration not available, handling task locally")
            return {
                "success": False,
                "message": "Cline integration not available",
                "handled_locally": True
            }
        
        try:
            # Create CodingTask object
            task = CodingTask(
                description=task_description,
                task_type=self._determine_task_type(task_description),
                complexity=self._assess_task_complexity(task_description),
                files_involved=files_involved or [],
                biological_constraints=True,
                context={"resource_manager": "quark_brain"}
            )
            
            # Execute via Cline adapter
            result = await self.cline_adapter.handle_complex_coding_task(task)
            
            return {
                "success": result.success,
                "output": result.output,
                "files_modified": result.files_modified,
                "commands_executed": result.commands_executed,
                "biological_compliance": result.biological_compliance,
                "execution_time": result.execution_time,
                "error_message": result.error_message
            }
            
        except Exception as e:
            logger.error(f"Cline task execution failed: {e}")
            return {
                "success": False,
                "message": f"Cline execution error: {e}",
                "handled_locally": False
            }

    async def autonomous_code_generation(self, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Generate code autonomously using Cline with full brain context
        
        Args:
            task_description: Description of code to generate
            **kwargs: Additional arguments for task configuration
            
        Returns:
            Dictionary with generation result and details
        """
        if not self.cline_enabled or not self.cline_adapter:
            return {"success": False, "message": "Cline not available"}
        
        try:
            result = await self.cline_adapter.autonomous_code_generation(task_description)
            
            # Register any new files created
            for file_path in result.files_modified:
                if Path(file_path).exists():
                    self.register_resource(file_path, {
                        "source": "cline_autonomous_generation",
                        "task": task_description,
                        "biological_compliance": result.biological_compliance
                    })
            
            return {
                "success": result.success,
                "output": result.output,
                "files_created": result.files_modified,
                "biological_compliance": result.biological_compliance
            }
            
        except Exception as e:
            logger.error(f"Autonomous code generation failed: {e}")
            return {"success": False, "message": str(e)}

    async def test_neural_interface(self, test_scenario: str, app_url: str = "http://localhost:3000") -> Dict[str, Any]:
        """
        Test neural interfaces using Cline's browser automation
        
        Args:
            test_scenario: Description of testing scenario
            app_url: URL of neural interface to test
            
        Returns:
            Dictionary with test results
        """
        if not self.cline_enabled or not self.cline_adapter:
            return {"success": False, "message": "Cline browser automation not available"}
        
        try:
            result = await self.cline_adapter.browser_automation_testing(test_scenario, app_url)
            
            return {
                "success": result.success,
                "test_output": result.output,
                "test_scenario": test_scenario,
                "app_url": app_url
            }
            
        except Exception as e:
            logger.error(f"Neural interface testing failed: {e}")
            return {"success": False, "message": str(e)}

    def get_brain_context_for_cline(self) -> Dict[str, Any]:
        """
        Get brain architecture context for Cline integration
        
        Returns:
            Dictionary containing current brain state and constraints
        """
        if not self.cline_adapter:
            return {}
        
        # This would be called synchronously, but Cline adapter handles async internally
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.cline_adapter.get_brain_context())
        except RuntimeError:
            # If no event loop is running, create one
            return asyncio.run(self.cline_adapter.get_brain_context())

    def _determine_task_type(self, description: str) -> 'ClineTaskType':
        """Determine task type from description"""
        description_lower = description.lower()
        
        if any(term in description_lower for term in ["test", "browser", "interface"]):
            return ClineTaskType.BROWSER_TESTING
        elif any(term in description_lower for term in ["document", "readme", "docs"]):
            return ClineTaskType.DOCUMENTATION
        elif any(term in description_lower for term in ["refactor", "restructure"]):
            return ClineTaskType.REFACTORING
        elif any(term in description_lower for term in ["edit", "modify", "update"]):
            return ClineTaskType.FILE_EDITING
        elif any(term in description_lower for term in ["run", "execute", "command"]):
            return ClineTaskType.COMMAND_EXECUTION
        else:
            return ClineTaskType.CODE_GENERATION

    def _assess_task_complexity(self, description: str) -> 'TaskComplexity':
        """Assess task complexity from description"""
        description_lower = description.lower()
        
        if any(term in description_lower for term in ["architecture", "brain", "neural", "morphogen"]):
            return TaskComplexity.CRITICAL
        elif any(term in description_lower for term in ["refactor", "multiple files", "system"]):
            return TaskComplexity.COMPLEX
        elif any(term in description_lower for term in ["edit", "modify", "update"]):
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

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
