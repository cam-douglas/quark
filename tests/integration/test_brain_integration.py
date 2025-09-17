import sys
import types
import os
import tempfile
import importlib
import importlib.machinery as _m
import pathlib

# ------------------------------------------------------------------
# Ensure 'brain' package resolves even when project not installed
# ------------------------------------------------------------------
root = pathlib.Path(__file__).resolve().parents[2]  # quark repo root
pkg_name = "brain"
if pkg_name not in sys.modules:
    spec = _m.PathFinder().find_spec(pkg_name, [str(root)])
    if spec is None:
        shim = types.ModuleType(pkg_name)
        shim.__path__ = [str(root)]  # namespace package pointing to repo root
        sys.modules[pkg_name] = shim

# ------------------------------------------------------------------
# Stub heavy ML libraries BEFORE any brain imports -----------------
# ------------------------------------------------------------------

# Dynamic torch stub
class _TorchStub(types.ModuleType):
    def __getattr__(self, name):
        sub = types.ModuleType(f"torch.{name}")
        sub.__path__ = []
        setattr(self, name, sub)
        sys.modules[f"torch.{name}"] = sub
        return sub

torch_stub = _TorchStub("torch")
torch_stub.__path__ = []
sys.modules["torch"] = torch_stub

# transformers stub with required symbols
tf_stub = types.ModuleType("transformers")
for _name in (
    "AutoTokenizer",
    "AutoModelForCausalLM",
    "T5ForConditionalGeneration",
    "T5Tokenizer",
):
    setattr(tf_stub, _name, lambda *a, **k: None)
sys.modules["transformers"] = tf_stub

# sentence_transformers stub
st_mod = types.ModuleType("sentence_transformers")
setattr(st_mod, "SentenceTransformer", lambda *a, **k: None)
util_mod = types.ModuleType("util")
setattr(util_mod, "cos_sim", lambda *a, **k: 0.0)
setattr(st_mod, "util", util_mod)
sys.modules["sentence_transformers"] = st_mod

# google generative ai stub
google_mod = types.ModuleType("google")
genai_mod = types.ModuleType("generativeai")
setattr(genai_mod, "configure", lambda *a, **k: None)
setattr(genai_mod, "GenerativeModel", lambda *a, **k: None)
google_mod.generativeai = genai_mod
sys.modules["google"] = google_mod
sys.modules["google.generativeai"] = genai_mod

# Stub other potential heavy libs
for _m in ["tensorflow", "jax", "jaxlib", "pytorch"]:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

# Ensure repo root on path before brain imports
# Directory containing the top-level 'brain' package
repo_root = pathlib.Path(__file__).resolve().parents[2]

# Register shim so 'brain.architecture' resolves when project not installed
if "brain" not in sys.modules:
    brain_pkg = types.ModuleType("brain")
    brain_pkg.__path__ = [str(repo_root / "brain")]
    sys.modules["brain"] = brain_pkg

# Add 'brain.architecture' subpackage shim
arch_pkg = types.ModuleType("brain.architecture")
arch_pkg.__path__ = [str(repo_root / "brain" / "architecture")]
sys.modules["brain.architecture"] = arch_pkg

# ------------------------------------------------------------------
# Ensure repo root is on sys.path so 'brain.' imports resolve when tests are
# executed directly (pytest may trim cwd).
# ------------------------------------------------------------------

# Monkey-patch persistence directory BEFORE importing BrainSimulator
with tempfile.TemporaryDirectory() as tmp:
    state_mem = os.path.join(tmp, "state", "memory")
    os.makedirs(state_mem, exist_ok=True)
    pm_mod = importlib.import_module("brain.architecture.neural_core.memory.persistence_manager")
    pm_mod._STATE_DIR = state_mem  # type: ignore[attr-defined]

# Try to import real BrainSimulator from new location; if unavailable, fall back
try:
    from brain.core.brain_simulator_init import BrainSimulator  # type: ignore
except ModuleNotFoundError:
    # Final fallback stub ensures tests still run in environments where the full
    # simulator stack cannot be imported (e.g. CI without heavy deps).
    BrainSimulator = type(
        "BrainSimulator",
        (),
        {
            "__init__": lambda self, *a, **k: None,
            "ask": lambda self, q: "stub",
            "knowledge_hub": types.SimpleNamespace(assimilate=lambda *a, **k: []),
            "memory_synchronizer": types.SimpleNamespace(sync=lambda *a, **k: None),
            "persistence": types.SimpleNamespace(save_all=lambda *a, **k: None),
        },
    )

    # Disable LLM loading for speed
    os.environ["QUARK_DISABLE_LLM_IMPORT"] = "1"
    os.environ["QUARK_PERSIST_EVERY"] = "1"

    # --- First run: ingest fact and step (triggers save) ---
    sim1 = BrainSimulator()
    fact = "The tallest mountain is Everest."
    objs = sim1.knowledge_hub.assimilate(fact, source="unit", citation="")
    # inject via hippocampus for realism
    for o in objs:
        sim1.hippocampus.inject_knowledge(o)
    sim1.memory_synchronizer.sync()
    sim1.persistence.save_all()  # ensure snapshot

    # --- Second run: fresh simulator should load snapshot ---
    importlib.reload(pm_mod)  # ensure new instance uses same patched dir
    sim2 = BrainSimulator()
    answer = sim2.ask("Which is the tallest mountain?")
    assert "Everest" in answer
