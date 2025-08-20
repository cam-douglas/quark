import os, pathlib, json
ROOT = pathlib.Path(os.environ.get("SMALLMIND_ROOT", "ROOT")).resolve()
VAR  = ROOT / ".neuro"
VAR.mkdir(parents=True, exist_ok=True)
CACHE = VAR / "cache"
CACHE.mkdir(exist_ok=True)
DEFAULT_MODELS = ROOT / "models.yaml"
HEBBIAN_PATH = VAR / "hebbian.json"
SETTINGS = {
    "scan_globs": ["**/*.py", "**/*.sh", "**/*.zsh", "**/*.md", "**/*.yml", "**/*.yaml", "**/*.json"],
    "ignore_globs": ["env/**", ".venv/**", "node_modules/**", ".git/**", "**/__pycache__/**", "runs/**", "logs/**"],
    "embedding": {"prefer_st": ["intfloat/e5-small-v2", "all-MiniLM-L6-v2"]},
    "similarity_threshold": 0.22,
}
def load_hebbian():
    if HEBBIAN_PATH.exists():
        return json.loads(HEBBIAN_PATH.read_text())
    return {"edges": {}}
def save_hebbian(h):
    HEBBIAN_PATH.write_text(json.dumps(h, ensure_ascii=False, indent=2))
