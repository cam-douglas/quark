import yaml, pathlib, os
from typing import Any, Dict, List

ROOT = pathlib.Path("ROOT")

class ModelRegistry:
    def __init__(self, cfg_path: pathlib.Path = ROOT/"models.yaml"):
        self.cfg_path = cfg_path
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f) or {}
        self._index = {}
        for group, items in self.cfg.items():
            if group in ("routing",): continue
            for it in items or []:
                self._index[it["id"]] = it

    @property
    def routing(self) -> List[Dict[str, Any]]:
        return self.cfg.get("routing", [])

    def list(self) -> List[Dict[str, Any]]:
        return list(self._index.values())

    def get(self, model_id: str) -> Dict[str, Any]:
        return self._index[model_id]
