from __future__ import annotations
import ast, json, re, pathlib, yaml
from typing import Dict, List, Set, Tuple
from .....................................................scanners import read_text_safe

def analyze_python(path: pathlib.Path, text: str) -> Dict:
    imports: Set[str] = set()
    symbols: Set[str] = set()
    entry: bool = False
    try:
        tree = ast.parse(text or "", filename=str(path))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module: imports.add(n.module.split(".")[0])
            elif isinstance(n, ast.FunctionDef):
                symbols.add(n.name)
            elif isinstance(n, ast.ClassDef):
                symbols.add(n.name)
        entry = "__main__" in text and "if __name__ == \"__main__\"" in text.replace("'",'"')
    except Exception:
        pass
    return {"kind":"python","imports":sorted(imports), "symbols":sorted(symbols), "entrypoint":entry}

def analyze_shell(path: pathlib.Path, text: str) -> Dict:
    cmds = re.findall(r'\n?([a-zA-Z0-9_\-\.]+)\s', text)
    py_calls = re.findall(r'python(?:3)?\s+\-m\s+([a-zA-Z0-9_\.\-]+)', text)
    return {"kind":"shell","cmds":cmds[:128], "python_modules":py_calls[:64]}

def analyze_md(path: pathlib.Path, text: str) -> Dict:
    fences = re.findall(r"```([a-zA-Z0-9_\-\+\.]*)\n(.*?)```", text, flags=re.S)
    langs = [lang or "text" for lang,_ in fences]
    return {"kind":"markdown","code_blocks":len(fences),"languages":langs[:64]}

def analyze_config(path: pathlib.Path, text: str) -> Dict:
    try:
        data = yaml.safe_load(text) if path.suffix in [".yml",".yaml"] else json.loads(text)
        keys = list(data.keys())[:64] if isinstance(data, dict) else []
        return {"kind":"config","keys":keys}
    except Exception:
        return {"kind":"config","keys":[]}

def analyze_file(path: pathlib.Path, text: str) -> Dict:
    suff = path.suffix.lower()
    if suff == ".py":    return analyze_python(path, text)
    if suff in [".sh",".zsh"]: return analyze_shell(path, text)
    if suff in [".md"]:  return analyze_md(path, text)
    if suff in [".yml",".yaml",".json"]: return analyze_config(path, text)
    return {"kind":"other"}
