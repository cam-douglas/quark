from __future__ import annotations
import os, shutil, pathlib, re
from typing import Dict, List, Set, Tuple, Optional
from .....................................................config import ROOT
from .....................................................analyzers import analyze_file
from .....................................................scanners import read_text_safe, scan_files

# File organization rules based on content and patterns
ORGANIZATION_RULES = {
    # Core system directories
    "models": {
        "patterns": [r".*model.*\.py$", r".*\.pkl$", r".*\.pt$", r".*\.pth$", r".*checkpoint.*"],
        "keywords": ["torch", "tensorflow", "sklearn", "model", "checkpoint", "weights"],
        "description": "Machine learning models and checkpoints"
    },
    "data": {
        "patterns": [r".*\.csv$", r".*\.json$", r".*\.parquet$", r".*\.h5$", r".*dataset.*"],
        "keywords": ["pandas", "dataset", "data", "csv", "json"],
        "description": "Data files and datasets"
    },
    "scripts": {
        "patterns": [r".*script.*\.py$", r".*run.*\.py$", r".*main.*\.py$"],
        "keywords": ["argparse", "if __name__", "main()", "script"],
        "description": "Executable scripts and entry points"
    },
    "tests": {
        "patterns": [r"test_.*\.py$", r".*_test\.py$", r".*tests?\.py$"],
        "keywords": ["pytest", "unittest", "test", "assert"],
        "description": "Test files and test suites"
    },
    "configs": {
        "patterns": [r".*config.*\.(yml|yaml|json|toml)$", r".*settings.*\.(yml|yaml|json)$"],
        "keywords": ["config", "settings", "parameters"],
        "description": "Configuration files"
    },
    "docs": {
        "patterns": [r".*\.md$", r".*\.rst$", r"README.*", r"CHANGELOG.*"],
        "keywords": ["documentation", "readme", "guide"],
        "description": "Documentation and guides"
    },
    "notebooks": {
        "patterns": [r".*\.ipynb$"],
        "keywords": ["jupyter", "notebook"],
        "description": "Jupyter notebooks and analysis"
    },
    "simulation": {
        "patterns": [r".*simulation.*\.py$", r".*sim.*\.py$"],
        "keywords": ["simulation", "physics", "mujoco", "compucell"],
        "description": "Simulation and physics engines"
    },
    "agents": {
        "patterns": [r".*agent.*\.py$", r".*bot.*\.py$"],
        "keywords": ["agent", "llm", "openai", "anthropic", "chat"],
        "description": "AI agents and chatbots"
    },
    "visualization": {
        "patterns": [r".*viz.*\.py$", r".*plot.*\.py$", r".*visual.*\.py$"],
        "keywords": ["matplotlib", "plotly", "seaborn", "visualization"],
        "description": "Visualization and plotting tools"
    }
}

def analyze_file_purpose(file_path: pathlib.Path) -> Dict[str, float]:
    """Analyze a file to determine its likely purpose/category."""
    text = read_text_safe(file_path)
    analysis = analyze_file(file_path, text)
    
    scores = {}
    file_name = file_path.name.lower()
    file_path_str = str(file_path).lower()
    
    for category, rules in ORGANIZATION_RULES.items():
        score = 0.0
        
        # Pattern matching (filename/path)
        for pattern in rules["patterns"]:
            if re.search(pattern, file_path_str, re.IGNORECASE):
                score += 3.0
                
        # Keyword matching in content
        text_lower = text.lower()
        for keyword in rules["keywords"]:
            if keyword in text_lower:
                score += 1.0
                
        # Import analysis for Python files
        if analysis.get("kind") == "python":
            imports = set(analysis.get("imports", []))
            for keyword in rules["keywords"]:
                if keyword in imports:
                    score += 2.0
                    
        # Symbol analysis
        symbols = set(analysis.get("symbols", []))
        for keyword in rules["keywords"]:
            for symbol in symbols:
                if keyword.lower() in symbol.lower():
                    score += 1.5
                    
        scores[category] = score
        
    return scores

def suggest_organization(dry_run: bool = True) -> Dict[str, List[Tuple[pathlib.Path, str, float]]]:
    """Suggest file organization based on content analysis."""
    files = scan_files()
    suggestions = {}
    
    for file_path in files:
        # Skip files already in organized directories
        parts = file_path.parts
        if len(parts) >= 2 and parts[-2] in ORGANIZATION_RULES:
            continue
            
        # Skip system files
        if any(skip in str(file_path).lower() for skip in ['.git', '__pycache__', '.venv', 'env/']):
            continue
            
        scores = analyze_file_purpose(file_path)
        
        # Find best category (must have score > threshold)
        best_category = max(scores.items(), key=lambda x: x[1])
        if best_category[1] > 2.0:  # Minimum confidence threshold
            category, score = best_category
            if category not in suggestions:
                suggestions[category] = []
            suggestions[category].append((file_path, category, score))
            
    return suggestions

def organize_files(suggestions: Dict, dry_run: bool = True) -> Dict[str, List[str]]:
    """Organize files based on suggestions."""
    results = {"moved": [], "created_dirs": [], "errors": []}
    
    for category, file_list in suggestions.items():
        target_dir = ROOT / category
        
        if not dry_run and not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            results["created_dirs"].append(str(target_dir))
            
        for file_path, _, score in file_list:
            target_path = target_dir / file_path.name
            
            try:
                if not dry_run:
                    # Avoid overwriting existing files
                    if target_path.exists():
                        base = target_path.stem
                        suffix = target_path.suffix
                        counter = 1
                        while target_path.exists():
                            target_path = target_dir / f"{base}_{counter}{suffix}"
                            counter += 1
                            
                    shutil.move(str(file_path), str(target_path))
                    
                results["moved"].append(f"{file_path} -> {target_path} (score: {score:.1f})")
                
            except Exception as e:
                results["errors"].append(f"Failed to move {file_path}: {e}")
                
    return results

def generate_file_titles(file_path: pathlib.Path) -> List[str]:
    """Generate descriptive titles for files based on content."""
    text = read_text_safe(file_path)
    analysis = analyze_file(file_path, text)
    
    titles = []
    
    # Extract titles from different sources
    if analysis.get("kind") == "python":
        # Look for docstrings
        docstring_matches = re.findall(r'"""([^"]*?)"""', text, re.DOTALL)
        for match in docstring_matches[:3]:  # First 3 docstrings
            lines = match.strip().split('\n')
            if lines:
                title = lines[0].strip()
                if len(title) > 10 and len(title) < 100:
                    titles.append(title)
                    
        # Look for class names
        classes = [s for s in analysis.get("symbols", []) if s[0].isupper()]
        titles.extend(classes[:3])
        
    elif analysis.get("kind") == "markdown":
        # Extract headers
        headers = re.findall(r'^#+\s*(.+)$', text, re.MULTILINE)
        titles.extend(headers[:5])
        
    elif analysis.get("kind") == "config":
        # Use config keys as titles
        keys = analysis.get("keys", [])
        titles.extend(keys[:3])
        
    # Clean and format titles
    clean_titles = []
    for title in titles:
        if isinstance(title, str):  # Ensure title is a string
            # Remove extra whitespace and formatting
            clean_title = re.sub(r'\s+', ' ', title.strip())
            if 5 <= len(clean_title) <= 80:
                clean_titles.append(clean_title)
            
    return clean_titles[:5]  # Return top 5 titles
