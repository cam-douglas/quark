#!/usr/bin/env python3
"""
Workspace Cleanup Utility

Purpose: Maintain a clean, fast-loading Cursor workspace by removing temporary files and heavy directories
Inputs: None (scans project automatically)
Outputs: Cleaned workspace, size reduction report
Dependencies: os, shutil, pathlib
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total

def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"

def cleanup_test_runs(base_path: Path) -> Tuple[int, List[str]]:
    """Clean up test run directories, keeping only the 3 most recent."""
    cleaned_size = 0
    cleaned_paths = []
    
    test_dirs = [
        base_path / "tests" / "comprehensive_repo_tests",
        base_path / "tests" / "focused_repo_tests"
    ]
    
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue
            
        # Find test_run_* directories
        test_runs = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("test_run_")])
        
        # Keep only the 3 most recent, remove the rest
        for old_run in test_runs[:-3]:
            size = get_directory_size(old_run)
            try:
                shutil.rmtree(old_run)
                cleaned_size += size
                cleaned_paths.append(str(old_run))
            except Exception as e:
                print(f"Warning: Could not remove {old_run}: {e}")
    
    return cleaned_size, cleaned_paths

def cleanup_outputs(base_path: Path) -> Tuple[int, List[str]]:
    """Clean up test outputs, keeping only representative samples."""
    cleaned_size = 0
    cleaned_paths = []
    
    outputs_dir = base_path / "tests" / "outputs"
    if not outputs_dir.exists():
        return 0, []
    
    # Keep representative files
    keep_patterns = [
        "brain_connectivity_matrix.html",
        "brain_initialization_test.html", 
        "acetylcholine_system_analysis.html",
        "brain_connectivity_test.png",
        "comprehensive_test_report.md"
    ]
    
    for file in outputs_dir.iterdir():
        if file.is_file() and not any(pattern in file.name for pattern in keep_patterns):
            if file.suffix in ['.html', '.png'] and not file.name.endswith('_report.json'):
                size = file.stat().st_size
                try:
                    file.unlink()
                    cleaned_size += size
                    cleaned_paths.append(str(file))
                except Exception as e:
                    print(f"Warning: Could not remove {file}: {e}")
    
    return cleaned_size, cleaned_paths

def cleanup_result_images(base_path: Path) -> Tuple[int, List[str]]:
    """Remove matplotlib result images directory if it exists."""
    result_images = base_path / "result_images"
    if not result_images.exists():
        return 0, []
    
    size = get_directory_size(result_images)
    try:
        shutil.rmtree(result_images)
        return size, [str(result_images)]
    except Exception as e:
        print(f"Warning: Could not remove {result_images}: {e}")
        return 0, []

def cleanup_cache_dirs(base_path: Path) -> Tuple[int, List[str]]:
    """Clean up various cache directories."""
    cleaned_size = 0
    cleaned_paths = []
    
    cache_dirs = [
        base_path / "cache",
        base_path / "wikipedia_cache",
        base_path / "__pycache__",
        base_path / ".pytest_cache"
    ]
    
    # Also find all __pycache__ directories recursively
    for pycache in base_path.rglob("__pycache__"):
        cache_dirs.append(pycache)
    
    for cache_dir in cache_dirs:
        if cache_dir.exists() and cache_dir.is_dir():
            size = get_directory_size(cache_dir)
            try:
                shutil.rmtree(cache_dir)
                cleaned_size += size
                cleaned_paths.append(str(cache_dir))
            except Exception as e:
                print(f"Warning: Could not remove {cache_dir}: {e}")
    
    return cleaned_size, cleaned_paths

def main():
    """Main cleanup function."""
    base_path = Path(__file__).parent.parent.parent  # Go up to project root
    print(f"🧹 Cleaning workspace: {base_path}")
    
    total_cleaned = 0
    all_cleaned_paths = []
    
    # 1. Clean up test runs
    print("\n1. Cleaning test run directories...")
    size, paths = cleanup_test_runs(base_path)
    total_cleaned += size
    all_cleaned_paths.extend(paths)
    print(f"   Removed {len(paths)} test run directories ({format_size(size)})")
    
    # 2. Clean up test outputs
    print("\n2. Cleaning test output files...")
    size, paths = cleanup_outputs(base_path)
    total_cleaned += size
    all_cleaned_paths.extend(paths)
    print(f"   Removed {len(paths)} output files ({format_size(size)})")
    
    # 3. Clean up result images
    print("\n3. Cleaning matplotlib result images...")
    size, paths = cleanup_result_images(base_path)
    total_cleaned += size
    all_cleaned_paths.extend(paths)
    if paths:
        print(f"   Removed result_images directory ({format_size(size)})")
    else:
        print("   No result_images directory found")
    
    # 4. Clean up cache directories
    print("\n4. Cleaning cache directories...")
    size, paths = cleanup_cache_dirs(base_path)
    total_cleaned += size
    all_cleaned_paths.extend(paths)
    print(f"   Removed {len(paths)} cache directories ({format_size(size)})")
    
    # Summary
    print(f"\n✅ Workspace cleanup complete!")
    print(f"📊 Total space freed: {format_size(total_cleaned)}")
    print(f"🗑️  Removed {len(all_cleaned_paths)} items")
    
    # Check remaining large directories
    print(f"\n📁 Large directories still present:")
    large_dirs = [
        ("models", base_path / "models"),
        ("scaled_wikipedia_trained", base_path / "knowledge_systems" / "training_pipelines" / "scaled_wikipedia_trained"),
        ("connectome_exports", base_path / "brain_modules" / "connectome" / "exports"),
        ("venv", base_path / "venv"),
        ("wikipedia_env", base_path / "wikipedia_env")
    ]
    
    for name, path in large_dirs:
        if path.exists():
            size = get_directory_size(path)
            if size > 100 * 1024 * 1024:  # > 100MB
                print(f"   {name}: {format_size(size)}")
    
    print(f"\n💡 Tip: These large directories are excluded from Cursor indexing via pyrightconfig.json")
    print(f"🔄 Run this script regularly to maintain workspace performance")

if __name__ == "__main__":
    main()


