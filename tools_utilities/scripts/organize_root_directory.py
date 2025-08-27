#!/usr/bin/env python3
"""
Root Directory Organization Script

Purpose: Move all misplaced files from root directory to proper subdirectories
Inputs: Root directory scan
Outputs: Organized directory structure
Dependencies: pathlib, shutil
"""

import shutil
from pathlib import Path

def organize_root_directory():
    """Organize files in the root directory into proper subdirectories."""
    
    root = Path(".")
    moved_files = []
    
    # Define organization rules for root directory files
    organization_rules = {
        # Python scripts
        "*.py": "tools_utilities/scripts/",
        
        # Documentation files  
        "*README*.md": "docs/",
        "*SUMMARY*.md": "summaries/",
        "*INDEX*.md": "docs/",
        "*STRATEGY*.md": "docs/",
        "*SETUP*.md": "docs/",
        "*INTEGRATION*.md": "summaries/",
        "*MIGRATION*.md": "summaries/",
        "*MONITORING*.md": "summaries/",
        
        # Configuration and data files
        "*.json": "configs/project/",
        "*.html": "results/experiments/",
        "*.whl": "dist/",
        
        # Test files (but keep essential ones in root)
        "test_*.py": "tests/",
        "run_*_test*.py": "tests/",
    }
    
    # Files that should stay in root (essential project files)
    keep_in_root = {
        "README.md",
        "pyrightconfig.json", 
        ".gitignore",
        "requirements.txt",
        "setup.py",
        "pyproject.toml"
    }
    
    # Process each file in root directory
    for item in root.iterdir():
        if item.is_file() and item.name not in keep_in_root:
            # Find matching rule
            target_dir = None
            
            for pattern, destination in organization_rules.items():
                if item.match(pattern):
                    target_dir = Path(destination)
                    break
            
            # Special handling for numbered files (they look like generated files)
            if target_dir is None and item.name[0].isdigit():
                if item.suffix == '.py':
                    target_dir = Path("tools_utilities/scripts/")
                elif item.suffix == '.md':
                    target_dir = Path("docs/")
                elif item.suffix == '.json':
                    target_dir = Path("configs/project/")
            
            # Move the file if we found a target
            if target_dir:
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / item.name
                
                # Handle naming conflicts
                counter = 1
                while target_file.exists():
                    stem = item.stem
                    suffix = item.suffix
                    target_file = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                try:
                    shutil.move(str(item), str(target_file))
                    moved_files.append({
                        "from": item.name,
                        "to": str(target_file),
                        "type": "file"
                    })
                    print(f"✅ Moved: {item.name} → {target_file}")
                except Exception as e:
                    print(f"❌ Failed to move {item.name}: {e}")
    
    # Also organize some directories that might be misplaced
    dir_rules = {
        "notebooks": "research_lab/",
        "scripts": "tools_utilities/",
        "reports": "results/",
        "graphs": "results/experiments/"
    }
    
    for dir_name, target_parent in dir_rules.items():
        source_dir = root / dir_name
        if source_dir.exists() and source_dir.is_dir():
            target_parent_path = Path(target_parent)
            target_parent_path.mkdir(parents=True, exist_ok=True)
            target_dir = target_parent_path / dir_name
            
            # Handle conflicts
            if target_dir.exists():
                # Merge contents if target exists
                for item in source_dir.iterdir():
                    target_item = target_dir / item.name
                    counter = 1
                    while target_item.exists():
                        stem = item.stem if item.is_file() else item.name
                        suffix = item.suffix if item.is_file() else ""
                        target_item = target_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    try:
                        shutil.move(str(item), str(target_item))
                        print(f"✅ Merged: {item.name} → {target_item}")
                    except Exception as e:
                        print(f"❌ Failed to merge {item}: {e}")
                
                # Remove empty source directory
                try:
                    source_dir.rmdir()
                    moved_files.append({
                        "from": dir_name + "/",
                        "to": str(target_dir) + "/",
                        "type": "directory_merge"
                    })
                except:
                    pass
            else:
                # Move entire directory
                try:
                    shutil.move(str(source_dir), str(target_dir))
                    moved_files.append({
                        "from": dir_name + "/",
                        "to": str(target_dir) + "/", 
                        "type": "directory_move"
                    })
                    print(f"✅ Moved directory: {dir_name}/ → {target_dir}/")
                except Exception as e:
                    print(f"❌ Failed to move directory {dir_name}: {e}")
    
    return moved_files

def main():
    """Main function to organize root directory."""
    print("🗂️  Organizing root directory files...")
    print("=" * 50)
    
    moved_files = organize_root_directory()
    
    print("\n" + "=" * 50)
    print(f"✅ Root directory organization complete!")
    print(f"📁 Moved {len(moved_files)} items")
    
    # Show summary by type
    files_moved = sum(1 for item in moved_files if item["type"] == "file")
    dirs_moved = sum(1 for item in moved_files if item["type"].startswith("directory"))
    
    print(f"   • Files moved: {files_moved}")
    print(f"   • Directories moved: {dirs_moved}")
    
    print(f"\n📋 Root directory is now clean and professional!")

if __name__ == "__main__":
    main()
