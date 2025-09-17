#!/usr/bin/env python3
"""
Data Validation for Brainstem Segmentation Datasets

Validates downloaded data and reports status.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def validate_allen_data(data_dir: Path) -> Tuple[bool, str]:
    """Validate Allen Brain Atlas data.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Tuple of (success, message)
    """
    allen_dir = data_dir / "raw" / "allen" / "metadata"
    
    if not allen_dir.exists():
        return False, "Allen metadata directory not found"
    
    json_files = list(allen_dir.glob("experiment_*.json"))
    summary_file = allen_dir / "experiments_summary.json"
    
    if not json_files:
        return False, "No experiment metadata files found"
    
    if not summary_file.exists():
        return False, "Summary file not found"
    
    # Check summary content
    with open(summary_file) as f:
        summary = json.load(f)
    
    exp_count = summary.get("total_experiments", 0)
    msg = f"✅ Allen: {exp_count} experiments, {len(json_files)} metadata files"
    
    return True, msg


def validate_devccf_data(data_dir: Path) -> Tuple[bool, str]:
    """Validate DevCCF atlas data.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Tuple of (success, message)
    """
    template_dir = data_dir / "templates" / "devccf"
    
    if not template_dir.exists():
        return False, "DevCCF template directory not found"
    
    expected_files = {
        "E11.5": ["Annotations", "Reference"],
        "E13.5": ["Annotations", "Reference"],
        "E15.5": ["Annotations", "Reference"]
    }
    
    found_stages = []
    missing_stages = []
    
    for stage, file_types in expected_files.items():
        stage_found = False
        for ftype in file_types:
            pattern = f"*{stage}*{ftype}*.nii*"
            if list(template_dir.glob(pattern)):
                stage_found = True
                break
        
        if stage_found:
            found_stages.append(stage)
        else:
            missing_stages.append(stage)
    
    if not found_stages:
        return False, "❌ DevCCF: No atlases found (manual download required)"
    
    msg = f"⚠️ DevCCF: {len(found_stages)}/3 stages found"
    if missing_stages:
        msg += f" (missing: {', '.join(missing_stages)})"
    
    return len(found_stages) == 3, msg


def validate_registration_config(data_dir: Path) -> Tuple[bool, str]:
    """Validate registration configuration.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Tuple of (success, message)
    """
    config_file = data_dir / "metadata" / "registration_config.json"
    
    if not config_file.exists():
        return False, "Registration config not found"
    
    with open(config_file) as f:
        config = json.load(f)
    
    required_keys = ["template", "registration_method", "parameters"]
    missing = [k for k in required_keys if k not in config]
    
    if missing:
        return False, f"Config missing keys: {missing}"
    
    return True, f"✅ Registration: {config['registration_method']} configured"


def generate_status_report(data_dir: Path) -> None:
    """Generate comprehensive status report.
    
    Args:
        data_dir: Base data directory
    """
    print("\n" + "="*60)
    print("BRAINSTEM SEGMENTATION DATA VALIDATION REPORT")
    print("="*60)
    
    # Check each component
    checks = [
        ("Allen Brain Atlas", validate_allen_data(data_dir)),
        ("DevCCF Atlases", validate_devccf_data(data_dir)),
        ("Registration Config", validate_registration_config(data_dir))
    ]
    
    all_valid = True
    for name, (valid, msg) in checks:
        print(f"\n{msg}")
        if not valid:
            all_valid = False
    
    # Directory sizes
    print("\n" + "-"*40)
    print("STORAGE USAGE:")
    
    for subdir in ["raw", "templates", "registered", "metadata"]:
        dir_path = data_dir / subdir
        if dir_path.exists():
            # Count files and estimate size
            file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
            print(f"  {subdir:15} : {file_count:4} files")
    
    # Recommendations
    print("\n" + "-"*40)
    print("NEXT STEPS:")
    
    if not checks[1][1][0]:  # DevCCF not complete
        print("1. ⚠️ Download DevCCF atlases manually")
        print("   See: manual_download_guide.md")
    else:
        print("1. ✅ DevCCF atlases ready")
    
    print("2. Set up ANTs registration pipeline")
    print("3. Begin registration to common template")
    print("4. Proceed to Step 2 (literature review)")
    
    # Overall status
    print("\n" + "="*60)
    if all_valid:
        print("STATUS: ✅ READY FOR REGISTRATION")
    else:
        print("STATUS: ⚠️ MANUAL DOWNLOADS REQUIRED")
    print("="*60 + "\n")


def main():
    """Run validation checks."""
    data_dir = Path("/Users/camdouglas/quark/data/datasets/brainstem_segmentation")
    
    if not data_dir.exists():
        print("❌ Data directory not found!")
        print(f"Expected: {data_dir}")
        return
    
    generate_status_report(data_dir)


if __name__ == "__main__":
    main()
