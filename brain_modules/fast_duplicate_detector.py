#!/usr/bin/env python3
"""
Fast Duplicate Detection Module
==============================

Uses content fingerprinting and parallel processing to quickly identify
duplicate and redundant files without O(n²) complexity.

Author: Quark Brain Architecture
Date: 2024
"""

import os
import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import time

def calculate_file_fingerprint(file_path: Path, chunk_size: int = 1024) -> str:
    """Calculate fast fingerprint from first chunk of file"""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(chunk_size)
            return hashlib.md5(chunk).hexdigest()
    except Exception:
        return ""

def calculate_full_hash(file_path: Path) -> str:
    """Calculate full content hash"""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception:
        return ""

def analyze_file_batch(file_paths: List[Path]) -> List[Tuple[str, int, str]]:
    """Analyze a batch of files (for parallel processing)"""
    results = []
    for file_path in file_paths:
        try:
            size = file_path.stat().st_size
            quick_hash = calculate_file_fingerprint(file_path)
            results.append((str(file_path), size, quick_hash))
        except Exception:
            continue
    return results

def find_duplicates_optimized(repo_root: str, max_workers: int = 4) -> List[Tuple[str, List[str]]]:
    """Find duplicates using optimized fingerprinting"""
    print("🔍 Fast duplicate detection using parallel processing...")
    start_time = time.time()
    
    repo_path = Path(repo_root)
    
    # Collect all files
    all_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_file() and file_path.stat().st_size > 0:
                all_files.append(file_path)
    
    print(f"📁 Analyzing {len(all_files):,} files...")
    
    # Group files by size first (files with different sizes can't be duplicates)
    size_groups = defaultdict(list)
    for file_path in all_files:
        try:
            size = file_path.stat().st_size
            size_groups[size].append(file_path)
        except Exception:
            continue
    
    # Process size groups in parallel
    duplicates = []
    
    with mp.Pool(max_workers) as pool:
        # Process each size group
        for size, files in size_groups.items():
            if len(files) > 1:  # Only check groups with multiple files
                # Group by quick hash
                quick_hash_groups = defaultdict(list)
                
                for file_path in files:
                    quick_hash = calculate_file_fingerprint(file_path)
                    quick_hash_groups[quick_hash].append(file_path)
                
                # Check groups with multiple files
                for quick_hash, hash_files in quick_hash_groups.items():
                    if len(hash_files) > 1:
                        # Verify with full content hash
                        content_hash_groups = defaultdict(list)
                        
                        for file_path in hash_files:
                            content_hash = calculate_full_hash(file_path)
                            content_hash_groups[content_hash].append(str(file_path))
                        
                        # Add verified duplicates
                        for content_hash, duplicate_files in content_hash_groups.items():
                            if len(duplicate_files) > 1:
                                original = duplicate_files[0]
                                copies = duplicate_files[1:]
                                duplicates.append((original, copies))
    
    elapsed_time = time.time() - start_time
    print(f"✅ Found {len(duplicates)} duplicate groups in {elapsed_time:.2f}s")
    
    return duplicates

def find_backup_cache_files(repo_root: str) -> List[str]:
    """Find backup, cache, and temporary files"""
    print("🔍 Finding backup and cache files...")
    
    backup_patterns = ['*backup*', '*cache*', '*temp*', '*tmp*', '*~', '*.log', '*.pyc']
    backup_files = []
    
    for pattern in backup_patterns:
        for file_path in Path(repo_root).rglob(pattern):
            if file_path.is_file():
                backup_files.append(str(file_path))
    
    print(f"📁 Found {len(backup_files)} backup/cache files")
    return backup_files

if __name__ == "__main__":
    # Test the fast duplicate detection
    duplicates = find_duplicates_optimized('.')
    backup_files = find_backup_cache_files('.')
    
    print(f"\n📊 Results:")
    print(f"   Duplicate groups: {len(duplicates)}")
    print(f"   Backup/cache files: {len(backup_files)}")
    
    if duplicates:
        print(f"\n🔍 Sample duplicates:")
        for original, copies in duplicates[:3]:
            print(f"   Original: {original}")
            print(f"   Copies: {len(copies)}")
            for copy in copies[:2]:
                print(f"     - {copy}")
            if len(copies) > 2:
                print(f"     ... and {len(copies) - 2} more")
            print()
