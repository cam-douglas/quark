#!/usr/bin/env python3
"""
Quick QUARK State Checker

Run this script to quickly see QUARK's current development status,
recent work, and next steps without reading the full QUARK_STATE.md file.
"""

import os
import re
from pathlib import Path

def extract_state_info():
    """Extract key information from QUARK_STATE.md"""
    state_file = Path("quark_state_system/QUARK_STATE.md")
    
    if not state_file.exists():
        print("❌ QUARK_STATE.md not found!")
        print("Please ensure you're in the QUARK project root directory.")
        return
    
    with open(state_file, 'r') as f:
        content = f.read()
    
    # Extract key information using regex
    current_stage = re.search(r'\*\*Current Development Stage\*\*: (.+)', content)
    overall_progress = re.search(r'\*\*Overall Progress\*\*: (.+)', content)
    next_milestone = re.search(r'\*\*Next Major Milestone\*\*: (.+)', content)
    
    # Extract recent work section
    recent_work_match = re.search(r'### \*\*Embodied Cognition & Brain-Body Integration\*\* 🎯\n\*\*Status\*\*: (.+?)\n', content, re.DOTALL)
    
    # Extract next steps
    next_steps_match = re.search(r'### \*\*Immediate \(This Session\)\*\*:(.+?)(?=###|\Z)', content, re.DOTALL)
    
    print("🧠 QUARK PROJECT STATUS CHECK")
    print("=" * 50)
    
    if current_stage:
        print(f"🎯 Current Stage: {current_stage.group(1)}")
    
    if overall_progress:
        print(f"📊 Overall Progress: {overall_progress.group(1)}")
    
    if next_milestone:
        print(f"🚀 Next Goal: {next_milestone.group(1)}")
    
    print()
    
    if recent_work_match:
        print("🔄 RECENT DEVELOPMENT WORK:")
        print(f"   Status: {recent_work_match.group(1).strip()}")
        print()
    
    if next_steps_match:
        print("📋 IMMEDIATE NEXT STEPS:")
        steps = next_steps_match.group(1).strip()
        # Clean up the steps
        steps = re.sub(r'\n\s*\n', '\n', steps)
        steps = re.sub(r'^\s+', '', steps, flags=re.MULTILINE)
        print(steps)
    
    print()
    print("📖 For complete details, read: QUARK_STATE.md")
    print("🔍 For testing instructions, see the Testing & Validation section")

if __name__ == "__main__":
    extract_state_info()
