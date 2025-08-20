#!/usr/bin/env python3
"""
Rule File Hierarchy Renamer
Renames all .cursor/rules files according to hierarchical importance structure.
"""

import os
import shutil
from pathlib import Path

def rename_rules_hierarchy():
    """Rename all rule files according to hierarchical structure."""
    
    # Define the hierarchical structure
    hierarchy = {
        # Priority 0 (SUPREME AUTHORITY)
        "compliance_review.md": "00-compliance_review.md",
        
        # Priority 1 (EXECUTIVE LEVEL)
        "cognitive_brain_roadmap.md": "01-cognitive_brain_roadmap.md",
        "roles.md": "02-roles.md",
        
        # Priority 2 (MANAGEMENT LEVEL)
        "master-config.mdc": "03-master-config.mdc",
        "unified_learning_architecture.md": "04-unified_learning_architecture.md",
        "cognitive-brain-rules.mdc": "05-cognitive-brain-rules.mdc",
        
        # Priority 3 (OPERATIONAL LEVEL)
        "brain-simulation-rules.mdc": "06-brain-simulation-rules.mdc",
        "omnirules.mdc": "07-omnirules.mdc",
        "braincomputer.mdc": "08-braincomputer.mdc",
        
        # Priority 4 (SPECIALIZED SYSTEMS)
        "cognitive_load_sleep_system.md": "09-cognitive_load_sleep_system.md",
        "testing_validation_rules.md": "10-testing_validation_rules.md",
        "validation_framework.md": "11-validation_framework.md",
        "multi_model_validation_protocol.md": "12-multi_model_validation_protocol.md",
        
        # Priority 5 (INTEGRATION & ROADMAP)
        "integrated_task_roadmap.md": "13-integrated_task_roadmap.md",
        "master_roadmap_integration.md": "14-master_roadmap_integration.md",
        "roadmap_integration_summary.md": "15-roadmap_integration_summary.md",
        "biological_agi_blueprint.md": "16-biological_agi_blueprint.md",
        
        # Priority 6 (TECHNICAL IMPLEMENTATION)
        "ml_workflow.md": "17-ml_workflow.md",
        "cloud_computing_rules.md": "18-cloud_computing_rules.md",
        "testing_protocol.md": "19-testing_protocol.md",
        "technicalrules.md": "20-technicalrules.md",
        
        # Priority 7 (UTILITY & SUPPORT)
        "cursor_hierarchy.md": "21-cursor_hierarchy.md",
        "activation_triggers.md": "22-activation_triggers.md",
        "enhanced_terminal_rules.md": "23-enhanced_terminal_rules.md",
        "terminal_rules.zsh": "24-terminal_rules.zsh",
        
        # Priority 8 (LEGACY & ARCHIVE)
        "agi_capabilities.md": "25-agi_capabilities.md",
        "organization_summary.md": "26-organization_summary.md",
        "cursor_hierarchy_summary.md": "27-cursor_hierarchy_summary.md",
        "unified_learning_integration_summary.md": "28-unified_learning_integration_summary.md",
        
        # Priority 9 (CONFIGURATION & TEMPLATES)
        "template_inspector.py": "29-template_inspector.py",
        "mutator.py": "30-mutator.py",
        "visualize_graph.py": "31-visualize_graph.py",
        
        # Priority 10 (DNA & BIOLOGICAL)
        "dna_controller.py": "32-dna_controller.py",
        "DNA_CONTROLLER_README.md": "33-DNA_CONTROLLER_README.md",
        "requirements_dna.txt": "34-requirements_dna.txt",
    }
    
    # Get the current directory (should be .cursor/rules)
    rules_dir = Path(".")
    
    if not rules_dir.exists():
        print(f"Error: Current directory {rules_dir} does not exist")
        return
    
    print("🔧 Rule File Hierarchy Renamer")
    print("=" * 50)
    print(f"Working in: {rules_dir.absolute()}")
    
    # Track renamed files
    renamed_count = 0
    skipped_count = 0
    
    # Process each file in the hierarchy
    for old_name, new_name in hierarchy.items():
        old_path = rules_dir / old_name
        new_path = rules_dir / new_name
        
        if old_path.exists():
            try:
                # Check if new name already exists
                if new_path.exists():
                    print(f"⚠️  Skipping {old_name} -> {new_name} (target already exists)")
                    skipped_count += 1
                    continue
                
                # Rename the file
                shutil.move(str(old_path), str(new_path))
                print(f"✅ {old_name} -> {new_name}")
                renamed_count += 1
                
            except Exception as e:
                print(f"❌ Error renaming {old_name}: {e}")
        else:
            print(f"⚠️  File not found: {old_name}")
            skipped_count += 1
    
    # Handle remaining files that weren't in the hierarchy
    remaining_files = []
    for file_path in rules_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith(('0', '1', '2', '3')) and not file_path.name.endswith('.py'):
            remaining_files.append(file_path.name)
    
    if remaining_files:
        print(f"\n📁 Remaining files (not in hierarchy):")
        for file_name in sorted(remaining_files):
            print(f"   - {file_name}")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Renamed: {renamed_count}")
    print(f"   ⚠️  Skipped: {skipped_count}")
    print(f"   📁 Remaining: {len(remaining_files)}")
    
    # Create hierarchy documentation
    create_hierarchy_docs(rules_dir, hierarchy)

def create_hierarchy_docs(rules_dir, hierarchy):
    """Create documentation for the new hierarchy."""
    
    docs_content = """# Cursor Rules Hierarchy Documentation

## Overview
This document describes the hierarchical organization of all rule files in the `.cursor/rules` directory, ordered by importance and authority level.

## Hierarchy Structure

### Priority 0 (SUPREME AUTHORITY)
- **00-compliance_review.md** - Supreme authority, can override any rule set

### Priority 1 (EXECUTIVE LEVEL)
- **01-cognitive_brain_roadmap.md** - Source document for brain simulation architecture
- **02-roles.md** - Neuro-architectural framework implementation

### Priority 2 (MANAGEMENT LEVEL)
- **03-master-config.mdc** - Primary coordination layer
- **04-unified_learning_architecture.md** - Autonomous cognitive evolution framework
- **05-cognitive-brain-rules.mdc** - Brain simulation implementation

### Priority 3 (OPERATIONAL LEVEL)
- **06-brain-simulation-rules.mdc** - Technical brain simulation details
- **07-omnirules.mdc** - General development team (parallel system)
- **08-braincomputer.mdc** - Technical implementation support

### Priority 4 (SPECIALIZED SYSTEMS)
- **09-cognitive_load_sleep_system.md** - Performance management
- **10-testing_validation_rules.md** - Testing protocols
- **11-validation_framework.md** - Validation systems
- **12-multi_model_validation_protocol.md** - Multi-model validation

### Priority 5 (INTEGRATION & ROADMAP)
- **13-integrated_task_roadmap.md** - Task coordination
- **14-master_roadmap_integration.md** - Roadmap integration
- **15-roadmap_integration_summary.md** - Integration summaries
- **16-biological_agi_blueprint.md** - Biological AGI specifications

### Priority 6 (TECHNICAL IMPLEMENTATION)
- **17-ml_workflow.md** - Machine learning workflows
- **18-cloud_computing_rules.md** - Cloud computing specifications
- **19-testing_protocol.md** - Testing protocols
- **20-technicalrules.md** - Technical specifications

### Priority 7 (UTILITY & SUPPORT)
- **21-cursor_hierarchy.md** - Cursor hierarchy management
- **22-activation_triggers.md** - Activation trigger systems
- **23-enhanced_terminal_rules.md** - Terminal rule enhancements
- **24-terminal_rules.zsh** - Terminal rule implementations

### Priority 8 (LEGACY & ARCHIVE)
- **25-agi_capabilities.md** - AGI capability specifications
- **26-organization_summary.md** - Organization summaries
- **27-cursor_hierarchy_summary.md** - Hierarchy summaries
- **28-unified_learning_integration_summary.md** - Learning integration summaries

### Priority 9 (CONFIGURATION & TEMPLATES)
- **29-template_inspector.py** - Template inspection tools
- **30-mutator.py** - Mutation utilities
- **31-visualize_graph.py** - Visualization tools

### Priority 10 (DNA & BIOLOGICAL)
- **32-dna_controller.py** - DNA controller implementation
- **33-DNA_CONTROLLER_README.md** - DNA controller documentation
- **34-requirements_dna.txt** - DNA requirements

## File Mapping
"""
    
    # Add file mapping
    for old_name, new_name in sorted(hierarchy.items(), key=lambda x: x[1]):
        docs_content += f"- **{new_name}** ← {old_name}\n"
    
    docs_content += """
## Usage Guidelines

### For Brain Simulation Development
- **Always start with Priority 0-2** for core architecture decisions
- **Use Priority 3-4** for operational implementation
- **Reference Priority 5-6** for integration and technical details
- **Consult Priority 7-8** for utility and legacy information

### Authority Chain
1. **00-compliance_review.md** has supreme authority
2. **01-02** define the core framework
3. **03-05** coordinate implementation
4. **06-08** provide operational support
5. **09-10** offer specialized tools and configurations

### Activation Triggers
- **Brain simulation**: Activates Priority 0-6
- **General development**: Activates Priority 7-8
- **Compliance review**: Activates Priority 0
- **Integration tasks**: Activates Priority 5-6

## Maintenance Notes
- This hierarchy should be updated when new rule files are added
- Priority levels can be adjusted based on evolving project needs
- All changes must be approved through the compliance review process
"""
    
    # Write documentation
    docs_path = rules_dir / "00-HIERARCHY_DOCUMENTATION.md"
    with open(docs_path, 'w') as f:
        f.write(docs_content)
    
    print(f"\n📚 Created hierarchy documentation: {docs_path}")

if __name__ == "__main__":
    rename_rules_hierarchy()
