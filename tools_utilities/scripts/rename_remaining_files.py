#!/usr/bin/env python3
"""
Remaining Files Renamer
Renames the remaining 14 files according to their importance and priority.
"""

import os
import shutil
from pathlib import Path

def rename_remaining_files():
    """Rename remaining files according to importance hierarchy."""
    
    # Define the remaining files hierarchy
    remaining_hierarchy = {
        # HIGH PRIORITY (Core Documentation & Indexes) - Supreme importance
        "HIERARCHY_ANALYSIS_SUMMARY.md": "00-HIERARCHY_ANALYSIS.md",
        "HIERARCHY_VISUAL_MAP.md": "00-HIERARCHY_VISUAL_MAP.md",
        
        # HIGH-MEDIUM PRIORITY (Core Rules & Integration) - Executive level
        "index.md": "01-index.md",
        "AGI_INTEGRATION_COMPLETE.md": "02-agi_integration_complete.md",
        "integrated-rules.mdc": "03-integrated-rules.mdc",
        
        # MEDIUM PRIORITY (Specialized Rules) - Specialized level
        "rules-general.md": "35-rules_general.md",
        "rules-model-behavior.md": "36-rules_model_behavior.md",
        "rules-security.md": "37-rules_security.md",
        
        # MEDIUM-LOW PRIORITY (Utility & Configuration) - Utility level
        "explain.mdc": "38-explain.mdc",
        "cursor-rules-updater.service": "39-cursor_rules_updater.service",
        "markers.json": "40-markers.json",
        
        # LOW PRIORITY (Backups & Duplicates) - Archive level
        "terminal_rules.zsh.backup.20250818_171515": "41-terminal_rules_backup.zsh",
        "brain-simulation-rules.md": "42-brain_simulation_rules_duplicate.md",
        "brain-simulation-rules.mdc": "43-brain_simulation_rules_duplicate.mdc",
        "cognitive-brain-rules.md": "44-cognitive_brain_rules_duplicate.md",
    }
    
    # Get the current directory
    rules_dir = Path(".")
    
    if not rules_dir.exists():
        print(f"Error: Current directory {rules_dir} does not exist")
        return
    
    print("🔧 Remaining Files Renamer")
    print("=" * 50)
    print(f"Working in: {rules_dir.absolute()}")
    
    # Track renamed files
    renamed_count = 0
    skipped_count = 0
    
    # Process each file in the hierarchy
    for old_name, new_name in remaining_hierarchy.items():
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
    
    # Check for any remaining unnumbered files
    remaining_files = []
    for file_path in rules_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
            remaining_files.append(file_path.name)
    
    if remaining_files:
        print(f"\n📁 Remaining unnumbered files:")
        for file_name in sorted(remaining_files):
            print(f"   - {file_name}")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Renamed: {renamed_count}")
    print(f"   ⚠️  Skipped: {skipped_count}")
    print(f"   📁 Remaining unnumbered: {len(remaining_files)}")
    
    # Create updated hierarchy documentation
    create_updated_hierarchy_docs(rules_dir)

def create_updated_hierarchy_docs(rules_dir):
    """Create updated hierarchy documentation including all 44 files."""
    
    docs_content = """# Complete Cursor Rules Hierarchy (44 Files)

## 🏛️ **SUPREME AUTHORITY (Priority 0)**

### **00-compliance_review.md** 🟢
- **Purpose**: Supreme authority that can override any rule set
- **Authority**: Can veto, override, or modify any rule set or component

### **00-MASTER_INDEX.md** 🟢
- **Purpose**: Comprehensive cross-referenced index of all rule files
- **Authority**: Master navigation and cross-reference system

### **00-HIERARCHY_DOCUMENTATION.md** 🟢
- **Purpose**: Complete hierarchy documentation and guidelines
- **Authority**: Official hierarchy reference and maintenance guide

### **00-HIERARCHY_ANALYSIS.md** 🟢
- **Purpose**: Comprehensive hierarchy analysis and recommendations
- **Authority**: Analysis and planning for hierarchy optimization

### **00-HIERARCHY_VISUAL_MAP.md** 🟢
- **Purpose**: Visual representation of the complete hierarchy
- **Authority**: Visual guide for understanding rule relationships

---

## 🔴 **EXECUTIVE LEVEL (Priority 1)**

### **01-cognitive_brain_roadmap.md** 🔴
- **Purpose**: Source document for brain simulation architecture
- **Authority**: Defines core framework and neuro-architectural specifications

### **01-index.md** 🔴
- **Purpose**: Main index and navigation system
- **Authority**: Primary navigation and organization system

### **02-roles.md** 🔴
- **Purpose**: Neuro-architectural framework implementation
- **Authority**: Defines expert roles and brain module specifications

### **02-agi_integration_complete.md** 🔴
- **Purpose**: AGI integration status and completion report
- **Authority**: AGI integration status and planning

---

## 🟠 **MANAGEMENT LEVEL (Priority 2)**

### **03-master-config.mdc** 🟠
- **Purpose**: Primary coordination layer for brain simulation
- **Authority**: Implements the neuro-architectural framework

### **03-integrated-rules.mdc** 🟠
- **Purpose**: Integrated rules system and coordination
- **Authority**: Manages rule integration and coordination

### **04-unified_learning_architecture.md** 🟠
- **Purpose**: Autonomous cognitive evolution framework
- **Authority**: Implements learning protocols and RSIP

### **05-cognitive-brain-rules.mdc** 🟠
- **Purpose**: Brain simulation implementation framework
- **Authority**: Implements brain simulation specifications

---

## 🟡 **OPERATIONAL LEVEL (Priority 3)**

### **06-brain-simulation-rules.mdc** 🟡
- **Purpose**: Technical brain simulation details
- **Authority**: Provides technical implementation support

### **07-omnirules.mdc** 🟡
- **Purpose**: General development team (parallel system)
- **Authority**: Provides general development support

### **08-braincomputer.mdc** 🟡
- **Purpose**: Technical implementation support
- **Authority**: Provides technical brain-computer interface support

---

## 🟢 **SPECIALIZED SYSTEMS (Priority 4)**

### **09-cognitive_load_sleep_system.md** 🟢
- **Purpose**: Performance management and sleep cycles
- **Authority**: Manages cognitive load and sleep triggers

### **10-testing_validation_rules.md** 🟢
- **Purpose**: Testing protocols and validation systems
- **Authority**: Ensures quality and compliance

### **11-validation_framework.md** 🟢
- **Purpose**: Validation systems and frameworks
- **Authority**: Provides validation infrastructure

### **12-multi_model_validation_protocol.md** 🟢
- **Purpose**: Multi-model validation protocols
- **Authority**: Ensures multi-model consistency

---

## 🔵 **INTEGRATION & ROADMAP (Priority 5)**

### **13-integrated_task_roadmap.md** 🔵
- **Purpose**: Task coordination and integration planning
- **Authority**: Coordinates task execution across all levels

### **14-master_roadmap_integration.md** 🔵
- **Purpose**: Master roadmap integration and coordination
- **Authority**: Integrates all roadmap components

### **15-roadmap_integration_summary.md** 🔵
- **Purpose**: Roadmap integration summaries and status
- **Authority**: Provides integration status and summaries

### **16-biological_agi_blueprint.md** 🔵
- **Purpose**: Biological AGI specifications and blueprint
- **Authority**: Defines biological AGI requirements

---

## 🟣 **TECHNICAL IMPLEMENTATION (Priority 6)**

### **17-ml_workflow.md** 🟣
- **Purpose**: Machine learning workflows and processes
- **Authority**: Defines ML implementation workflows

### **18-cloud_computing_rules.md** 🟣
- **Purpose**: Cloud computing specifications and rules
- **Authority**: Defines cloud computing requirements

### **19-testing_protocol.md** 🟣
- **Purpose**: Testing protocols and procedures
- **Authority**: Defines testing procedures

### **20-technicalrules.md** 🟣
- **Purpose**: Technical specifications and rules
- **Authority**: Defines technical requirements

---

## ⚪ **UTILITY & SUPPORT (Priority 7)**

### **21-cursor_hierarchy.md** ⚪
- **Purpose**: Cursor hierarchy management
- **Authority**: Manages cursor hierarchy organization

### **22-activation_triggers.md** ⚪
- **Purpose**: Activation trigger systems
- **Authority**: Manages rule activation triggers

### **23-enhanced_terminal_rules.md** ⚪
- **Purpose**: Terminal rule enhancements
- **Authority**: Provides enhanced terminal functionality

### **24-terminal_rules.zsh** ⚪
- **Purpose**: Terminal rule implementations
- **Authority**: Implements terminal rules

---

## 🔘 **LEGACY & ARCHIVE (Priority 8)**

### **25-agi_capabilities.md** 🔘
- **Purpose**: AGI capability specifications
- **Authority**: Documents AGI capabilities

### **26-organization_summary.md** 🔘
- **Purpose**: Organization summaries
- **Authority**: Provides organizational summaries

### **27-cursor_hierarchy_summary.md** 🔘
- **Purpose**: Cursor hierarchy summaries
- **Authority**: Provides hierarchy summaries

### **28-unified_learning_integration_summary.md** 🔘
- **Purpose**: Learning integration summaries
- **Authority**: Provides learning integration summaries

---

## 🟤 **CONFIGURATION & TEMPLATES (Priority 9)**

### **29-template_inspector.py** 🟤
- **Purpose**: Template inspection tools
- **Authority**: Provides template inspection functionality

### **30-mutator.py** 🟤
- **Purpose**: Mutation utilities
- **Authority**: Provides mutation functionality

### **31-visualize_graph.py** 🟤
- **Purpose**: Visualization tools
- **Authority**: Provides visualization functionality

---

## 🧬 **DNA & BIOLOGICAL (Priority 10)**

### **32-dna_controller.py** 🧬
- **Purpose**: DNA controller implementation
- **Authority**: Implements DNA control systems

### **33-DNA_CONTROLLER_README.md** 🧬
- **Purpose**: DNA controller documentation
- **Authority**: Documents DNA controller functionality

### **34-requirements_dna.txt** 🧬
- **Purpose**: DNA requirements specification
- **Authority**: Defines DNA system requirements

---

## 🔧 **SPECIALIZED RULES (Priority 11)**

### **35-rules_general.md** 🔧
- **Purpose**: General rules and guidelines
- **Authority**: Provides general rule framework

### **36-rules_model_behavior.md** 🔧
- **Purpose**: Model behavior rules and specifications
- **Authority**: Defines model behavior requirements

### **37-rules_security.md** 🔧
- **Purpose**: Security rules and protocols
- **Authority**: Defines security requirements

---

## ⚙️ **UTILITY & CONFIGURATION (Priority 12)**

### **38-explain.mdc** ⚙️
- **Purpose**: Explanation system and documentation
- **Authority**: Provides explanation and clarification

### **39-cursor_rules_updater.service** ⚙️
- **Purpose**: Cursor rules updater service configuration
- **Authority**: Manages rule update services

### **40-markers.json** ⚙️
- **Purpose**: Configuration markers and data
- **Authority**: Provides configuration data

---

## 📦 **BACKUP & ARCHIVE (Priority 13)**

### **41-terminal_rules_backup.zsh** 📦
- **Purpose**: Backup of terminal rules
- **Authority**: Historical reference and backup

### **42-brain_simulation_rules_duplicate.md** 📦
- **Purpose**: Duplicate brain simulation rules (archive)
- **Authority**: Historical reference and backup

### **43-brain_simulation_rules_duplicate.mdc** 📦
- **Purpose**: Duplicate brain simulation rules (archive)
- **Authority**: Historical reference and backup

### **44-cognitive_brain_rules_duplicate.md** 📦
- **Purpose**: Duplicate cognitive brain rules (archive)
- **Authority**: Historical reference and backup

---

## 🔗 **COMPLETE FILE MAPPING**

### **Priority 0 (Supreme Authority)**: 5 files
- 00-compliance_review.md
- 00-MASTER_INDEX.md
- 00-HIERARCHY_DOCUMENTATION.md
- 00-HIERARCHY_ANALYSIS.md
- 00-HIERARCHY_VISUAL_MAP.md

### **Priority 1 (Executive Level)**: 4 files
- 01-cognitive_brain_roadmap.md
- 01-index.md
- 02-roles.md
- 02-agi_integration_complete.md

### **Priority 2 (Management Level)**: 4 files
- 03-master-config.mdc
- 03-integrated-rules.mdc
- 04-unified_learning_architecture.md
- 05-cognitive-brain-rules.mdc

### **Priority 3 (Operational Level)**: 3 files
- 06-brain-simulation-rules.mdc
- 07-omnirules.mdc
- 08-braincomputer.mdc

### **Priority 4 (Specialized Systems)**: 4 files
- 09-cognitive_load_sleep_system.md
- 10-testing_validation_rules.md
- 11-validation_framework.md
- 12-multi_model_validation_protocol.md

### **Priority 5 (Integration & Roadmap)**: 4 files
- 13-integrated_task_roadmap.md
- 14-master_roadmap_integration.md
- 15-roadmap_integration_summary.md
- 16-biological_agi_blueprint.md

### **Priority 6 (Technical Implementation)**: 4 files
- 17-ml_workflow.md
- 18-cloud_computing_rules.md
- 19-testing_protocol.md
- 20-technicalrules.md

### **Priority 7 (Utility & Support)**: 4 files
- 21-cursor_hierarchy.md
- 22-activation_triggers.md
- 23-enhanced_terminal_rules.md
- 24-terminal_rules.zsh

### **Priority 8 (Legacy & Archive)**: 4 files
- 25-agi_capabilities.md
- 26-organization_summary.md
- 27-cursor_hierarchy_summary.md
- 28-unified_learning_integration_summary.md

### **Priority 9 (Configuration & Templates)**: 3 files
- 29-template_inspector.py
- 30-mutator.py
- 31-visualize_graph.py

### **Priority 10 (DNA & Biological)**: 3 files
- 32-dna_controller.py
- 33-DNA_CONTROLLER_README.md
- 34-requirements_dna.txt

### **Priority 11 (Specialized Rules)**: 3 files
- 35-rules_general.md
- 36-rules_model_behavior.md
- 37-rules_security.md

### **Priority 12 (Utility & Configuration)**: 3 files
- 38-explain.mdc
- 39-cursor_rules_updater.service
- 40-markers.json

### **Priority 13 (Backup & Archive)**: 4 files
- 41-terminal_rules_backup.zsh
- 42-brain_simulation_rules_duplicate.md
- 43-brain_simulation_rules_duplicate.mdc
- 44-cognitive_brain_rules_duplicate.md

---

## 📊 **TOTAL: 44 FILES**

**Complete Hierarchy**: All 44 files are now properly numbered and organized by priority and importance.

**Navigation**: Use 00-MASTER_INDEX.md as the primary navigation system.

**Maintenance**: All files maintain cross-references and internal indexes for complete connectivity.
"""
    
    # Write updated documentation
    docs_path = rules_dir / "00-COMPLETE_HIERARCHY.md"
    with open(docs_path, 'w') as f:
        f.write(docs_content)
    
    print(f"\n📚 Created complete hierarchy documentation: {docs_path}")

if __name__ == "__main__":
    rename_remaining_files()
