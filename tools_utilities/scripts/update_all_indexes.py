#!/usr/bin/env python3
"""
Update All Indexes Script
Efficiently updates all remaining rule files with comprehensive internal indexes.
"""

import os
from pathlib import Path

def get_index_template():
    """Get the comprehensive internal index template."""
    return """## 📚 **INTERNAL INDEX & CROSS-REFERENCES**

### **🏛️ SUPREME AUTHORITY REFERENCES**
- **00-compliance_review.md** - Supreme authority, can override any rule set
- **00-MASTER_INDEX.md** - Comprehensive cross-referenced index of all rule files
- **00-HIERARCHY_DOCUMENTATION.md** - Complete hierarchy documentation and guidelines
- **00-HIERARCHY_ANALYSIS.md** - Comprehensive hierarchy analysis and recommendations
- **00-HIERARCHY_VISUAL_MAP.md** - Visual representation of the complete hierarchy

### **🔴 EXECUTIVE LEVEL REFERENCES**
- **01-cognitive_brain_roadmap.md** - Source document for brain simulation architecture
- **01-index.md** - Main index and navigation system
- **02-roles.md** - Neuro-architectural framework implementation
- **02-agi_integration_complete.md** - AGI integration status and completion report

### **🟠 MANAGEMENT LEVEL REFERENCES**
- **03-master-config.mdc** - Primary coordination layer for brain simulation
- **03-integrated-rules.mdc** - Integrated rules system and coordination
- **04-unified_learning_architecture.md** - Autonomous cognitive evolution framework
- **05-cognitive-brain-rules.mdc** - Brain simulation implementation framework

### **🟡 OPERATIONAL LEVEL REFERENCES**
- **06-brain-simulation-rules.mdc** - Technical brain simulation details
- **07-omnirules.mdc** - General development team (parallel system)
- **08-braincomputer.mdc** - Technical implementation support

### **🟢 SPECIALIZED SYSTEMS REFERENCES**
- **09-cognitive_load_sleep_system.md** - Performance management and sleep cycles
- **10-testing_validation_rules.md** - Testing protocols and validation systems
- **11-validation_framework.md** - Validation systems and frameworks
- **12-multi_model_validation_protocol.md** - Multi-model validation protocols

### **🔵 INTEGRATION & ROADMAP REFERENCES**
- **13-integrated_task_roadmap.md** - Task coordination and integration planning
- **14-master_roadmap_integration.md** - Master roadmap integration and coordination
- **15-roadmap_integration_summary.md** - Roadmap integration summaries and status
- **16-biological_agi_blueprint.md** - Biological AGI specifications and blueprint

### **🟣 TECHNICAL IMPLEMENTATION REFERENCES**
- **17-ml_workflow.md** - Machine learning workflows and processes
- **18-cloud_computing_rules.md** - Cloud computing specifications and rules
- **19-testing_protocol.md** - Testing protocols and procedures
- **20-technicalrules.md** - Technical specifications and rules

### **⚪ UTILITY & SUPPORT REFERENCES**
- **21-cursor_hierarchy.md** - Cursor hierarchy management
- **22-activation_triggers.md** - Activation trigger systems
- **23-enhanced_terminal_rules.md** - Terminal rule enhancements
- **24-terminal_rules.zsh** - Terminal rule implementations

### **🔘 LEGACY & ARCHIVE REFERENCES**
- **25-agi_capabilities.md** - AGI capability specifications
- **26-organization_summary.md** - Organization summaries
- **27-cursor_hierarchy_summary.md** - Hierarchy summaries
- **28-unified_learning_integration_summary.md** - Learning integration summaries

### **🟤 CONFIGURATION & TEMPLATES REFERENCES**
- **29-template_inspector.py** - Template inspection tools
- **30-mutator.py** - Mutation utilities
- **31-visualize_graph.py** - Visualization tools

### **🧬 DNA & BIOLOGICAL REFERENCES**
- **32-dna_controller.py** - DNA controller implementation
- **33-DNA_CONTROLLER_README.md** - DNA controller documentation
- **34-requirements_dna.txt** - DNA requirements

### **🔧 SPECIALIZED RULES REFERENCES**
- **35-rules_general.md** - General rules and guidelines
- **36-rules_model_behavior.md** - Model behavior rules and specifications
- **37-rules_security.md** - Security rules and protocols

### **⚙️ UTILITY & CONFIGURATION REFERENCES**
- **38-explain.mdc** - Explanation system and documentation
- **39-cursor_rules_updater.service** - Cursor rules updater service configuration
- **40-markers.json** - Configuration markers and data

### **📦 BACKUP & ARCHIVE REFERENCES**
- **41-terminal_rules_backup.zsh** - Backup of terminal rules
- **42-brain_simulation_rules_duplicate.md** - Duplicate brain simulation rules (archive)
- **43-brain_simulation_rules_duplicate.mdc** - Duplicate brain simulation rules (archive)
- **44-cognitive_brain_rules_duplicate.md** - Duplicate cognitive brain rules (archive)

### **🔗 QUICK NAVIGATION**
- **Master Index**: 00-MASTER_INDEX.md
- **Complete Hierarchy**: 00-COMPLETE_HIERARCHY.md
- **Compliance Review**: 00-compliance_review.md
- **Visual Map**: 00-HIERARCHY_VISUAL_MAP.md

---

"""

def get_priority_info(filename):
    """Get priority information for a file based on its number prefix."""
    if filename.startswith('00-'):
        return "🏛️ SUPREME AUTHORITY (Priority 0)"
    elif filename.startswith('01-') or filename.startswith('02-'):
        return "🔴 EXECUTIVE LEVEL (Priority 1-2)"
    elif filename.startswith('03-') or filename.startswith('04-') or filename.startswith('05-'):
        return "🟠 MANAGEMENT LEVEL (Priority 3-5)"
    elif filename.startswith('06-') or filename.startswith('07-') or filename.startswith('08-'):
        return "🟡 OPERATIONAL LEVEL (Priority 6-8)"
    elif filename.startswith('09-') or filename.startswith('10-') or filename.startswith('11-') or filename.startswith('12-'):
        return "🟢 SPECIALIZED SYSTEMS (Priority 9-12)"
    elif filename.startswith('13-') or filename.startswith('14-') or filename.startswith('15-') or filename.startswith('16-'):
        return "🔵 INTEGRATION & ROADMAP (Priority 13-16)"
    elif filename.startswith('17-') or filename.startswith('18-') or filename.startswith('19-') or filename.startswith('20-'):
        return "🟣 TECHNICAL IMPLEMENTATION (Priority 17-20)"
    elif filename.startswith('21-') or filename.startswith('22-') or filename.startswith('23-') or filename.startswith('24-'):
        return "⚪ UTILITY & SUPPORT (Priority 21-24)"
    elif filename.startswith('25-') or filename.startswith('26-') or filename.startswith('27-') or filename.startswith('28-'):
        return "🔘 LEGACY & ARCHIVE (Priority 25-28)"
    elif filename.startswith('29-') or filename.startswith('30-') or filename.startswith('31-'):
        return "🟤 CONFIGURATION & TEMPLATES (Priority 29-31)"
    elif filename.startswith('32-') or filename.startswith('33-') or filename.startswith('34-'):
        return "🧬 DNA & BIOLOGICAL (Priority 32-34)"
    elif filename.startswith('35-') or filename.startswith('36-') or filename.startswith('37-'):
        return "🔧 SPECIALIZED RULES (Priority 35-37)"
    elif filename.startswith('38-') or filename.startswith('39-') or filename.startswith('40-'):
        return "⚙️ UTILITY & CONFIGURATION (Priority 38-40)"
    elif filename.startswith('41-') or filename.startswith('42-') or filename.startswith('43-') or filename.startswith('44-'):
        return "📦 BACKUP & ARCHIVE (Priority 41-44)"
    else:
        return "📄 UNCATEGORIZED"

def update_file_with_index(file_path):
    """Update a single file with the internal index."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file already has an index
        if "## 📚 **INTERNAL INDEX & CROSS-REFERENCES**" in content:
            print(f"⚠️  {file_path.name} already has an index - skipping")
            return False
        
        # Get priority info
        priority_info = get_priority_info(file_path.name)
        
        # Create the new index section
        new_index = f"""# {file_path.stem.replace('-', ' ').title()}
*{priority_info}*

{get_index_template()}"""
        
        # Add the new index to the beginning of the file
        updated_content = new_index + content
        
        # Write the updated content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {file_path.name} with internal index")
        return True
        
    except Exception as e:
        print(f"❌ Error updating {file_path.name}: {e}")
        return False

def update_all_files():
    """Update all numbered rule files with internal indexes."""
    
    rules_dir = Path(".")
    
    if not rules_dir.exists():
        print(f"Error: Current directory {rules_dir} does not exist")
        return
    
    print("🔧 Update All Indexes Script")
    print("=" * 50)
    print(f"Working in: {rules_dir.absolute()}")
    
    # Get all numbered files
    numbered_files = []
    for file_path in rules_dir.iterdir():
        if file_path.is_file() and file_path.name.startswith(('0', '1', '2', '3', '4')) and file_path.suffix in ['.md', '.mdc']:
            numbered_files.append(file_path)
    
    # Sort by priority
    numbered_files.sort(key=lambda x: x.name)
    
    print(f"\n📁 Found {len(numbered_files)} numbered files to update:")
    for file_path in numbered_files:
        print(f"   - {file_path.name}")
    
    # Update each file
    updated_count = 0
    skipped_count = 0
    
    for file_path in numbered_files:
        if update_file_with_index(file_path):
            updated_count += 1
        else:
            skipped_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Updated: {updated_count}")
    print(f"   ⚠️  Skipped: {skipped_count}")
    print(f"   📁 Total processed: {len(numbered_files)}")
    
    print(f"\n🎯 All rule files now have comprehensive internal indexes!")
    print(f"   Use 00-MASTER_INDEX.md as the primary navigation system")

if __name__ == "__main__":
    update_all_files()
