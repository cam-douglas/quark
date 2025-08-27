#!/usr/bin/env python3
"""
Update New Integrations Script
Adds internal indexes to all newly integrated brain module files.
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
- **00-UPDATED_HIERARCHY.md** - Updated hierarchy including brain modules

### **🔴 EXECUTIVE LEVEL REFERENCES**
- **01-cognitive_brain_roadmap.md** - Source document for brain simulation architecture
- **01-index.md** - Main index and navigation system
- **01-safety_officer_readme.md** - Safety officer system overview and implementation
- **01-safety_officer_implementation.md** - Safety officer implementation summary and status
- **02-roles.md** - Neuro-architectural framework implementation
- **02-agi_integration_complete.md** - AGI integration status and completion report
- **02-rules_security.md** - Security rules and protocols (HIGH PRIORITY)
- **02-safety_officer_implementation.md** - Safety officer implementation summary

### **🟠 MANAGEMENT LEVEL REFERENCES**
- **03-master-config.mdc** - Primary coordination layer for brain simulation
- **03-integrated-rules.mdc** - Integrated rules system and coordination
- **03-safety_officer_core.py** - Core safety officer implementation
- **04-unified_learning_architecture.md** - Autonomous cognitive evolution framework
- **04-sentinel_agent.py** - Sentinel agent for safety monitoring
- **05-cognitive-brain-rules.mdc** - Brain simulation implementation framework
- **05-alphagenome_integration_readme.md** - Alphagenome integration overview and documentation

### **🟡 OPERATIONAL LEVEL REFERENCES**
- **06-brain-simulation-rules.mdc** - Technical brain simulation details
- **06-biological_simulator.py** - Biological simulation implementation
- **07-omnirules.mdc** - General development team (parallel system)
- **07-genome_analyzer.py** - Genome analysis implementation
- **08-braincomputer.mdc** - Technical implementation support
- **08-cell_constructor.py** - Cell construction implementation
- **09-cognitive_load_sleep_system.md** - Performance management and sleep cycles
- **09-dna_controller.py** - DNA controller implementation
- **10-testing_validation_rules.md** - Testing protocols and validation systems
- **10-test_integration.py** - Integration testing implementation

### **🟢 SPECIALIZED SYSTEMS REFERENCES**
- **11-validation_framework.md** - Validation systems and frameworks
- **11-audit_system.py** - Audit system implementation
- **12-multi_model_validation_protocol.md** - Multi-model validation protocols
- **12-biological_protocols.py** - Biological protocols implementation
- **13-integrated_task_roadmap.md** - Task coordination and integration planning
- **13-safety_constraints.py** - Safety constraints implementation

### **🔵 INTEGRATION & ROADMAP REFERENCES**
- **14-master_roadmap_integration.md** - Master roadmap integration and coordination
- **15-roadmap_integration_summary.md** - Roadmap integration summaries and status
- **16-biological_agi_blueprint.md** - Biological AGI specifications and blueprint
- **17-ml_workflow.md** - Machine learning workflows and processes
- **18-cloud_computing_rules.md** - Cloud computing specifications and rules
- **19-testing_protocol.md** - Testing protocols and procedures
- **20-technicalrules.md** - Technical specifications and rules

### **⚪ UTILITY & SUPPORT REFERENCES**
- **21-cursor_hierarchy.md** - Cursor hierarchy management
- **22-activation_triggers.md** - Activation trigger systems
- **23-enhanced_terminal_rules.md** - Terminal rule enhancements
- **24-terminal_rules.zsh** - Terminal rule implementations
- **25-agi_capabilities.md** - AGI capability specifications
- **26-organization_summary.md** - Organization summaries
- **27-cursor_hierarchy_summary.md** - Hierarchy summaries
- **28-unified_learning_integration_summary.md** - Learning integration summaries
- **29-template_inspector.py** - Template inspection tools
- **30-mutator.py** - Mutation utilities

### **🟤 CONFIGURATION & TEMPLATES REFERENCES**
- **31-visualize_graph.py** - Visualization tools
- **32-dna_controller.py** - DNA controller implementation (legacy)
- **33-DNA_CONTROLLER_README.md** - DNA controller documentation (legacy)
- **34-requirements_dna.txt** - DNA requirements (legacy)
- **35-rules_general.md** - General rules and guidelines
- **36-rules_model_behavior.md** - Model behavior rules and specifications
- **37-explain.mdc** - Explanation system and documentation
- **38-cursor_rules_updater.service** - Cursor rules updater service configuration
- **39-markers.json** - Configuration markers and data
- **40-terminal_rules_backup.zsh** - Backup of terminal rules

### **📦 BACKUP & ARCHIVE REFERENCES**
- **41-brain_simulation_rules_duplicate.md** - Duplicate brain simulation rules (archive)
- **42-brain_simulation_rules_duplicate.mdc** - Duplicate brain simulation rules (archive)
- **43-cognitive_brain_rules_duplicate.md** - Duplicate cognitive brain rules (archive)

### **🔗 QUICK NAVIGATION**
- **Master Index**: 00-MASTER_INDEX.md
- **Updated Hierarchy**: 00-UPDATED_HIERARCHY.md
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
    elif filename.startswith('06-') or filename.startswith('07-') or filename.startswith('08-') or filename.startswith('09-') or filename.startswith('10-'):
        return "🟡 OPERATIONAL LEVEL (Priority 6-10)"
    elif filename.startswith('11-') or filename.startswith('12-') or filename.startswith('13-'):
        return "🟢 SPECIALIZED SYSTEMS (Priority 11-13)"
    elif filename.startswith('14-') or filename.startswith('15-') or filename.startswith('16-') or filename.startswith('17-') or filename.startswith('18-') or filename.startswith('19-') or filename.startswith('20-'):
        return "🔵 INTEGRATION & ROADMAP (Priority 14-20)"
    elif filename.startswith('21-') or filename.startswith('22-') or filename.startswith('23-') or filename.startswith('24-') or filename.startswith('25-') or filename.startswith('26-') or filename.startswith('27-') or filename.startswith('28-') or filename.startswith('29-') or filename.startswith('30-'):
        return "⚪ UTILITY & SUPPORT (Priority 21-30)"
    elif filename.startswith('31-') or filename.startswith('32-') or filename.startswith('33-') or filename.startswith('34-') or filename.startswith('35-') or filename.startswith('36-') or filename.startswith('37-') or filename.startswith('38-') or filename.startswith('39-') or filename.startswith('40-'):
        return "🟤 CONFIGURATION & TEMPLATES (Priority 31-40)"
    elif filename.startswith('41-') or filename.startswith('42-') or filename.startswith('43-') or filename.startswith('44-') or filename.startswith('45-') or filename.startswith('46-'):
        return "📦 BACKUP & ARCHIVE (Priority 41-46)"
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

def update_new_integrations():
    """Update all newly integrated brain module files with internal indexes."""
    
    rules_dir = Path(".")
    
    if not rules_dir.exists():
        print(f"Error: Current directory {rules_dir} does not exist")
        return
    
    print("🔧 Update New Integrations Script")
    print("=" * 50)
    print(f"Working in: {rules_dir.absolute()}")
    
    # Get all numbered files that need indexes
    numbered_files = []
    for file_path in rules_dir.iterdir():
        if file_path.is_file() and file_path.name.startswith(('0', '1', '2', '3', '4')) and file_path.suffix in ['.md', '.mdc', '.py']:
            numbered_files.append(file_path)
    
    # Sort by priority
    numbered_files.sort(key=lambda x: x.name)
    
    print(f"\n📁 Found {len(numbered_files)} numbered files to check:")
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
    
    print(f"\n🎯 All newly integrated files now have comprehensive internal indexes!")
    print(f"   Use 00-UPDATED_HIERARCHY.md for the complete hierarchy")

if __name__ == "__main__":
    update_new_integrations()
