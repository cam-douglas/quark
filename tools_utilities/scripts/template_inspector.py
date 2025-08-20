# 29 Template_Inspector
*⚪ UTILITY & SUPPORT (Priority 21-30)*

## 📚 **INTERNAL INDEX & CROSS-REFERENCES**

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

#!/usr/bin/env python3
"""
🔍 Template Inspector - Validates biological compliance of templates
Checks for critical markers, regulatory pathways, and biological integrity
"""

import json
from pathlib import Path

CRITICAL_MARKERS = {"GFAP", "NeuN"}
REGULATORY_MAP = {
    "GAP43": {"activates": ["NeuN"], "coexpresses": ["NSE"]},
    "GFAP": {"stabilizes": ["Vimentin"], "inhibits": ["Tau"]},
    "S100B": {"coexpresses": ["GFAP"]},
    "MBP": {"insulates": ["NeuN"]},
    "MOG": {"coexpresses": ["MBP"]}
}

MARKERS_PATH = Path("markers.json")
META_PATH = Path("memory/metadata.json")

def load_markers():
    """Load biological markers configuration"""
    if not MARKERS_PATH.exists():
        raise FileNotFoundError("markers.json not found.")
    return json.load(MARKERS_PATH.open())

def load_metadata():
    """Load metadata from DNA controller"""
    if not META_PATH.exists():
        raise FileNotFoundError("metadata.json not found. Run dna_controller.py first.")
    return json.load(META_PATH.open())

def inspect_templates():
    """Inspect biological compliance of templates"""
    print("\n[🔍] Inspecting biological compliance of templates...")
    
    try:
        markers = load_markers()
        print(f"✓ Loaded {len(markers)} template markers")
        
        # Gather all expressed markers
        expressed = set()
        for file_markers in markers.values():
            expressed.update(file_markers)
        
        print(f"✓ Total biological markers: {len(expressed)}")
        print(f"   Markers: {', '.join(sorted(expressed))}")
        
        # Check for critical markers
        missing = CRITICAL_MARKERS - expressed
        if missing:
            print(f"❌ Missing critical marker(s): {', '.join(missing)}")
        else:
            print("✅ All critical markers present: GFAP, NeuN")
        
        # Regulatory integrity check
        print("\n[🧬] Checking regulatory pathways:")
        for regulator, rules in REGULATORY_MAP.items():
            if regulator in expressed:
                for effect_type, targets in rules.items():
                    for target in targets:
                        if target in expressed:
                            print(f"✅ {regulator} {effect_type} {target}")
                        else:
                            print(f"⚠️ {regulator} {effect_type} {target} — target missing")
        
        # Check marker distribution
        print(f"\n[📊] Marker distribution:")
        marker_counts = {}
        for file_markers in markers.values():
            for marker in file_markers:
                marker_counts[marker] = marker_counts.get(marker, 0) + 1
        
        for marker, count in sorted(marker_counts.items()):
            print(f"   {marker}: {count} templates")
        
        print("\n[✓] Inspection complete.\n")
        
    except Exception as e:
        print(f"❌ Error during inspection: {e}")

if __name__ == "__main__":
    inspect_templates()

