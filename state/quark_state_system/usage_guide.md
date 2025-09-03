# 🚀 QUARK STATE SYSTEM USAGE GUIDE

**Purpose**: Complete guide for using the QUARK state system  
**Target**: Developers and users working with QUARK

---

## 🎯 **QUICK START COMMANDS**

### **Essential Commands**:
```bash
# Get current status
python QUARK_STATE_SYSTEM.py status

# Get recommendations  
python QUARK_STATE_SYSTEM.py recommendations

# List all tasks with IDs
python QUARK_STATE_SYSTEM.py list-tasks

# Mark task complete
python QUARK_STATE_SYSTEM.py complete-task <task_id>

# Sync all state files
python QUARK_STATE_SYSTEM.py sync
```

---

## 📋 **TASK MANAGEMENT WORKFLOW**

### **1. Getting Tasks**:
```bash
# View current recommendations (top 5 tasks)
python QUARK_STATE_SYSTEM.py recommendations

# List all tasks with full details
python QUARK_STATE_SYSTEM.py list-tasks
```

### **2. Working on Tasks**:
- Tasks follow **Phase → Batch → Step → P/F/A/O** structure
- Each task includes sprint management labels
- Tasks are organized by roadmap sub-headings

### **3. Completing Tasks**:
```bash
# Mark task complete (moves to archive, updates roadmap with DONE)
python QUARK_STATE_SYSTEM.py complete-task stage1_embryonic_rules_foundation-layer_0_1234
```

---

## 🗺️ **ROADMAP INTEGRATION**

### **How It Works**:
1. System scans ALL roadmap files in `management/rules/roadmap/`
2. Extracts tasks from files marked "📋 In Progress"
3. Organizes tasks by original sub-headings
4. Applies sprint-batch-task-management structure

### **File Generation**:
- **`in-progress_tasks.yaml`**: All active roadmap tasks
- **`tasks_archive.yaml`**: Completed tasks with timestamps
- **Roadmap files**: Updated with "DONE" tags when tasks completed

---

## 🔄 **AUTOMATED WORKFLOWS**

### **Task Pipeline Activation**:
- Every call to recommendations/tasks automatically refreshes from roadmaps
- No manual sync required for roadmap changes
- Archive exclusion prevents legacy task contamination

### **Sprint Structure**:
- **Phase**: Extracted from stage number (Stage 1 → Phase 1)
- **Batch**: Assigned by section type (Engineering → B, Goals → A, etc.)
- **Step**: Sequential numbering within batch (1-5 per batch)
- **P/F/A/O**: Mapped to task type and category

---

## 🧠 **NATURAL LANGUAGE INTERFACE**

### **Ask QUARK Directly**:
```python
from state.quark_state_system import ask_quark

# Get recommendations
recs = ask_quark("What are QUARK's recommendations?")

# Get tasks
tasks = ask_quark("What are QUARK's tasks?")

# Get status
status = ask_quark("What is QUARK's status?")
```

---

## 🔍 **TROUBLESHOOTING**

### **Common Issues**:
- **No tasks found**: Check if roadmaps are marked "📋 In Progress"
- **Old tasks appearing**: Run `python QUARK_STATE_SYSTEM.py sync` to refresh
- **Missing integrations**: Check module imports in affected files

### **Debug Commands**:
```bash
# Check what roadmaps are active
python -c "from state.quark_state_system.dynamic_state_summary import get_dynamic_state_summary; print(get_dynamic_state_summary())"

# Force task regeneration
python -c "from state.quark_state_system.task_loader import reset_all, generate_tasks_from_active_roadmaps; reset_all(); generate_tasks_from_active_roadmaps()"
```

---

*This guide reflects the current modular state system architecture.*  
*All operations use live roadmap data, not static documentation.*
