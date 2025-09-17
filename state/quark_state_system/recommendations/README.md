# 🧠 QUARK Recommendations System

**Updated**: 2025-01-04  
**Status**: ✅ Active - Clear separation implemented

## 📋 **System Overview**

The QUARK recommendations system now provides **clear separation** between:

1. **Roadmap Recommendations**: High-level tasks from `management/rules/roadmap/` files
2. **Task Documentation**: Detailed phase-specific tasks from `state/tasks/roadmap_tasks/`

## 🎯 **Query Routing**

### **Roadmap Recommendations** 
*Triggered by queries like:*
- `"quark recommendations"`
- `"quarks recommendations"`
- `"next tasks"`
- `"roadmap tasks"`
- `"what should i do next"`

**Sources**: Only reads from `management/rules/roadmap/*.md` files  
**Excludes**: `/state/tasks/roadmap_tasks/` directory  
**Output**: High-level roadmap guidance with current status

### **Task Documentation**
*Triggered by queries like:*
- `"tasks doc"`
- `"task doc"`
- `"tasks documentation"`
- `"phase tasks"`
- `"show me the tasks doc"`

**Sources**: Only reads from `state/tasks/roadmap_tasks/*.md` files  
**Purpose**: Detailed phase-specific task breakdowns  
**Output**: Granular sub-tasks for current conversation/phase

## 🏗️ **Architecture**

```
state/quark_state_system/recommendations/
├── quark_guidance_router.py         # Main entry point & routing logic
├── recommendation_engine.py         # Core recommendation logic  
├── roadmap_analyzer.py             # Roadmap file analysis
├── task_documentation_handler.py   # Task doc management
└── __init__.py                     # Legacy compatibility & main interface
```

### **Key Components**

#### **QuarkGuidanceRouter** (`quark_guidance_router.py`)
- Main entry point: `handle_user_query(query: str) -> str`
- Routes based on query intent detection
- Coordinates between recommendation and documentation systems

#### **RecommendationEngine** (`recommendation_engine.py`) 
- Context detection: `detect_context_from_query(query: str) -> str`
- Recommendation generation: `get_recommendations_by_context(context, tasks)`
- **Updated**: Excludes roadmap_tasks directory

#### **RoadmapAnalyzer** (`roadmap_analyzer.py`)
- Roadmap status: `get_active_roadmap_status() -> Dict[str, str]`
- Active tasks: `get_active_roadmap_tasks() -> List[str]`
- **Updated**: Only reads from `management/rules/roadmap/`

#### **TaskDocumentationHandler** (`task_documentation_handler.py`)
- Task doc detection: `detect_task_doc_request(query: str) -> bool`
- Phase tasks: `get_current_phase_tasks() -> Dict[str, Any]`
- **New**: Handles detailed task documentation separately

## 🚀 **Usage Examples**

### **Python API**

```python
from state.quark_state_system.recommendations.quark_guidance_router import handle_user_query

# Get roadmap recommendations
recommendations = handle_user_query("quark recommendations")

# Get detailed task documentation  
task_docs = handle_user_query("tasks doc")
```

### **Legacy Compatibility**

```python
from state.quark_state_system.recommendations import QuarkRecommendationsEngine

quark = QuarkRecommendationsEngine()

# Both routing types work through the main interface
roadmap_guidance = quark.provide_intelligent_guidance("what should i do next")
task_documentation = quark.provide_intelligent_guidance("tasks doc")
```

## 📊 **System Status**

Use `get_system_status()` to verify separation is working:

```python
from state.quark_state_system.recommendations.quark_guidance_router import get_system_status

status = get_system_status()
# Returns:
# {
#   "roadmap_recommendations": {"active_roadmaps_count": 6, "active_tasks_count": 10, ...},
#   "task_documentation": {"detailed_task_files_count": 1, ...},
#   "separation_working": True
# }
```

## ⚙️ **Configuration**

### **Excluded Paths** (from roadmap recommendations)
- `/state/tasks/roadmap_tasks/`
- `/state/tasks/chat_tasks`
- `/state/tasks/detailed_`
- `/state/tasks/phase_`

### **Query Keywords**

**Roadmap Recommendations:**
```python
["quark recommendations", "quarks recommendations", "recommend", "recommendation", 
 "next task", "next tasks", "roadmap task", "roadmap tasks", "what should i do", 
 "what to do next", "next step", "next steps", "do next", "continue", "proceed"]
```

**Task Documentation:**
```python
["tasks doc", "task doc", "tasks documentation", "phase tasks", "detailed tasks", 
 "current tasks doc", "task breakdown", "show me the tasks doc", "show tasks doc"]
```

## 🔧 **Maintenance**

### **Adding New Query Types**
1. Update keyword lists in `recommendation_engine.py` and `task_documentation_handler.py`
2. Add routing logic in `quark_guidance_router.py`
3. Test with `handle_user_query()` function

### **Debugging Routing**
```python
from state.quark_state_system.recommendations.recommendation_engine import detect_context_from_query
from state.quark_state_system.recommendations.task_documentation_handler import detect_task_doc_request

query = "your test query"
is_task_doc = detect_task_doc_request(query)
context = detect_context_from_query(query)
print(f"Task Doc: {is_task_doc}, Context: {context}")
```

## ✅ **Verification**

The system has been tested and verified to:
- ✅ Route "quark recommendations" to roadmap system (excludes roadmap_tasks)
- ✅ Route "tasks doc" to documentation system (includes roadmap_tasks)
- ✅ Maintain backward compatibility with existing code
- ✅ Provide clear separation of concerns
- ✅ Handle edge cases and query variations

---

**Last Updated**: 2025-01-04  
**Next Review**: 2025-01-11  
**Status**: ✔ active
