# 31 Visualize_Graph
*🟤 CONFIGURATION & TEMPLATES (Priority 31-40)*

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
🎨 Graph Visualizer - Renders semantic network with biological marker coloring
Shows rule relationships and biological compliance visually
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

MARKER_COLORS = {
    "NeuN": "skyblue",      # Neuronal identity
    "GAP43": "orange",      # Learning and plasticity
    "GFAP": "red",          # Structural integrity
    "S100B": "purple",      # Context modulation
    "NSE": "green",         # Neural support
    "Vimentin": "gray",     # Flexibility
    "Tau": "brown",         # Stabilization
    "MBP": "gold",          # Myelination
    "MOG": "pink"           # Oligodendrocyte
}

def load_graph(graph_path):
    """Load graph from JSON file"""
    with open(graph_path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)

def visualize(graph):
    """Visualize the semantic network with biological marker coloring"""
    pos = nx.spring_layout(graph, seed=42, k=3, iterations=50)
    
    # Determine node colors based on biological markers
    colors = []
    node_labels = {}
    
    for node, data in graph.nodes(data=True):
        node_markers = data.get("markers", [])
        base_color = "lightgray"  # Default color
        
        # Find the most prominent marker for coloring
        for marker in node_markers:
            if marker in MARKER_COLORS:
                base_color = MARKER_COLORS[marker]
                break
        
        colors.append(base_color)
        
        # Create node labels (file names)
        file_name = data.get("file", f"node_{node}")
        node_labels[node] = file_name
    
    # Create the visualization
    plt.figure(figsize=(16, 12))
    
    # Draw nodes with biological marker colors
    nx.draw_networkx_nodes(
        graph, pos, 
        node_color=colors, 
        node_size=500, 
        alpha=0.8,
        edgecolors='black',
        linewidths=1
    )
    
    # Draw edges with varying thickness based on weight
    edge_weights = [graph[u][v].get('weight', 0.5) for u, v in graph.edges()]
    nx.draw_networkx_edges(
        graph, pos, 
        alpha=0.4,
        width=[w * 3 for w in edge_weights],
        edge_color='gray'
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        graph, pos, 
        labels=node_labels, 
        font_size=8,
        font_weight='bold'
    )
    
    # Create legend
    legend_elements = []
    for marker, color in MARKER_COLORS.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, label=marker))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("🧬 Semantic Rule Network - Colored by Biological Markers", fontsize=16, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("semantic_network.png", dpi=300, bbox_inches='tight')
    print("✓ Graph visualization saved as 'semantic_network.png'")
    
    plt.show()

def analyze_graph(graph):
    """Analyze graph structure and biological compliance"""
    print(f"\n[📊] Graph Analysis:")
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    
    # Analyze biological marker distribution
    marker_counts = {}
    for _, data in graph.nodes(data=True):
        markers = data.get("markers", [])
        for marker in markers:
            marker_counts[marker] = marker_counts.get(marker, 0) + 1
    
    print(f"\n[🧬] Biological Marker Distribution:")
    for marker, count in sorted(marker_counts.items()):
        print(f"   {marker}: {count} nodes")
    
    # Check for critical markers
    critical_markers = {"GFAP", "NeuN"}
    present_critical = critical_markers.intersection(set(marker_counts.keys()))
    print(f"\n[✅] Critical Markers: {', '.join(present_critical)} present")
    
    if len(present_critical) == len(critical_markers):
        print("   ✓ All critical markers are present")
    else:
        missing = critical_markers - present_critical
        print(f"   ⚠️ Missing: {', '.join(missing)}")

if __name__ == "__main__":
    path = Path("memory/rule_graph.json")
    if not path.exists():
        print("❌ No graph found. Generate one with dna_controller.py first.")
        print("   Run: python3 dna_controller.py")
    else:
        print("🧬 Loading semantic network...")
        g = load_graph(path)
        analyze_graph(g)
        visualize(g)

