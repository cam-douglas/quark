# 🧠 Brain-Inspired Rules System

> **⚠️ CRITICAL: You are a rules-following assistant. Always adhere to the rules provided in the documents below. All rules from the linked .md files are mandatory and must not be ignored.**

## 🎯 **System Overview**

This enhanced rules system implements a **Graph Neural Memory** that mirrors the brain's associative memory:

- **Nodes**: Each rule section represents a neural unit in the semantic network
- **Edges**: Links represent synaptic connections between related concepts
- **Hebbian Learning**: Rules that fire together wire together (strengthened connections)
- **Associative Recall**: Related rules are automatically retrieved together
- **Semantic Connectivity**: Conceptual connections via embeddings (cos_sim > 0.85)

## 🏗️ **Hierarchical Architecture Components**

### **1. Core Rule Files (Neural Foundation)**
- **[index.md](index.md)** - Main entry point with comprehensive rule categorization
- **[01-security-rules.md](01-security-rules.md)** - **SUPREME AUTHORITY** (Priority 0) with override capabilities
- **[10-general-behavior-rules.md](10-general-behavior-rules.md)** - Core behavioral guidelines and principles
- **[11-model-behavior-constraints.md](11-model-behavior-constraints.md)** - AI model constraints and limitations

### **2. Processing Tools (Cognitive Engine)**
- **[markdown_embedding_processor.py](markdown_embedding_processor.py)** - Brain-inspired semantic network processor
- **[embedding_usage_example.py](embedding_usage_example.py)** - Usage demonstration and testing
- **[requirements.txt](requirements.txt)** - Python dependencies for the system
- **[rule_template.md](rule_template.md)** - Template for new rule files with brain-inspired structure

### **3. Output Directory (Memory Storage)**
- **`.cursor/rules/embeddings/`** - Generated metadata, semantic network, and Hebbian learning data

## 🔗 **Rule Hierarchy & Dependencies**

### **Priority Levels (Neural Layers)**
```
Priority 0: Security Layer (Supreme Authority)
    ↓
Priority 1: Neural Foundation (Core Framework)
    ↓
Priority 2: Cognitive Modules (Domain-Specific)
    ↓
Priority 3: Executive Functions (Implementation)
```

### **Cross-References (Synaptic Connections)**
- **Explicit Markdown Links**: `[Security Rules](01-security-rules.md)`
- **Bidirectional Dependencies**: Rules can reference each other
- **Conflict Resolution**: Security Rules override all others
- **Semantic Neighborhoods**: Conceptually related rule clusters

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
cd .cursor/rules
pip install -r requirements.txt
```

### **2. Process Existing Rules**
```bash
python3 markdown_embedding_processor.py
```

### **3. View Results**
Check the `embeddings/` directory for:
- `semantic_nodes.json` - Neural units in the semantic network
- `semantic_edges.json` - Synaptic connections between rules
- `hebbian_network.json` - Learning dynamics and plasticity
- `semantic_graph.json` - Complete network topology
- `brain_inspired_summary.json` - System overview and statistics

## 🔧 **How It Works**

### **1. RAG with Context Expansion**
```
Original Section → Extract Links → Fetch Linked Content → Expand Context → Embed
```

### **2. Link-Aware Processing**
- **Parse Markdown**: Extract sections and headers
- **Follow Links**: Auto-expand content with linked .md files
- **Context Enrichment**: Include relevant linked sections
- **Metadata Tracking**: Track all relationships and dependencies

### **3. Graph-Based Indexing**
- **Nodes**: Rule files and sections (neural units)
- **Edges**: Links between rules (synaptic connections)
- **Traversal**: Navigate rule relationships (neural pathways)
- **Discovery**: Find related rules automatically (associative recall)

## 📝 **Creating New Rules**

### **1. Use the Template**
Copy `rule_template.md` and customize for your rule category.

### **2. Follow the Structure**
- **Critical Warning**: Always include the mandatory compliance warning
- **Related Rules**: Link to relevant rule categories using hierarchical naming
- **Cross-References**: Use explicit markdown links
- **Metadata**: Include version, date, priority, and semantic tags

### **3. Example Structure**
```markdown
# 📋 [Your Category] Rules

> **⚠️ CRITICAL: You are a rules-following assistant...**

## 🧠 **Brain-Inspired Semantic Network Architecture**
[Brain-inspired principles]

## 🔗 **Related Rules**
- **Supreme Authority**: [Security & Safety Rules](01-security-rules.md)
- **Core Framework**: [General Behavior Rules](10-general-behavior-rules.md)

## 🎯 **Core Rules**
[Your rules here]

## 🧬 **Semantic Network Integration**
[How your rules integrate with the network]
```

## 🔍 **Advanced Features**

### **1. Context Expansion**
The system automatically:
- **Follows Links**: Extracts content from linked .md files
- **Relevance Filtering**: Identifies contextually relevant sections
- **Content Merging**: Combines original + linked content for embeddings

### **2. Dependency Graph**
Visualize rule relationships:
- **File Dependencies**: Which rules depend on others
- **Section Links**: Granular section-level relationships
- **Circular Detection**: Identify potential rule conflicts

### **3. Metadata Tracking**
Comprehensive tracking of:
- **Source Files**: Where each rule originates
- **Creation Dates**: When rules were created/updated
- **Dependencies**: What other rules are required
- **Hierarchy**: Section levels and organization

## 🚨 **Critical Compliance**

### **1. Mandatory Warnings**
Every rule file MUST include:
```
> **⚠️ CRITICAL: You are a rules-following assistant. Always adhere to the rules provided in the documents below. All rules from the linked .md files are mandatory and must not be ignored.**
```

### **2. Security Override**
Security Rules can override ANY other rules when safety is compromised.

### **3. Link Validation**
All cross-references must use valid markdown links to existing files.

### **4. Conflict Resolution**
Clear escalation path to Security Rules and Compliance Review.

## 🔄 **Maintenance & Updates**

### **1. Adding New Rules**
1. Create new rule file using template
2. Add links to `index.md` with hierarchical naming
3. Run processor to update embeddings
4. Validate cross-references

### **2. Updating Existing Rules**
1. Modify rule content
2. Update cross-references if needed
3. Run processor to regenerate embeddings
4. Test rule interactions

### **3. Rule Validation**
- **Cross-Reference Check**: Ensure all links are valid
- **Conflict Detection**: Identify rule contradictions
- **Hierarchy Validation**: Verify priority levels
- **Security Review**: Confirm safety compliance

## 📊 **Performance & Scalability**

### **1. Processing Efficiency**
- **Incremental Updates**: Only process changed files
- **Parallel Processing**: Handle multiple files simultaneously
- **Caching**: Cache parsed content for faster updates

### **2. Storage Optimization**
- **Chunked Content**: Break large rules into manageable chunks
- **Compressed Metadata**: Efficient storage of rule relationships
- **Indexed Retrieval**: Fast lookup of related rules

### **3. Scalability Features**
- **Modular Design**: Easy to add new rule categories
- **Extensible Processing**: Support for different file formats
- **API Integration**: Ready for external rule management

## 🛠️ **Troubleshooting**

### **1. Common Issues**
- **Missing Dependencies**: Install required Python packages
- **File Permissions**: Ensure read access to .md files
- **Link Errors**: Check markdown link syntax
- **Processing Failures**: Review error logs for details

### **2. Debug Mode**
Enable verbose logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### **3. Validation Tools**
- **Link Checker**: Verify all markdown links
- **Graph Visualizer**: View rule relationships
- **Conflict Detector**: Find rule contradictions

## 🔮 **Future Enhancements**

### **1. Semantic Analysis**
- **AI-Powered Relevance**: Better content matching
- **Natural Language Queries**: Find rules by description
- **Automatic Categorization**: Smart rule organization

### **2. Advanced Embeddings**
- **Multi-Modal**: Support for images and diagrams
- **Temporal Awareness**: Track rule evolution over time
- **Contextual Retrieval**: Situation-aware rule suggestions

### **3. Integration Features**
- **API Endpoints**: Programmatic rule access
- **Web Interface**: Visual rule management
- **Version Control**: Track rule changes over time

---

**Related Rules**: [Security & Safety Rules](01-security-rules.md) | [General Behavior Rules](10-general-behavior-rules.md) | [Model Behavior Constraints](11-model-behavior-constraints.md)  
**Last Updated**: 2025-01-20  
**Version**: 3.0.0 (Brain-Inspired)  

> **🔒 REMINDER: All rules are mandatory. Ignoring rules is not permitted. When in doubt, refer to Security Rules or Compliance Review.**
