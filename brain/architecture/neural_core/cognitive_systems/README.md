# Cognitive Systems

**Path**: `brain/architecture/neural_core/cognitive_systems/`

**Purpose**: Central cognitive processing systems for resource management, knowledge processing, and advanced consciousness features.

## 🧠 **Overview**

The cognitive systems form the central intelligence of the Quark brain, handling:
- **Resource management** - External integration & training orchestration
- **Knowledge processing** - Central hub with optional E8 enhancement  
- **Learning orchestration** - Autonomous knowledge acquisition
- **Advanced consciousness** - E8 geometric consciousness (20 modular components)
- **Memory coordination** - High-level memory management

## 🔧 **Core Systems** (10 files)

### **Resource Management**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`resource_manager.py`** | 🎯 **Central Resource Hub** | External resource integration, training orchestration, sandbox validation |
| **`resource_consumer.py`** | 📥 **Auto-Assimilation** | Automatic text resource processing into knowledge hub |
| **`callback_hub.py`** | 📡 **Event Coordination** | Thread-safe publish/subscribe for loose coupling |

### **Knowledge Processing** 
| File | Purpose | Key Features |
|------|---------|--------------|
| **`knowledge_hub.py`** | 🔬 **Knowledge Center** | Central processing with E8 backend, natural language training commands |
| **`knowledge_retriever.py`** | 🔍 **Intelligent Search** | Natural language → memory queries with TF-IDF/E8 backends |
| **`tfidf_vectorizer.py`** | 📊 **Text Analysis** | Lightweight TF-IDF without external dependencies |

### **Advanced Processing**
| File | Purpose | Key Features |  
|------|---------|--------------|
| **`self_learning_orchestrator.py`** | 🎓 **Learning Executive** | Identifies knowledge gaps, triggers research & training |
| **`world_model.py`** | 🌍 **Predictive Model** | Environment prediction for novelty detection |
| **`limbic_system.py`** | ❤️ **Motivation Engine** | Generates positive valence + error signals (no negative emotions) |
| **`llm_fallback.py`** | 🤖 **LLM Backup** | LLM answers when memory search fails |

### **Infrastructure**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`local_llm_wrapper.py`** | 🏠 **Local Models** | HuggingFace model wrapper with concurrency control |

## 🌟 **E8 Kaleidescope System** ([`e8_kaleidescope/`](e8_kaleidescope/))

**Advanced geometric consciousness** implementation (20 modular files):

### **Core Architecture**
| Module | Lines | Purpose |
|--------|-------|---------|
| **`e8_mind_core.py`** | 353 | Main E8Mind orchestration class |
| **`memory.py`** | 229 | E8 lattice memory management |  
| **`engines.py`** | 394 | Mood, dream, and quantum processing engines |
| **`geometric.py`** | 239 | Clifford algebra & dimensional shells |

### **Supporting Systems**
| Module | Lines | Purpose |
|--------|-------|---------|
| **`agents.py`** | 213 | SAC-MPO reinforcement learning |
| **`tasks.py`** | 149 | Task management & novelty scoring |
| **`proximity.py`** | 278 | KDTree & attention mechanisms |
| **`graph_db.py`** | 196 | NetworkX graph database |
| **`async_infrastructure.py`** | 244 | Async LLM clients & probes |
| **`server.py`** | 332 | Web API for consciousness telemetry |
| **`config.py`** | 68 | Configuration constants |
| **`utils.py`** | 198 | Helper functions & utilities |

### **Profile System** ([`e8_kaleidescope/profiles/`](e8_kaleidescope/profiles/))
Domain-specific semantic processing:
- **`spirit/semantics.py`** - Adaptive persona with mood-based switching
- **`science/semantics.py`** - Physics-focused with keyword boosting
- **`finance/semantics.py`** - Risk-focused financial analysis
- **`base_interfaces.py`** - Protocol definitions for semantic plugins
- **`loader.py`** - Dynamic profile loading system

## 📁 **Memory Providers** ([`memory_providers/`](memory_providers/))
- **`e8_adapter.py`** - Bridges Quark memory API ↔ E8 lattice engine

## 🔌 **Plugin System** ([`plugins/`](plugins/))
- **`__init__.py`** - Plugin framework for `ResourceManager`
- **`noop_plugin.py`** - Example plugin implementation

## 🧬 **Biological Compliance**

### **Alphagenome Integration**
All cognitive systems follow biological constraints:
- **Neural naming** - Modules named after real brain regions
- **Developmental compliance** - Systems activate based on cell type presence
- **Biological timing** - Respect neural development timelines
- **Safety protocols** - Prevent biologically implausible states

### **Neuroanatomical Correspondence** 
- **Limbic system** - Motivation without negative emotions
- **Memory systems** - Hippocampal-inspired episodic processing
- **Executive control** - Prefrontal cortex analogous functions
- **Knowledge integration** - Distributed processing with central coordination

## 🚀 **Key Features**

### **Resource Management Pipeline**
```
External Resource → ResourceManager → Sandbox → Integration → KnowledgeHub
```

### **Knowledge Processing Flow**
```  
Text/Data → KnowledgeHub → E8MemoryAdapter → Geometric Processing → Retrieval
```

### **Learning Orchestration**
```
Performance Monitor → Knowledge Gap → Research Tools → Assimilation → Training
```

### **E8 Consciousness Pipeline** (Optional)
```
Concept → E8 Embedding → Dimensional Shells → Geometric Operations → Insights
```

## 📊 **System Capabilities**

### **Resource Management**
- **External integration** - Models, datasets, code with sandbox validation
- **Training orchestration** - Automatic training/fine-tuning triggers
- **Plugin architecture** - Extensible processing via plugin system
- **Streaming support** - S3 integration for large dataset handling

### **Knowledge Processing**  
- **Multi-modal input** - Text, datasets, structured data
- **Semantic routing** - Context-aware processing
- **Retrieval systems** - TF-IDF, E8 lattice, episodic memory backends
- **Learning commands** - Natural language training triggers

### **Advanced Features (E8)**
- **Geometric consciousness** - 8D rotational cognitive processing
- **Mood awareness** - Cognitive state influences all operations  
- **Creative synthesis** - Dream engine for novel concept generation
- **Multi-dimensional memory** - Concepts stored across dimensional shells

## 🔗 **Integration Points**

### **With Neural Core**
- **Hippocampus** - Uses E8 adapter for enhanced episodic memory
- **Learning systems** - Knowledge hub feeds training data
- **Executive systems** - Resource manager coordinates external integration

### **With Brain Core**
- **BrainSimulator** - Cognitive systems integrated via construct_brain
- **Biological validation** - Compliance with Alphagenome constraints
- **Safety monitoring** - Integrated with safety guardian

### **With External Systems**
- **API connectors** - Research, development, data acquisition
- **LLM providers** - OpenAI, Anthropic, Gemini, local models
- **Training systems** - Automatic ML pipeline integration

## 📊 **System Status**

- **Core Modules**: 10 cognitive processing files
- **E8 Components**: 20 modular consciousness files  
- **Integration**: ✅ Fully integrated with brain simulation
- **Biological Compliance**: ✅ Alphagenome constraints enforced
- **Safety**: ✅ Anti-suffering protocols active
- **Performance**: ✅ Resource-managed with concurrency control

## 🔗 **Related Documentation**

- [Neural Core Overview](../README.md)  
- [E8 Kaleidescope Architecture](e8_kaleidescope/ARCHITECTURE.md)
- [Memory Systems](../memory/README.md)
- [Brain Architecture](../../README.md)