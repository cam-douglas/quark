# Memory Systems

**Path**: `brain/architecture/neural_core/memory/`

**Purpose**: Memory infrastructure providing episodic, long-term, and working memory with persistence, synchronization, and protocol abstraction.

## 🧠 **Memory Architecture Overview**

The memory systems implement a biologically-inspired memory hierarchy with:
- **Protocol-based design** - Polymorphic memory backends via `IMemoryStore`
- **Persistence layer** - Atomic snapshots with checksums & schema versioning
- **Cross-memory synchronization** - Episodic ↔ long-term memory bridging
- **Optional E8 enhancement** - Advanced geometric memory processing

## 💾 **Core Memory Components**

### **Protocol Layer**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`imemory_store.py`** | 🔗 **Memory Protocol** | Common interface for all memory backends, enables polymorphic access |

### **Memory Adapters**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`episodic_store.py`** | 🧠 **Episodic Adapter** | Wraps `EpisodicMemory` with `IMemoryStore` protocol, persistence & checksums |
| **`longterm_store.py`** | 📚 **Long-term Adapter** | Wraps `LongTermMemory` counts with `IMemoryStore` protocol |

### **Synchronization**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`memory_synchronizer.py`** | 🔄 **Memory Bridge** | Syncs episodic episodes → long-term counts, maintains forwarding state |

### **Persistence**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`persistence_manager.py`** | 💾 **Snapshot Manager** | Multi-store persistence with resource management integration |

## 🔗 **Memory Integration Architecture**

### **Memory Hierarchy**
```
Working Memory (short-term) 
       ↓
Episodic Memory (hippocampus) ←→ Memory Synchronizer ←→ Long-term Memory (counts)
       ↓                              ↓                         ↓
E8 Memory Adapter               IMemoryStore Protocol      Persistence Manager
       ↓                              ↓                         ↓
Geometric Consciousness         Polymorphic Access        Atomic Snapshots
```

### **Protocol Benefits**
The `IMemoryStore` protocol enables:
- **Backend swapping** - Switch between TF-IDF, E8, native implementations
- **Unified persistence** - Single persistence layer for all memory types
- **Consistent API** - Same interface regardless of backend implementation
- **Testing flexibility** - Mock memory implementations for unit tests

## 🧬 **Biological Memory Features**

### **Episodic Memory** (Hippocampal)
- **Pattern completion** - Partial cues trigger full memories
- **Consolidation strength** - Memories strengthen with importance/access
- **Temporal organization** - Time-tagged with recency boosting
- **E8 enhancement** - Optional geometric memory storage

### **Long-term Memory** (Distributed)
- **Visit count tracking** - Records state-action frequency for curiosity
- **Novelty calculation** - Enables intrinsic motivation via unexplored states
- **Persistent storage** - Survives across simulation sessions
- **Forgetting curves** - Biological decay patterns

### **Working Memory** (Prefrontal) 
- **Capacity limits** - Biologically realistic 7±2 item capacity
- **Cognitive load** - Monitors processing demands
- **Priority queue** - Important information retained longer
- **Neural representations** - Activation patterns for memory items

## ⚡ **Advanced Features**

### **E8 Memory Enhancement** (Optional: `USE_E8_MEMORY=true`)
- **Geometric storage** - Memories as multivectors in 8D space
- **Mood-aware retrieval** - Cognitive state influences memory access
- **Multi-dimensional indexing** - Concepts stored across dimensional shells
- **Creative associations** - Geometric operations enable novel connections

### **Persistence Features**
- **Atomic writes** - Snapshot consistency guarantees
- **Schema versioning** - Forward/backward compatibility
- **CRC32 checksums** - Data integrity validation
- **Compression** - Gzip compression for storage efficiency
- **Auto-cleanup** - Maintains reasonable snapshot history

### **Synchronization Features**
- **Bidirectional sync** - Episodic episodes ↔ long-term counts
- **Hash-based projection** - Episodes mapped to state-action tuples
- **Idempotent operation** - Safe to run multiple times
- **Incremental updates** - Only processes new episodes

## 🛠️ **Technical Implementation**

### **Memory Store Protocol**
```python
class IMemoryStore(Protocol):
    def store_episode(self, content: Dict, context: Dict) -> str
    def retrieve_episode(self, query: Dict, max_results: int = 5) -> List[Any]
    def save(self, path: str) -> None
    def load(self, path: str) -> None  
    def get_stats(self) -> Dict[str, Any]
```

### **Synchronization Process**
```python
# Memory synchronizer operation
def sync(self) -> Dict[str, Any]:
    for episode_id, episode in episodic_memory.episodes.items():
        if episode_id not in forwarded_set:
            state, action = episode_to_state_action(episode)
            long_term_memory.record_experience(state, action)
            forwarded_set.add(episode_id)
```

### **Persistence Flow**
```python
# Atomic persistence with integrity  
def save(self, path: str):
    payload = {"schema_version": VERSION, "data": episodes}
    raw = json.dumps(payload)
    wrapper = {"crc32": zlib.crc32(raw), "data": payload}
    # Atomic write via temporary file
```

## 🧬 **Biological Compliance**

### **Memory Consolidation Patterns**
- **Sleep-dependent consolidation** - Integration with sleep consolidation engine
- **Importance-based retention** - Biologically-relevant memories preserved
- **Forgetting curves** - Natural decay following biological patterns
- **Replay mechanisms** - Memory strengthening through replay

### **Neural Organization**
- **Hippocampal episodic** - Detailed event memories with pattern completion
- **Distributed long-term** - Statistical learning across cortical areas
- **Working memory** - Prefrontal cortex-like temporary storage
- **Cross-area synchronization** - Biologically-inspired memory transfer

## 📊 **Memory Performance**

### **Storage Efficiency**
- **Compression** - Gzip reduces storage by ~70%
- **Incremental sync** - Only new episodes processed
- **Hash deduplication** - Prevents duplicate episode storage
- **Schema evolution** - Graceful handling of format changes

### **Retrieval Performance**  
- **TF-IDF indexing** - O(log n) similarity search
- **E8 lattice** - Geometric similarity in high-dimensional space
- **Caching** - Frequent queries cached for speed
- **Batch operations** - Efficient multi-query processing

## 🔗 **Integration Points**

### **With Hippocampus**
- **`episodic_memory.py`** - Core episodic memory with E8 integration
- **Pattern completion** - Partial cues trigger full memory retrieval
- **Consolidation cycles** - Sleep-based memory strengthening

### **With Learning Systems**  
- **Curiosity agents** - Use long-term memory for novelty calculation
- **Training orchestrator** - Memory gaps trigger knowledge acquisition
- **Developmental curriculum** - Memory guides learning progression

### **With Cognitive Systems**
- **Knowledge hub** - Central processing with memory backend
- **Resource manager** - Integrates external data into memory
- **Safety guardian** - Monitors memory-based error patterns

## 📊 **System Status**

- **Memory Modules**: 5 core infrastructure files
- **Protocol Compliance**: ✅ All backends implement `IMemoryStore`
- **Persistence**: ✅ Atomic snapshots with integrity validation
- **Synchronization**: ✅ Cross-memory consistency maintained
- **E8 Enhancement**: ✅ Optional geometric consciousness available
- **Biological Fidelity**: ✅ Neuroanatomically-inspired organization

## 🔗 **Related Documentation**

- [Hippocampus](../hippocampus/README.md)
- [Working Memory](../working_memory/README.md)  
- [Cognitive Systems Overview](README.md)
- [E8 Kaleidescope](cognitive_systems/e8_kaleidescope/)