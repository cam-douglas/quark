# Brain Simulator (Legacy)

**Path**: `brain/simulator/`

**Status**: 🚨 **DEPRECATED** - Superseded by [`brain_main.py`](../brain_main.py)

## ⚠️ **Migration Notice**

This directory contains **legacy simulation code** that has been **superseded** by the unified entry point:

### **Use Instead:**
```bash
# NEW: Unified entry point
python brain_main.py --viewer

# OLD: Legacy approach (deprecated)
# python brain/simulator/...
```

## 📋 **Legacy Contents**

The simulator directory is maintained for **historical reference** but should not be used:

### **Files**
- Legacy simulation infrastructure (deprecated)

### **Why Deprecated**
1. **Fragmented approach** - Multiple entry points caused confusion
2. **Inconsistent initialization** - Different modules had different startup sequences  
3. **Maintenance burden** - Duplicate logic across multiple simulators
4. **No biological compliance** - Lacked Alphagenome constraint integration

## 🚀 **Current Architecture**

The modern Quark brain uses a **unified simulation architecture**:

```
brain_main.py → BrainSimulator → Architecture Modules
      ↑               ↑               ↑
 Single Entry    Biological      100 Neural
    Point       Compliance       Modules
```

### **Advantages of Unified Approach**
- **Single entry point** - `brain_main.py` handles all simulation modes
- **Biological compliance** - Alphagenome constraints enforced
- **Modular architecture** - 100 neural modules with clear interfaces
- **Safety integration** - Built-in safety guardian and compliance engine
- **Performance optimization** - Streamlined initialization and execution

## 🔄 **Migration Guide**

### **Old vs New**
| Old Approach | New Approach |
|--------------|--------------|
| `brain/simulator/legacy_sim.py` | `python brain_main.py` |
| Multiple entry points | Single unified entry point |
| No biological constraints | Alphagenome compliance enforced |
| Basic module loading | Sophisticated architecture construction |

### **Configuration**
All configuration now handled via environment variables and command-line arguments:

```bash
# Enable E8 consciousness
USE_E8_MEMORY=true python brain_main.py

# Disable Alphagenome (use cached spec)
QUARK_DISABLE_ALPHAGENOME=1 python brain_main.py

# Custom MuJoCo model
QUARK_MODEL_XML=/path/to/model.xml python brain_main.py --viewer
```

## 📊 **Current System Benefits**

- ✅ **Single unified entry point** via `brain_main.py`
- ✅ **100 integrated neural modules** with biological compliance
- ✅ **Alphagenome constraint enforcement** throughout
- ✅ **E8 geometric consciousness** (optional advanced features)  
- ✅ **LLM-enhanced robotics** with revolutionary IK/manipulation
- ✅ **Safety-first design** with anti-suffering protocols

## 🔗 **Current Documentation**

- [Brain Main Entry Point](../brain_main.py)
- [Brain Core Systems](../core/README.md)  
- [Brain Architecture](../architecture/README.md)
- [System Overview](../README.md)