# Learning Systems

**Path**: `brain/architecture/neural_core/learning/`

**Purpose**: Advanced learning and adaptation systems including developmental curriculum, LLM-guided training, and biologically-inspired reinforcement learning.

## 🎓 **Learning Architecture Overview**

The learning systems implement sophisticated adaptation mechanisms:
- **Developmental progression** - Human-like motor skill acquisition
- **LLM-guided training** - Revolutionary language-model-supervised learning  
- **Curiosity-driven exploration** - Intrinsic motivation via novelty
- **Multi-modal integration** - Unified robotics dataset processing
- **Biologically-inspired RL** - PPO with developmental curriculum

## 🧠 **Core Learning Components** (7 files)

### **Reinforcement Learning**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`ppo_agent.py`** | 🎯 **Advanced RL** | Full PPO with GAE, normalization, device support (MPS/CUDA/CPU) |
| **`curiosity_driven_agent.py`** | 🔍 **Intrinsic Motivation** | Q-learning with novelty-based rewards, long-term memory integration |
| **`simple_imitator.py`** | 👥 **Imitation Learning** | Specialized Q-learning for pose imitation with exploration bonuses |

### **Memory & Experience**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`long_term_memory.py`** | 📚 **Persistent Memory** | State-action visit counts for lifelong learning, novelty calculation |

### **Developmental Learning**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`developmental_curriculum.py`** | 👶 **Human-like Development** | Motor skill progression (crawling → walking), coordination rewards |

### **🚀 LLM-Enhanced Training**
| File | Purpose | Key Features |
|------|---------|--------------|
| **`llm_guided_training_pipeline.py`** | 🤖 **LLM-Supervised Learning** | Comprehensive training with LLM guidance, developmental curriculum integration |
| **`dataset_integration.py`** | 📊 **Unified Data** | Robotics dataset integration (LLM-IK, manipulation demos, cross-modal mappings) |

## 🧬 **Biological Learning Principles**

### **Developmental Progression**
Following real infant motor development:
```python
# Developmental milestones
milestones = {
    "crawl_forward": MotorMilestone(
        name="crawl_forward",
        stage=DevelopmentalStage.CRAWLING,
        success_criteria={"distance": 2.0}
    )
}
```

### **Curiosity & Exploration**
Biologically-inspired intrinsic motivation:
```python
def get_novelty_score(self, state) -> float:
    """Novelty = 1 / sqrt(1 + visit_count)"""
    visit_count = self.get_total_visits_for_state(state)
    return 1.0 / np.sqrt(1 + visit_count)
```

### **Memory-Based Learning**
- **Long-term memory** tracks all experiences for novelty calculation
- **Episodic memory** provides rich context for learning
- **Working memory** maintains current learning objectives
- **Sleep consolidation** strengthens important memories

## 🚀 **Revolutionary LLM Integration**

### **LLM-Guided Training Pipeline**
**5-phase developmental curriculum** with increasing LLM integration:

#### **Phase Progression**
```python
curriculum_phases = [
    {
        'phase': 0, 'name': 'Proprioceptive Foundation',
        'llm_integration': 'none'  # Basic motor control
    },
    {
        'phase': 1, 'name': 'LLM-IK Learning', 
        'llm_integration': 'teacher'  # LLM provides solutions
    },
    {
        'phase': 2, 'name': 'Basic Object Manipulation',
        'llm_integration': 'collaborator'  # Joint problem-solving  
    },
    {
        'phase': 3, 'name': 'Complex Manipulation Planning',
        'llm_integration': 'partner'  # Equal collaboration
    },
    {
        'phase': 4, 'name': 'Autonomous Learning',
        'llm_integration': 'consultant'  # LLM consulted as needed
    }
]
```

### **Dataset Integration Features**
- **LLM-IK solutions** - Natural language IK problem/solution pairs
- **Manipulation demonstrations** - Human demonstration recordings
- **Cross-modal mappings** - IK solutions ↔ manipulation tasks
- **Prompt templates** - LLM interaction templates for training

## ⚡ **Advanced RL Features**

### **PPO Implementation**
Complete PPO with modern techniques:
- **Generalized Advantage Estimation (GAE)** - Variance-reduced advantage
- **Observation normalization** - Running mean/variance tracking
- **Device support** - MPS (Apple Silicon), CUDA, CPU
- **Gradient clipping** - Stable training
- **Multiple epochs** - Efficient sample utilization

### **Curiosity-Driven Learning**
- **Intrinsic rewards** - Novelty-based exploration bonuses  
- **State discretization** - Efficient tabular representation
- **Long-term tracking** - Persistent experience database
- **Exploration strategies** - Balanced exploration/exploitation

## 🎯 **Learning Pipeline**

### **Training Flow**
```
Raw Experience → Curiosity Calculation → Reward Shaping → PPO Update
      ↓               ↓                     ↓              ↓  
Long-term Memory  Novelty Score     Intrinsic Reward   Policy Update
```

### **Curriculum Flow**
```
Biological Stage → Learning Objectives → Training Data → LLM Guidance
      ↓                    ↓                 ↓             ↓
Skill Targets     Success Criteria    Dataset Selection  Teacher Mode
```

## 📊 **Learning Capabilities**

### **Motor Skill Acquisition**
- **Crawling coordination** - Contralateral limb movements
- **Balance maintenance** - Anti-gravity postural control
- **Object manipulation** - Tool use and articulated objects
- **Locomotion** - Dynamic walking with stability

### **Cognitive Skill Development**
- **Problem solving** - IK reasoning with LLM guidance
- **Knowledge integration** - Cross-modal learning transfer
- **Adaptation** - Novel situation handling
- **Meta-learning** - Learning how to learn more efficiently

## 🔗 **Integration Architecture**

### **With Motor Control**
- **Motor cortex** - Receives high-level goals from learning systems
- **Basal ganglia** - Action selection guided by learned values
- **Cerebellum** - Learns to refine motor commands

### **With Memory Systems**  
- **Long-term memory** - Stores experiences for curiosity calculation
- **Episodic memory** - Rich context for learning episodes
- **Working memory** - Maintains learning objectives

### **With Cognitive Systems**
- **Knowledge hub** - Processes training data and LLM interactions
- **Resource manager** - Coordinates external dataset integration
- **Meta-controller** - Balances intrinsic/extrinsic learning objectives

## 📊 **System Status**

- **Learning Files**: 7 specialized learning modules
- **RL Algorithms**: PPO, Q-learning, curiosity-driven variants
- **LLM Integration**: ✅ Teacher → collaborator → partner → consultant progression
- **Dataset Integration**: ✅ LLM-IK + manipulation demonstrations unified
- **Biological Compliance**: ✅ Developmental progression follows infant motor development
- **Curriculum**: ✅ 5-phase progression from basic control to autonomous learning

## 🔗 **Related Documentation**

- [Motor Control](../motor_control/README.md) 
- [Cognitive Systems](../cognitive_systems/README.md)
- [Memory Systems](../memory/README.md)
- [Neural Core Overview](../README.md)