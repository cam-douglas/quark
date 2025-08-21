# 🚀 QUICK REFERENCE GUIDE - QUARK Experimentation Protocols

**Status**: 🔴 HIGH PRIORITY - Developer Reference
**Last Updated**: August 21, 2025

---

## ⚡ Quick Start

### **1. Run Your First Experiment**
```python
from testing.testing_frameworks.core.experiment_framework import (
    ExperimentConfig, HybridSLMLLMExperiment, ExperimentManager
)

# Create experiment configuration
config = ExperimentConfig(
    experiment_id="my_first_experiment",
    title="My Experiment Title",
    objective="Test specific functionality",
    setup={"task": "specific_task", "parameters": "values"},
    control_baseline="baseline_description",
    metrics=["accuracy", "latency", "cost"],
    neuroalignment_targets=["Hippocampus", "PFC"]
)

# Create and run experiment
experiment = HybridSLMLLMExperiment(config)
manager = ExperimentManager(Path("testing/results_outputs/experiments"))
manager.register_experiment(experiment)
results = manager.run_experiment("my_first_experiment")
```

### **2. Check Results**
```python
# View performance metrics
for metric in results.performance_metrics:
    print(f"{metric.module}: {metric.accuracy:.2f} accuracy, {metric.latency:.2f}s latency")

# View hybrid metrics
print(f"Cost savings: {results.hybrid_metrics['cost_savings']:.1%}")
print(f"Neuroalignment: {results.neuroalignment_scores['overall_neuroalignment']:.2f}")
```

---

## 📋 Required Protocol Compliance

### **✅ Mandatory Requirements**
- **ALL** experiments must use `ExperimentConfig` dataclass
- **ALL** experiments must use `BaseExperiment` or derived classes
- **ALL** results must be stored using `ExperimentResult` dataclass
- **ALL** performance must be tracked using `PerformanceMetrics` dataclass

### **🔍 Validation Checklist**
- [ ] Experiment design validated by CEA
- [ ] Protocol compliance confirmed
- [ ] Neuroalignment targets defined
- [ ] Performance metrics specified
- [ ] Results storage configured
- [ ] Integration points documented

---

## 🧠 Brain Module Integration

### **Available Brain Modules**
```python
# Neural Core Modules
"hippocampus"      # Memory and sequence processing
"prefrontal_cortex" # Working memory and control
"basal_ganglia"    # Action selection and gating
"conscious_agent"  # Global workspace and awareness
"working_memory"   # Short-term memory buffer
"thalamus"         # Sensory relay and attention
```

### **Module-Specific Testing**
```python
# Example: Test hippocampus functionality
hippocampus_config = ExperimentConfig(
    experiment_id="hippocampus_memory_test",
    title="Hippocampus Memory Function Test",
    objective="Validate sequence recall capabilities",
    setup={"task": "sequence_recall", "sequence_length": 10},
    control_baseline="random_sequence_baseline",
    metrics=["recall_accuracy", "recall_latency", "memory_capacity"],
    neuroalignment_targets=["Hippocampus"]
)
```

---

## 🤖 Hybrid SLM+LLM Testing

### **SLM Module Testing**
```python
# Test SLM format compliance
slm_config = ExperimentConfig(
    experiment_id="slm_format_test",
    title="SLM Format Compliance Test",
    objective="Validate format fidelity and speed",
    setup={"task": "format_generation", "format_type": "json"},
    control_baseline="manual_format_generation",
    metrics=["format_accuracy", "generation_speed", "token_efficiency"],
    neuroalignment_targets=["PFC"]
)
```

### **LLM Module Testing**
```python
# Test LLM reasoning capabilities
llm_config = ExperimentConfig(
    experiment_id="llm_reasoning_test",
    title="LLM Abstract Reasoning Test",
    objective="Validate complex reasoning abilities",
    setup={"task": "abstract_reasoning", "complexity": "high"},
    control_baseline="human_expert_baseline",
    metrics=["reasoning_accuracy", "abstraction_depth", "planning_coherence"],
    neuroalignment_targets=["PFC", "Consciousness"]
)
```

### **Hybrid Integration Testing**
```python
# Test hybrid system performance
hybrid_config = ExperimentConfig(
    experiment_id="hybrid_integration_test",
    title="Hybrid SLM+LLM Integration Test",
    objective="Validate module switching and cost efficiency",
    setup={"task": "complex_workflow", "routing": "uncertainty_based"},
    control_baseline="llm_only_baseline",
    metrics=["switching_accuracy", "cost_savings", "overall_efficiency"],
    neuroalignment_targets=["PFC", "Basal_Ganglia", "Consciousness"]
)
```

---

## 📊 Performance Metrics

### **Standard Metrics**
```python
# Basic performance metrics
PerformanceMetrics(
    module="module_name",
    task="task_description",
    accuracy=0.95,           # 0.0 to 1.0
    latency=0.25,            # seconds
    format_compliance=0.98,  # 0.0 to 1.0 (for SLM)
    neuroaligned=True,       # boolean
    additional_metrics={
        "token_usage": 500,
        "cost_per_task": 0.05,
        "confidence": 0.92
    }
)
```

### **Hybrid-Specific Metrics**
```python
# Hybrid system metrics
hybrid_metrics = {
    "switching_accuracy": 0.95,    # Module switching success rate
    "cost_savings": 0.87,          # Cost reduction vs LLM-only
    "routing_efficiency": 0.89,    # Routing decision quality
    "overall_efficiency": 0.93     # Combined performance score
}
```

---

## 🔧 Configuration Options

### **Experiment Configuration**
```python
ExperimentConfig(
    experiment_id="unique_experiment_id",
    title="Human-readable title",
    objective="Clear objective statement",
    setup={
        "task": "specific_task_name",
        "parameters": "task_parameters",
        "environment": "test_environment",
        "constraints": "any_constraints"
    },
    control_baseline="baseline_description",
    metrics=["metric1", "metric2", "metric3"],
    neuroalignment_targets=["brain_module1", "brain_module2"],
    hybrid_config={
        "routing_policy": "uncertainty_based",
        "uncertainty_threshold": 0.7,
        "fallback_strategy": "escalate_to_llm"
    }
)
```

### **Routing Policy Configuration**
```python
routing_policy = {
    "uncertainty_threshold": 0.7,        # Confidence threshold for SLM
    "fallback_strategy": "escalate_to_llm", # What to do when uncertain
    "cost_optimization": True,            # Optimize for cost efficiency
    "performance_priority": "balanced"    # balanced, speed, accuracy
}
```

---

## 📁 File Organization

### **Required File Structure**
```
testing/
├── testing_frameworks/
│   ├── core/
│   │   └── experiment_framework.py      # Core framework
│   ├── quark_experimentation_protocols.md   # Main protocols
│   ├── main_rules_integration.md        # Integration status
│   └── quick_reference_guide.md        # This file
├── results_outputs/
│   └── experiments/                     # Results storage
│       ├── experiment_id_results.json   # Detailed results
│       ├── experiment_id_summary.csv    # Performance summary
│       └── experiment_summary.csv       # All experiments
└── README.md                            # Testing overview
```

### **File Naming Conventions**
- **Experiment IDs**: Use descriptive, lowercase names with underscores
- **Results Files**: Automatically named as `{experiment_id}_results.json`
- **Summary Files**: Automatically named as `{experiment_id}_summary.csv`
- **Integration Files**: Use descriptive names with clear purposes

---

## 🚨 Common Issues & Solutions

### **Issue: Import Errors**
```python
# ❌ Wrong import
from experiment_framework import ExperimentConfig

# ✅ Correct import
from testing.testing_frameworks.core.experiment_framework import ExperimentConfig
```

### **Issue: Configuration Errors**
```python
# ❌ Missing required fields
config = ExperimentConfig(
    experiment_id="test",  # Missing other required fields
)

# ✅ Complete configuration
config = ExperimentConfig(
    experiment_id="test",
    title="Test Experiment",
    objective="Test objective",
    setup={"task": "test_task"},
    control_baseline="test_baseline",
    metrics=["accuracy"],
    neuroalignment_targets=["PFC"]
)
```

### **Issue: Results Not Saving**
```python
# ❌ No output directory specified
manager = ExperimentManager()  # No output directory

# ✅ Specify output directory
manager = ExperimentManager(Path("testing/results_outputs/experiments"))
```

---

## 📞 Support & Resources

### **Documentation Files**
- **Main Protocols**: `quark_experimentation_protocols.md`
- **Integration Status**: `main_rules_integration.md`
- **Core Framework**: `core/experiment_framework.py`

### **Example Implementations**
- **Basic Experiment**: See `experiment_framework.py` main section
- **Hybrid Testing**: See `HybridSLMLLMExperiment` class
- **Results Analysis**: See `evaluate_results` methods

### **Getting Help**
1. **Check this guide** for common solutions
2. **Review main protocols** for detailed requirements
3. **Examine example code** in the framework
4. **Check integration status** for current capabilities

---

## 🎯 Next Steps

### **Immediate Actions**
1. **Read main protocols** to understand requirements
2. **Review core framework** to understand capabilities
3. **Create first experiment** using the quick start guide
4. **Validate results** against neuroalignment targets

### **Advanced Usage**
1. **Custom experiment types** by extending `BaseExperiment`
2. **Advanced routing policies** for hybrid systems
3. **Custom performance metrics** for specific needs
4. **Integration with existing systems** using the framework

---

**Remember**: **ALL EXPERIMENTATION MUST FOLLOW THESE PROTOCOLS** - This is a **SUPREME AUTHORITY** requirement integrated into your main rules!
