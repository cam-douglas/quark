# 🚀 Cognitive Benchmark Suite

This directory contains the suite of cognitive benchmarks designed to validate QUARK's progress during its Stage N4 evolution. These tests are based on established paradigms in cognitive science and are adapted for a computational agent.

## 🎯 Purpose
- **Validate AGI Capabilities**: To quantitatively measure QUARK's cognitive functions against scientific benchmarks.
- **Track Evolution**: To provide clear metrics on the development of higher-order cognitive functions like creative intelligence and superintelligence.
- **Ensure Robustness**: To test the adaptivity and resilience of QUARK's cognitive architecture.

## 🏛️ Architecture
The benchmark suite is designed to be modular and extensible. Key components include:
- `base_benchmark.py`: An abstract base class defining the interface for all cognitive tests. Each benchmark must implement `setup`, `run`, and `evaluate` methods.
- `working_memory_benchmark.py`: A test for spatial and temporal working memory, based on the n-back task.
- `decision_making_benchmark.py`: A test for optimal decision-making under uncertainty, inspired by the Iowa Gambling Task.
- `runner.py`: The main script to discover, run, and report on all benchmark tests.

## 🏃‍♀️ How to Run Benchmarks

To run the entire suite of benchmarks, use the following command from the root directory:
```bash
python testing/cognitive_benchmarks/runner.py
```

To run a specific benchmark:
```bash
python testing/cognitive_benchmarks/runner.py --test working_memory
```

## 📈 Evaluating Results
The benchmark runner will output results to the console and save a detailed report in `data_knowledge/data_repository/metrics/cognitive_benchmarks_report.json`. This report includes performance scores, latency, and other relevant metrics for each test.
