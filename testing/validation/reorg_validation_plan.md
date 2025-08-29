# Re-Org Validation Plan

This file enumerates the test matrix that must pass **after live moves**.

## 1. Unit Tests
| Suite | Marker | Expected |
|-------|--------|----------|
| Core logic         | `unit`         | 100% pass |
| Brain simulators    | `brain`        | pass |
| ML pipelines        | `ml`           | pass |
| State system        | `state`        | pass |

## 2. Integration Tests
* `tests/integration/test_agent_recommendation.py` – run autonomous agent, verify recommendation JSON.
* `tests/integration/test_brain_sim_end2end.py` – start brain simulator, run one tick.

## 3. Live Visual Smoke (HTML)
* `tests/live_visual/test_embodied_demo.py` – generates live plot; assert HTML file created.

## 4. Performance Baseline
* Benchmark script `testing/benchmarking/reorg_perf.py` runs core loop for 60 s and logs FPS; must be ≥ previous median.

## 5. CI Workflow
GitHub Actions job `reorg-validation.yml` will:
1. Install repo with `pip -e .`.
2. Run `pytest -q`.
3. Archive HTML visual outputs as artifacts.
