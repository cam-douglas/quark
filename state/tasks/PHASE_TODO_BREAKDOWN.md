# QUARK Roadmap TODO Breakdown (Phases 1–4)

Purpose: granular, verifiable tasks with acceptance gates. All items start as pending to avoid overconfidence. Aligns with `quark_state_system/quark_recommendations.py`.

## Global Gates and Governance
- Gate: Establish phase gates and overconfidence monitor
- Gate: .cursor/rules/index.md enforced in CI
- Gate: Seeds pinned; reproducible runs (REPRO-01)
- Gate: Live interactive visuals for all tests (TEST-01)

## Phase 1 — Foundational Scaffolding (Fetal)
Ready-when:
- Baseline neural dynamics stable for N minutes with fixed seeds
- Unit + integration tests green with live visuals
- Docs include sources and seeds

Tasks:
- Pin Python/toolchain, lint/format/test config (PH1-01)
- Create devcontainer and Makefile (PH1-02)
- Design baseline neural dynamics scaffold (PH1-03)
- Implement thalamic relay minimal gating (PH1-04)
- Implement hippocampal Hebbian memory traces (PH1-05)
- Implement basal ganglia action selection (PH1-06)
- Build proto-cortex sheet connectivity (PH1-07)
- Write unit tests for all modules (PH1-08)
- Add live visualization experiments (PH1-09)
- Document Phase 1 with sources and seeds (PH1-10)

## Phase 2 — Emergence of Core Functions (Neonate N0)
Ready-when:
- Sleep–wake cycles demonstrated; neuromodulation toggles show expected effects
- End-to-end neonatal demo runs with live visuals and seeded reproducibility

Tasks:
- Implement sleep–wake cycle scheduler (PH2-01)
- Add neuromodulator stubs/parameters (DA/NE/5-HT/ACh) (PH2-02)
- Implement salience detection pipeline (PH2-03)
- Implement attention gating network (PH2-04)
- Integrate early default mode network (PH2-05)
- Implement RL actor–critic baseline (PH2-06)
- Edge-case tests: sleep, salience, attention (PH2-07)
- End-to-end neonatal demo with visuals (PH2-08)
- Document Phase 2 with seeds and citations (PH2-09)

## Phase 3 — Higher-Order Cognition (Early Postnatal N1)
Ready-when:
- Working memory tasks passed; global broadcasting observable
- Cross-modal fusion verified; consciousness metrics within expected ranges

Tasks:
- Implement prefrontal working memory buffers (PH3-01)
- Implement global workspace broadcasting (PH3-02)
- Add cerebellar modulation adapters (PH3-03)
- Implement cross-modal sensory fusion (PH3-04)
- Consciousness integration metrics and tests (PH3-05)
- N3 benchmark suite and docs (PH3-06)

## Phase 4 — AGI Capabilities & Full Validation
Ready-when:
- Cognitive benchmark suite green under seeds; reproducible
- Robustness tests pass thresholds; metacognition calibrated
- Lifecycle learning without catastrophic forgetting

Tasks:
- Define Phase 4 acceptance criteria and gates (PH4-AC)
- Build cognitive benchmark harness (4.1) (PH4-01)
- Implement working memory benchmarks (PH4-01A)
- Implement decision-making benchmarks (PH4-01B)
- Implement attentional blink benchmarks (PH4-01C)
- Implement robustness/adaptivity tests (4.2) (PH4-02)
- Noise/occlusion/OOD trials (PH4-02A)
- Generalization to novel tasks (PH4-02B)
- Add metacognition: uncertainty/confidence (4.3) (PH4-03)
- Calibrate confidence vs accuracy curves (PH4-03A)
- Implement lifecycle management & replay (4.4) (PH4-04)
- Catastrophic forgetting tests (PH4-04A)

## Biology & Safety
- Validate biological rules in dna_controller (BIO-01)
- Validate cell_constructor/genome_analyzer/biological_simulator (BIO-02)
- Enforce safety gates and kill-switch tests (SAFE-01)

## Motivation, Governance, and Observability
- Integrate motivation system into roadmap execution (MOT-01)
- Align with quark_recommendations outputs (MOT-02)
- Add provenance, seeds, reproducibility pipeline (REPRO-01)
- Ensure tests stream live interactive visuals (TEST-01)
- Add dashboard for phase gates and status (OBS-01)

Acceptance:
- Each Ready-when met before advancing phases
- All tasks tracked in TODO system; none marked complete without validation
- Overconfidence monitor prevents auto-promotion without green gates

## Entry Points — Brain & Simulation Gates
Ready-when:
- BrainSimulator smoke test runs one step with mocked inputs; no exceptions
- BrainSimulator step-time average over 100 steps ≤ target ms on Mac M2 (recorded)
- EmbodiedBrainSimulation headless smoke test runs 5–10s; actions produced; no emergency shutdowns
- Live interactive run checklist executed manually; user confirms stability
- Seeds and configs pinned; reproducible outputs

Tasks:
- Add BrainSimulator smoke test (ENT-01)
- Add step-time benchmark and budget (ENT-02)
- Enable headless mode flag for embodied_brain_simulation (ENT-03)
- Add headless smoke test for embodied simulation (ENT-04)
- Manual live-run checklist (ENT-05)
- Wire entry-point tests into CI with gates (ENT-06)
- Pin seeds and configs for entry-point runs (ENT-07)
- Expose state-system command to run embodied smoke test (ENT-08)
- Update README run guides with acceptance checks (ENT-09)
- Align recommendations to include entry-point tests (ENT-10)
