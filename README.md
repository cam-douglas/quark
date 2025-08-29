# 🚀 QUARK: A Self-Determined, Biologically-Compliant AGI Framework

> **Project Overview**  
> QUARK is a modular brain-simulation and machine-learning platform that spans molecular genetics → morphogenesis → micro-circuits → cognition.  
> It emphasises biological fidelity, reproducibility and a roadmap-driven autonomous development process.

---

## 🤖 **New Operational Mode: The Quark Active Driver**

**The Quark system is now in a continuous, self-determined execution mode.**

All interactions and development are orchestrated by the **Quark Driver** (`quark_state_system/quark_driver.py`), a new core component that ensures every action is aligned with the project's strategic goals.

### How It Works:
1.  **Roadmap-Driven**: The system's primary directive comes from a unified understanding of all project roadmaps (`management/rules/roadmap/`).
2.  **Biologically Compliant**: Every action is validated by a **Compliance Engine** (`brain_modules/alphagenome_integration/compliance_engine.py`) to ensure it follows the centralized biological rules (`management/rules/biological_constraints.py`).
3.  **Autonomous Execution**: A new **Autonomous Agent** (`quark_state_system/autonomous_agent.py`) is responsible for formulating and executing plans to achieve the next roadmap goal.
4.  **Prompt Guardian**: Every prompt is intercepted and validated by a **Prompt Guardian** (`quark_state_system/prompt_guardian.py`) to ensure alignment and safety.

---

## 🚀 **Getting Started: Interacting with the Active System**

The main entry point for all operations remains `QUARK_STATE_SYSTEM.py`.

### 1. Check the Current Goal
To understand the system's current priority, simply ask:
> "what is quarks next steps according to the state system?"

### 2. Automating the System
You have two primary ways to drive the system forward:

**A) Step-by-Step Automation (Recommended):**
Use a generic prompt to authorize the agent to execute the *next single task*.
> "proceed"
> "continue"
> "evolve"

The agent will execute one goal and then wait for your next command.

**B) Continuous Full Automation:**
To run the agent continuously until all roadmap goals are complete, use the following command:
```bash
python3 QUARK_STATE_SYSTEM.py run-continuous
```

### 3. All Available Commands
```bash
# Get an explanation of the new active driver mode
python3 QUARK_STATE_SYSTEM.py activate

# Run the full automation loop
python3 QUARK_STATE_SYSTEM.py run-continuous

# Execute a single roadmap goal
python3 QUARK_STATE_SYSTEM.py execute

# Check the system's current status
python3 QUARK_STATE_SYSTEM.py status

# Get intelligent recommendations (based on old system)
python3 QUARK_STATE_SYSTEM.py recommendations

# Manually sync the state files
python3 QUARK_STATE_SYSTEM.py sync

# See all commands
python3 QUARK_STATE_SYSTEM.py help
```

---

## 📁 Repository Layout (high-level)

| Path | Purpose |
|------|---------|
| `brain/` | Core brain simulation and architecture code |
| `ml/` | Machine-learning training systems |
| `state/` | Quark state-system logic & tasks |
| `docs/` | Project documentation (see [`docs/INDEX.md`](docs/INDEX.md)) |
| `tests/` | Smoke + regression tests |
| `config/testing/pytest.ini` | Pytest config used by CI |

---

## 🧪 Benchmarks & Tests

Run the smoke & docs regression tests:

```bash
pytest -c config/testing/pytest.ini
```

Benchmarks live under `tests/benchmarking/` and `tests/cognitive_benchmarks/`.
