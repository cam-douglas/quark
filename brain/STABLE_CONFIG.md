# Stable Brain Simulation Configuration

This file documents the stable configuration that ensures the brain simulation runs without issues.

## Critical Settings (DO NOT CHANGE)

### AlphaGenome Configuration
- `multithreading_enabled: False` in `brain/modules/alphagenome_integration/configuration/config_core.py`
- `BiologicalSimulator.run_simulation(steps=100)` not `duration=72.0`

### Brain Module Safety Checks
- All brain modules in `brain/core/step_part1.py` have null safety checks
- Variables are properly scoped to avoid `UnboundLocalError`

### Default Execution Mode
- `--steps` defaults to `float('inf')` (infinite mode)
- `--viewer` defaults to `True` (MuJoCo enabled)
- Foreground execution is the default behavior

### QuarkDriver Rules
- Uses `.quark/rules/*.mdc` files instead of `.quarkrules` file
- Loads all `.mdc` files from the rules directory

### Clean User Interaction
- Verbose logging suppressed during user prompts
- Only "🧠 Quark is thinking..." appears before responses
- `TOKENIZERS_PARALLELISM=false` set during interaction

## Verified Working State
- AlphaGenome: ✅ Full 100-step biological simulation
- MuJoCo: ✅ Viewer with foreground/background detection  
- Interactive: ✅ Clean Ctrl+C user interaction
- Infinite Mode: ✅ Runs indefinitely by default
- No Crashes: ✅ All module safety checks in place

Last verified: $(date)
