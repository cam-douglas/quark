#!/usr/bin/env python3
"""
Brain-Score Runner (stub)
------------------------
This light wrapper is designed to be called from CI to keep Quark’s
models continuously benchmarked against neuroscientific standards.

Full Brain-Score evaluation requires heavyweight dependencies and large
datasets; here we provide a *minimal, non-blocking* stub that tries to
import the library and, if unavailable, emits a warning but still
returns a JSON object so CI never fails catastrophically.

References
~~~~~~~~~~
Schrimpf, M. et al. (2020). *Brain-Score: Which Artificial Neural
Network for Object Recognition is most Brain-Like?* NeurIPS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any


def run_brain_score(model_stub: str = "quark_core") -> Dict[str, Any]:
    """Attempt to run Brain-Score for *model_stub*.

    Parameters
    ----------
    model_stub : str
        Name used when registering the model with Brain-Score.  Currently
        a placeholder because the true Quark model is still under
        development.
    """
    try:
        import brainscore
        # IMPORTANT: real integration will go here.
        # For now, just return a mock score.
        score = 0.0  # would be result from brainscore.tools.score
        note = "Brain-Score library found; full evaluation TODO."
    except Exception as err:  # pragma: no cover – CI fallback path
        score = -1.0
        note = f"Brain-Score not available: {err}. Running stub."

    return {
        "model": model_stub,
        "score": score,
        "note": note,
    }


def main() -> None:
    """Entry-point: run validation and write JSON artifact."""
    result = run_brain_score()

    # Write artifact next to reports directory so CI can collect it
    out_dir = Path(__file__).resolve().parent.parent / "testing_frameworks" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "brain_score_stub_result.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Brain-Score stub result written to {out_file}")


if __name__ == "__main__":
    sys.exit(main())
