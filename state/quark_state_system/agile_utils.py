"""Agile utilities enforcing Phase→Step structure for Quark tasks.

This helper centralises formatting and parsing so all state-system modules
attach the correct labels (Phase X of Y – Step N of M) and honour the user’s
`continuous + <N>` directive.

Integration: Indirect integration via QuarkDriver and AutonomousAgent; orchestrates simulator runs.
Rationale: State system validates, plans, and triggers actions that the simulator executes.
"""
from __future__ import annotations

import re
from typing import Tuple

_PHASE_RE = re.compile(r"phase\s+(\d+)\s+of\s+(\d+)", re.I)
_STEP_RE = re.compile(r"step\s+(\d+)\s+of\s+(\d+)", re.I)
# Accept both 'continuous + N' and 'continuous N'
_CONT_RE_PLUS = re.compile(r"continuous\s*\+\s*(\d+)", re.I)
_CONT_RE_SPACE = re.compile(r"continuous\s+(\d+)", re.I)


def format_phase_step(phase: int, total_phases: int, step: int, total_steps: int) -> str:
    """Return canonical label e.g. 'Phase 2 of 4 – Step 3 of 5'."""
    return f"Phase {phase} of {total_phases} – Step {step} of {total_steps}"


def parse_phase_step(text: str) -> Tuple[int, int, int, int]:
    """Extract phase/step numbers from *text*. Returns (phase, total_phases, step, total_steps).

    Missing numbers default to 0.
    """
    phase, total_phases, step, total_steps = 0, 0, 0, 0
    m1 = _PHASE_RE.search(text)
    if m1:
        phase, total_phases = int(m1.group(1)), int(m1.group(2))
    m2 = _STEP_RE.search(text)
    if m2:
        step, total_steps = int(m2.group(1)), int(m2.group(2))
    return phase, total_phases, step, total_steps


def parse_continuous(text: str) -> int | None:
    """Return N if text contains 'continuous + N' or 'continuous N', else None."""
    m = _CONT_RE_PLUS.search(text)
    if not m:
        m = _CONT_RE_SPACE.search(text)
    if m:
        return int(m.group(1))
    return None
