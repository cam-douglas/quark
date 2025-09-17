from state.quark_state_system.agile_utils import (
    format_phase_step,
    parse_continuous,
    parse_phase_step,
)


def test_format_and_parse_roundtrip():
    label = format_phase_step(2, 4, 3, 5)
    phase, total_phases, step, total_steps = parse_phase_step(label)
    assert (phase, total_phases, step, total_steps) == (2, 4, 3, 5)


def test_parse_continuous():
    assert parse_continuous("continuous + 7") == 7
    assert parse_continuous("please run continuous +10 tasks") == 10
    assert parse_continuous("no directive") is None
