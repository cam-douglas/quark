"""Test INM engine and velocity validation"""

import pytest
from brain.modules.developmental_biology.inm_engine import INMEngine
from brain.modules.developmental_biology.inm_parameters import CellCyclePhase
from brain.modules.developmental_biology.inm_velocity_validator import INMVelocityValidator


def test_inm_engine_initialization():
    """Test INM engine initializes correctly"""
    engine = INMEngine()
    assert len(engine.cell_states) == 0
    
    # Initialize a cell
    state = engine.initialize_cell("test_cell", 0.5, CellCyclePhase.G1)
    assert state.cell_id == "test_cell"
    assert state.current_position == 0.5
    assert state.phase == CellCyclePhase.G1


def test_phase_velocity_ordering():
    """Test that phase velocities follow expected ordering S > G2 > M > G1"""
    engine = INMEngine()
    velocities = engine.get_phase_velocities()
    
    # Check ordering
    assert velocities[CellCyclePhase.S] > velocities[CellCyclePhase.G2]
    assert velocities[CellCyclePhase.G2] > velocities[CellCyclePhase.M]
    assert velocities[CellCyclePhase.M] > velocities[CellCyclePhase.G1]


def test_velocity_validator():
    """Test INM velocity validator"""
    engine = INMEngine()
    validator = INMVelocityValidator(tolerance=0.15)
    
    # Validate engine velocities
    results = validator.validate_engine_velocities(engine)
    assert len(results) == 4  # Four phases
    
    # Check ordering validation
    velocities = engine.get_phase_velocities()
    ordering_valid = validator.validate_phase_ordering(velocities)
    assert ordering_valid == True
    
    # Check summary
    summary = validator.get_validation_summary(results)
    assert summary["total_phases"] == 4
    assert summary["pass_rate"] == 1.0  # Should all pass with default parameters


def test_position_updates():
    """Test that position updates work correctly"""
    engine = INMEngine()
    
    # Initialize cell in G1 (should move toward apical)
    engine.initialize_cell("cell1", 0.5, CellCyclePhase.G1)
    
    # Update positions
    changes = engine.update_positions(dt_hours=1.0)
    
    assert "cell1" in changes
    # Should have moved toward target (0.1 for G1)
    assert changes["cell1"] < 0  # Moving apical (decreasing position)
