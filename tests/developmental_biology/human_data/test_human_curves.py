import pytest
import numpy as np

from brain.modules.developmental_biology import human_data_utils

@pytest.mark.parametrize("pcw", [5.0, 8.0, 11.0, 16.0])
def test_interpolation_cell_cycle(pcw):
    value = human_data_utils.interpolate("cell_cycle_length", pcw)
    assert value > 0

@pytest.mark.parametrize("pcw", [10.0, 14.0, 18.0])
def test_clone_size_mean(pcw):
    value = human_data_utils.interpolate("clone_size_mean", pcw)
    assert value > 0
