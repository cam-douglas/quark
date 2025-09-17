"""
Human proliferation and core benchmarks (non-spatial) for developmental validation.

Contains cell-cycle lengths, division rates, proliferation indices, clone size
summary, and mid-gestation VZ thickness references used as coarse anchors.
"""

from __future__ import annotations

from typing import Dict

from .proliferation_validation_types import ExperimentalData


def load_human_proliferation_data() -> Dict[str, ExperimentalData]:
    """Return human proliferation/core benchmark metrics keyed for the DB."""

    data: Dict[str, ExperimentalData] = {
        # Cell-cycle length (radial glia)
        "human_8pcw_cell_cycle": ExperimentalData(
            metric_name="cell_cycle_length",
            expected_value=12.0,
            standard_deviation=1.2,
            sample_size=35,
            reference_source="Nowakowski et al. 2016 Cell",
            developmental_stage="8pcw",
        ),
        "human_11pcw_cell_cycle": ExperimentalData(
            metric_name="cell_cycle_length",
            expected_value=17.5,
            standard_deviation=1.8,
            sample_size=28,
            reference_source="Linsley et al. 2019 Dev Cell",
            developmental_stage="11pcw",
        ),
        "human_16pcw_cell_cycle": ExperimentalData(
            metric_name="cell_cycle_length",
            expected_value=22.0,
            standard_deviation=2.0,
            sample_size=30,
            reference_source="Reillo & Borrell 2012 Cereb Cortex",
            developmental_stage="16pcw",
        ),

        # Division rate (λ)
        "human_8pcw_division_rate": ExperimentalData(
            metric_name="division_rate",
            expected_value=0.083,
            standard_deviation=0.008,
            sample_size=35,
            reference_source="Nowakowski et al. 2016 Cell",
            developmental_stage="8pcw",
        ),
        "human_11pcw_division_rate": ExperimentalData(
            metric_name="division_rate",
            expected_value=0.057,
            standard_deviation=0.006,
            sample_size=28,
            reference_source="Linsley et al. 2019 Dev Cell",
            developmental_stage="11pcw",
        ),

        # Proliferation index (Ki-67+)
        "human_8pcw_proliferation_index": ExperimentalData(
            metric_name="proliferation_index",
            expected_value=0.88,
            standard_deviation=0.03,
            sample_size=200,
            reference_source="Nowakowski et al. 2016 Cell",
            developmental_stage="8pcw",
        ),
        "human_16pcw_proliferation_index": ExperimentalData(
            metric_name="proliferation_index",
            expected_value=0.52,
            standard_deviation=0.04,
            sample_size=220,
            reference_source="Reillo & Borrell 2012 Cereb Cortex",
            developmental_stage="16pcw",
        ),

        # Spatial coarse anchors (mid-gestation VZ thickness)
        "human_15pcw_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=120.0,
            standard_deviation=10.0,
            sample_size=6,
            reference_source="Miller et al. 2014 PNAS",
            developmental_stage="15pcw",
        ),
        "human_21pcw_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=70.0,
            standard_deviation=8.0,
            sample_size=5,
            reference_source="Miller et al. 2014 PNAS",
            developmental_stage="21pcw",
        ),

        # Clone size distribution summary
        "human_10_18pcw_clone_mean": ExperimentalData(
            metric_name="clone_size_mean",
            expected_value=28.4,
            standard_deviation=5.1,
            sample_size=420,
            reference_source="Bhaduri et al. 2021 Science",
            developmental_stage="10-18pcw",
        ),
    }

    return data


