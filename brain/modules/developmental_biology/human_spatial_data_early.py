"""
Early human spatial metrics (≤ ~10 pcw) for developmental biology validation.

This module is split out to maintain architecture compliance (≤300 LOC per file).
It provides ventricular zone (VZ) thickness, lumen diameter, and apical nuclei
linear density metrics from human embryonic literature.
"""

from __future__ import annotations

from typing import Dict

from .proliferation_validation_types import ExperimentalData


def load_human_spatial_data_early() -> Dict[str, ExperimentalData]:
    """Return early-stage human spatial metrics keyed for the experimental DB."""

    data: Dict[str, ExperimentalData] = {
        # Cortex — VZ thickness (human)
        "human_6pcw_cortex_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=80.0,
            standard_deviation=10.0,
            sample_size=1,
            reference_source=(
                "Embryonic and Early Fetal Development of the Human Neocortex (J Neurosci 2000; "
                "doi:10.1523/JNEUROSCI.20-05-01858.2000; pmid:10684887; Fig 1E; scale bar 50 µm)"
            ),
            developmental_stage="6pcw",
        ),
        "human_9pcw_cortex_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=50.0,
            standard_deviation=10.0,
            sample_size=1,
            reference_source=(
                "Cerebral Cortex 2008; 'A Molecular Neuroanatomical Study of the Developing Human "
                "Neocortex from 8 to 17 Postconceptional Weeks' (doi:10.1093/cercor/bhm184; "
                "pmid:17965125; Fig 2, 9 pcw; scale bar 200 µm)"
            ),
            developmental_stage="9pcw",
        ),
        "human_10_5pcw_cortex_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=170.0,
            standard_deviation=20.0,
            sample_size=16,
            reference_source=(
                "Cerebral Cortex 2008; 'A Molecular Neuroanatomical Study...' (doi:10.1093/cercor/bhm184; "
                "pmid:17965125; Fig 3A 10.5 PCW; scale bar 200 µm)"
            ),
            developmental_stage="10.5pcw",
        ),

        # Cortex — Apical nuclei linear density (per 100 µm)
        "human_5pcw_cortex_apical_nuclei_density": ExperimentalData(
            metric_name="apical_nuclei_per_100um",
            expected_value=3.0,
            standard_deviation=0.5,
            sample_size=1,
            reference_source=(
                "Cell Stem Cell 2023; Gastrulation & early brain development (doi:10.1016/j.stem.2023.04.016; "
                "pmid:37192616; Fig 1H PCW5; scale bar 20 µm)"
            ),
            developmental_stage="5pcw",
        ),

        # Hindbrain — VZ thickness (human)
        "human_6pcw_hindbrain_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=85.0,
            standard_deviation=8.0,
            sample_size=1,
            reference_source="Pax6/Pax7 in human embryos (myelencephalon)",
            developmental_stage="6pcw",
        ),

        # Spinal cord — lumen diameters and VZ thickness (human)
        "human_cs15_spinal_lumen_diameter": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=55.0,
            standard_deviation=5.0,
            sample_size=1,
            reference_source="eLife 2022; Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS15",
        ),
        "human_cs16_spinal_lumen_diameter": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=62.0,
            standard_deviation=6.0,
            sample_size=1,
            reference_source="eLife 2022; Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS16",
        ),
        "human_cs17_spinal_lumen_diameter": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=78.0,
            standard_deviation=8.0,
            sample_size=1,
            reference_source="eLife 2022; Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS17",
        ),
        "human_cs18_spinal_lumen_diameter": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=40.0,
            standard_deviation=4.0,
            sample_size=1,
            reference_source="eLife 2022; Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS18",
        ),
        "human_5pcw_spinal_vz_floorplate_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=35.0,
            standard_deviation=10.0,
            sample_size=9,
            reference_source="Ventral midline thickness at 5 weeks (CS13–15)",
            developmental_stage="5pcw",
        ),
        "human_10pcw_spinal_apical_nuclei_density": ExperimentalData(
            metric_name="apical_nuclei_per_100um",
            expected_value=18.0,
            standard_deviation=2.0,
            sample_size=3,
            reference_source="Connexins 37/43/45 in developing human spinal cord (doi:10.3390/ijms21249356)",
            developmental_stage="10pcw",
        ),
        "human_cs12_spinal_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=28.0,
            standard_deviation=3.0,
            sample_size=1,
            reference_source=(
                "Spinal neural tube formation (PMID:39636098; PMCID:PMC11620743; doi:10.7554/eLife.88584)"
            ),
            developmental_stage="CS12",
        ),
        "human_cs13_spinal_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=30.0,
            standard_deviation=4.0,
            sample_size=1,
            reference_source="Spinal neural tube formation (digitized; same study context)",
            developmental_stage="CS13",
        ),
        "human_cs14_spinal_lumen_diameter_a": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=22.0,
            standard_deviation=2.0,
            sample_size=1,
            reference_source="Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS14",
        ),
        "human_cs14_spinal_lumen_diameter_b1": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=12.0,
            standard_deviation=1.0,
            sample_size=1,
            reference_source="Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS14",
        ),
        "human_cs14_spinal_lumen_diameter_b2": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=14.0,
            standard_deviation=1.0,
            sample_size=1,
            reference_source="Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS14",
        ),
        "human_cs14_spinal_lumen_diameter_c": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=18.0,
            standard_deviation=2.0,
            sample_size=1,
            reference_source="Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS14",
        ),
        "human_cs15_spinal_lumen_diameter_e": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=20.0,
            standard_deviation=2.0,
            sample_size=1,
            reference_source="Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS15",
        ),
        "human_cs15_spinal_lumen_diameter_f1": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=16.0,
            standard_deviation=2.0,
            sample_size=1,
            reference_source="Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS15",
        ),
        "human_cs15_spinal_lumen_diameter_f2": ExperimentalData(
            metric_name="lumen_diameter_um",
            expected_value=13.0,
            standard_deviation=2.0,
            sample_size=1,
            reference_source="Spinal neural tube formation (PMCID:PMC11620743)",
            developmental_stage="CS15",
        ),

        # Cortex & Hindbrain — VZ thickness at CS16
        "human_cs16_cortex_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=38.0,
            standard_deviation=4.0,
            sample_size=1,
            reference_source=(
                "Microglia dynamics across lifespan (PMID:35977545; PMCID:PMC9616795; doi:10.1016/j.devcel.2022.07.015)"
            ),
            developmental_stage="CS16",
        ),
        "human_cs16_hindbrain_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=32.0,
            standard_deviation=3.0,
            sample_size=1,
            reference_source="Microglia dynamics across lifespan (PMID:35977545; PMCID:PMC9616795)",
            developmental_stage="CS16",
        ),

        # Spinal — VZ thickness CS14 (digitized)
        "human_cs14_spinal_vz_thickness": ExperimentalData(
            metric_name="vz_thickness_um",
            expected_value=67.0,
            standard_deviation=5.0,
            sample_size=1,
            reference_source="Pax6/Pax7 in human embryos (doi:10.4067/S0717-95022020000200259) digitized",
            developmental_stage="CS14",
        ),
    }

    return data


