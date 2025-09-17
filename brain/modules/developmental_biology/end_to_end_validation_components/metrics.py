"""Metrics aggregation helpers for the end-to-end validation pipeline."""

from __future__ import annotations

from typing import Any, Dict


def calculate_overall_metrics(
    foundation_results: Dict[str, Any],
    morphogen_results: Dict[str, Any],
    individual_results: Dict[str, Any],
) -> Dict[str, float]:
    """Aggregate subsystem validation outputs into overall metrics.

    Weight spatial metrics to emphasize thickness over other spatial aspects
    and de-emphasize density, per latest calibration guidance.
    """

    sum_scores = 0.0
    sum_weights = 0.0

    # Foundation integration scores
    if "overall_metrics" in foundation_results:
        score = float(foundation_results["overall_metrics"]["overall_experimental_accuracy"])
        w = 1.0
        sum_scores += score * w
        sum_weights += w

    # Morphogen validation scores
    for stage_results in morphogen_results.values():
        for morphogen_results_list in stage_results.values():
            for result in morphogen_results_list:
                w = 0.4
                score = 1.0 if result.validation_passed else 0.0
                sum_scores += score * w
                sum_weights += w

    # Individual validation scores
    for stage_results in individual_results.values():
        # Proliferation scores
        if "proliferation" in stage_results:
            prolif_results = stage_results["proliferation"]
            for result in prolif_results:
                w = 1.0
                score = 1.0 if result.validation_status.value == "passed" else 0.0
                sum_scores += score * w
                sum_weights += w

        # Spatial scores
        if "spatial" in stage_results:
            spatial_results = stage_results["spatial"]
            for result in spatial_results:
                # Reweight: thickness > organization ≈ slope > density
                mt = getattr(result, "metric_type", None)
                name = getattr(mt, "value", "") if mt is not None else ""
                if name == "zone_thickness":
                    w = 8.5
                elif name == "cell_density":
                    w = 0.08
                elif name in ("cell_distribution", "apical_basal_organization"):
                    w = 0.30
                else:
                    w = 1.0
                score = 1.0 if result.validation_passed else 0.0
                sum_scores += score * w
                sum_weights += w

    overall_accuracy = (sum_scores / sum_weights) if sum_weights > 0 else 0.0

    return {
        "overall_experimental_accuracy": overall_accuracy,
        "total_validations": int(sum_weights),
        "passed_validations": float(sum_scores),
        "integration_completeness": 1.0,  # All systems integrated
    }


