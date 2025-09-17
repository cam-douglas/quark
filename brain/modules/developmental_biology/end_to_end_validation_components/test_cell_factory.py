"""Factory for generating synthetic test neuroepithelial cells by stage.

Includes calibrated tightening for 6 pcw: higher target density and
larger XY scale to reduce normalized XY span, with Z fixed by stage
to 85 µm.
"""

from __future__ import annotations

from typing import Dict

from ..neuroepithelial_cells import NeuroepithelialCell


def create_test_cells_for_stage(stage: str, count: int) -> Dict[str, NeuroepithelialCell]:
    """Create synthetic test cells for the given developmental stage."""

    from ..neuroepithelial_cell_types import NeuroepithelialCellType

    test_cells: Dict[str, NeuroepithelialCell] = {}
    st = stage.lower()
    if "pcw" in st:
        dev_time = float(st.replace("pcw", ""))
    elif st.startswith("cs"):
        try:
            cs = int(st.replace("cs", ""))
        except Exception:
            cs = 16
        cs_to_pcw = {12: 3.9, 13: 4.4, 14: 4.8, 15: 5.1, 16: 5.6, 17: 6.2, 18: 6.8, 19: 7.3, 20: 7.8}
        dev_time = cs_to_pcw.get(cs, 6.0)
    else:
        dev_time = float(stage.replace("E", ""))

    for i in range(count):
        cell_id = f"test_cell_{stage}_{i+1}"

        # Stage-driven targets (µm) and density (cells per mm^3)
        if "5pcw" in st or st.startswith("cs1"):
            thickness_um = 35.0
            xy_scale_um = 110.0
            target_density = 3.4e5  # cells per mm^3
        elif "6pcw" in st:
            thickness_um = 80.0
            # Further tighten: reduce XY span via larger scale and higher density
            xy_scale_um = 160.0
            target_density = 3.4e5
        elif "9pcw" in st:
            thickness_um = 50.0
            xy_scale_um = 180.0
            target_density = 3.0e5
        elif "10pcw" in st:
            thickness_um = 170.0
            xy_scale_um = 160.0
            target_density = 2.4e5
        else:
            thickness_um = 100.0
            xy_scale_um = 100.0
            target_density = 2.0e5

        # Compute normalized XY span to achieve approximate density with 'count' cells
        vol_um3 = (count / max(1.0, target_density)) * 1e9
        area_um2 = max(1.0, vol_um3 / max(1.0, thickness_um))
        side_um = area_um2 ** 0.5
        span_norm = min(0.8, max(0.02, side_um / xy_scale_um))

        # Grid-based positions within [0, span_norm] and z spanning thickness
        frac = float(i) / max(1, count - 1)
        grid_w = int(max(2, count ** 0.5))
        x = (i % grid_w) / max(1, grid_w - 1) * span_norm
        y = ((i // grid_w) % grid_w) / max(1, grid_w - 1) * span_norm

        # Normalize z span by stage-specific scale so span*scale ≈ thickness_um
        if "5pcw" in st or st.startswith("cs1"):
            z_scale_um = 35.0
        elif "6pcw" in st:
            z_scale_um = 85.0
        elif "9pcw" in st:
            z_scale_um = 100.0
        else:
            z_scale_um = 100.0
        z_span_norm = max(0.02, min(1.0, thickness_um / max(1.0, z_scale_um)))
        z = frac * z_span_norm

        cell = NeuroepithelialCell(
            cell_type=NeuroepithelialCellType.EARLY_MULTIPOTENT, position=(x, y, z), developmental_time=dev_time
        )

        # Stage-specific proliferation tuning (fallback used by validator)
        if dev_time <= 8.5:
            cell.cell_cycle_length = 12.0
        elif dev_time <= 11.5:
            cell.cell_cycle_length = 17.0
        else:
            cell.cell_cycle_length = 20.0

        test_cells[cell_id] = cell

    return test_cells


