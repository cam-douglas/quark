2025-09-17 – Grafana Dashboards (Project)
========================================

Overview
--------
Dashboards to monitor brainstem segmentation in production.

Files
-----
- [brainstem_segmentation.json](brainstem_segmentation.json) — Main dashboard with latency, run counts, overall Dice, and Dice drift. (2025-09-17) ✔ active

Import
------
1. Open Grafana → Dashboards → Import.
2. Paste JSON file content or upload `brainstem_segmentation.json`.
3. Ensure Prometheus datasource points to the environment scraping the Quark metrics endpoint.

Metrics used
------------
- `brainstem_segmentation_latency_seconds{}` (histogram)
- `brainstem_segmentation_runs_total`
- `brainstem_segmentation_success_total`
- `brainstem_segmentation_overall_dice`
- `brainstem_segmentation_dice_drift`

Note
----
Set env `BRAINSTEM_METRICS_PORT` (default 9109) to expose the metrics endpoint.

