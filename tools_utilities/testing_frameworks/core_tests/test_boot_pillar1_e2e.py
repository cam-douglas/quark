#!/usr/bin/env python3
"""
TEST: Pillar‑1 End‑to‑End Boot (Brain Scaffold v1)
Purpose: Boot minimal scaffold, run for N steps, collect telemetry, and visualize
Outputs: HTML line chart of key confidences and modulators
Seeds: 42
Dependencies: plotly
"""
from pathlib import Path
import plotly.graph_objects as go

from src.core.brain_launcher import Brain, load_connectome


def run_e2e(steps: int = 30, connectome_path: str = "src/config/connectome.yaml"):
	cfg = load_connectome(connectome_path)
	brain = Brain(cfg)
	telemetry = []
	for _ in range(steps):
		tel = brain.step()
		telemetry.append(tel)
	# Build visual
	t = [tel["t"] for tel in telemetry]
	mods_DA = [tel["mods"]["DA"] for tel in telemetry]
	mods_NE = [tel["mods"]["NE"] for tel in telemetry]
	pfc = [tel.get("pfc",{}).get("confidence",0.0) for tel in telemetry]
	wm = [tel.get("working_memory",{}).get("confidence",0.0) for tel in telemetry]
	bg = [tel.get("basal_ganglia",{}).get("confidence",0.0) for tel in telemetry]
	thal = [tel.get("thalamus",{}).get("confidence",0.0) for tel in telemetry]

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=t, y=pfc, name="PFC.conf"))
	fig.add_trace(go.Scatter(x=t, y=wm, name="WM.conf"))
	fig.add_trace(go.Scatter(x=t, y=bg, name="BG.conf"))
	fig.add_trace(go.Scatter(x=t, y=thal, name="Thal.conf"))
	fig.add_trace(go.Scatter(x=t, y=mods_DA, name="DA", yaxis="y2"))
	fig.add_trace(go.Scatter(x=t, y=mods_NE, name="NE", yaxis="y2"))
	fig.update_layout(
		title="Pillar‑1 E2E Boot — Confidences & Modulators",
		xaxis_title="t",
		yaxis=dict(title="confidence"),
		yaxis2=dict(title="modulators", overlaying="y", side="right")
	)
	out_dir = Path("tests/outputs"); out_dir.mkdir(parents=True, exist_ok=True)
	out = out_dir/"pillar1_e2e_boot.html"
	fig.write_html(str(out))
	print("✅ Pillar‑1 E2E boot completed")
	print(f"📊 Results saved to: {out}")
	return out

if __name__ == "__main__":
	run_e2e()
