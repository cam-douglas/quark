#!/usr/bin/env python3
"""
TEST: rules_loader (Pillar 1)
Purpose: Validate connectome by stage and runtime instrumentation enforcement
Outputs: HTML summary of validations
Seeds: 42
Dependencies: plotly
"""
import json
from pathlib import Path
import plotly.graph_objects as go

from src.core.rules_loader import load_rules, validate_connectome, ValidationError, instrument_agent

class DummyAgent:
	def step(self, inbox, ctx):
		# Return minimal telemetry missing required keys to test enforcement
		return [], {}


def run_tests():
	results = []
	# Minimal valid connectome for F
	base_cfg = {
		"architecture_agent": {},
		"modules": {
			"pfc": {}, "basal_ganglia": {}, "thalamus": {},
			"working_memory": {"slots": 3}, "hippocampus": {}, "dmn": {}
		},
		"attention": {}
	}

	rules = load_rules([])  # use defaults
	for stage in ("F", "N0", "N1"):
		cfg = json.loads(json.dumps(base_cfg))
		# Add required stage modules progressively
		if stage in ("N0", "N1"):
			cfg["modules"]["salience"] = {}
			cfg["modules"]["sleeper"] = {}
		if stage == "N1":
			cfg["modules"]["cerebellum"] = {}
		ok, err = True, ""
		try:
			validate_connectome(cfg, stage, rules)
		except Exception as e:
			ok, err = False, str(e)
		results.append({"stage": stage, "valid": ok, "error": err})

	# Negative case: missing module at N0
	bad_cfg = json.loads(json.dumps(base_cfg))
	bad_ok, bad_err = True, ""
	try:
		validate_connectome(bad_cfg, "N0", rules)
	except ValidationError as e:
		bad_ok, bad_err = False, str(e)
	results.append({"stage": "N0_missing", "valid": bad_ok, "error": bad_err})

	# Instrumentation: ensure missing telemetry keys are injected
	agent = instrument_agent("pfc", "PFC", DummyAgent(), stage="F", rules=rules)
	out, tel = agent.step([], {"global": {}, "modulators": {}})
	inject_ok = ("confidence" in tel and "demand" in tel)
	results.append({"stage": "instrumentation", "valid": inject_ok, "error": "" if inject_ok else "telemetry keys not injected"})

	# Visual summary
	labels = [r["stage"] for r in results]
	vals = [1 if r["valid"] else 0 for r in results]
	fig = go.Figure(data=[go.Bar(x=labels, y=vals, marker_color=["green" if v else "red" for v in vals])])
	fig.update_layout(title="rules_loader validations", yaxis=dict(range=[0,1], tickvals=[0,1], ticktext=["FAIL","OK"]))
	out_dir = Path("tests/outputs"); out_dir.mkdir(parents=True, exist_ok=True)
	fig.write_html(str(out_dir/"rules_loader_validation.html"))
	print("✅ rules_loader tests completed")
	print(f"📊 Results saved to: {out_dir/ 'rules_loader_validation.html'}")
	return results

if __name__ == "__main__":
	run_tests()
