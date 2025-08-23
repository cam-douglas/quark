#!/usr/bin/env python3
"""
Pillar‑1 Only Runner
Purpose: Execute only foundation-layer tests and emit a brief metrics summary
"""
import sys
import time
from pathlib import Path
import subprocess

PILLAR1_TESTS = [
	"src/core/tests/test_developmental_timeline.py",
	"src/core/tests/test_neural_components.py",
	"src/core/tests/test_brain_launcher.py",
	"src/core/tests/test_sleep_consolidation_engine.py",
	"src/core/tests/test_multi_scale_integration.py",
	"src/core/tests/test_capacity_progression.py",
	"src/core/tests/test_rules_loader.py",
	"src/core/tests/test_boot_pillar1_e2e.py",
]

def run():
	start = time.time()
	ok = 0
	for p in PILLAR1_TESTS:
		print(f"🧪 {p}")
		res = subprocess.run([sys.executable, p], capture_output=True, text=True)
		if res.returncode == 0:
			ok += 1
			print("  ✅ PASS")
		else:
			print("  ❌ FAIL")
			print(res.stdout)
			print(res.stderr)
	dur = time.time() - start
	Path("data/metrics").mkdir(parents=True, exist_ok=True)
	with open("data/metrics/pillar1_summary.txt", "w", encoding="utf-8") as f:
		f.write(f"tests={len(PILLAR1_TESTS)} pass={ok} fail={len(PILLAR1_TESTS)-ok} time={dur:.2f}s\n")
	print(f"\n📊 Pillar‑1 summary: {ok}/{len(PILLAR1_TESTS)} passed in {dur:.2f}s")
	print("📁 metrics: data/metrics/pillar1_summary.txt")

if __name__ == "__main__":
	run()
