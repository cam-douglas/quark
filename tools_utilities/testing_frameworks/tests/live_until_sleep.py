#!/usr/bin/env python3
"""
Live Brain Simulation (runs until sleep)
- Uses brain_launcher_v3 with stage N0 (includes sleep engine)
- Prints live terminal status with ASCII bars
- Exits automatically when state == 'sleep'

Usage:
  python tests/live_until_sleep.py --connectome src/config/connectome_v3.yaml --stage N0 --sleep_period 25 --sleep_length 6 --ticks_per_week 50
"""
import os, sys
import time
import argparse

# Ensure src on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.brain_launcher_v3 import Brain, load_connectome, Curriculum

def bar(val: float, width: int = 20, max_val: float = 1.0, char: str = '#') -> str:
	val = max(0.0, min(max_val, float(val)))
	filled = int((val / max_val) * width)
	return char * filled + '-' * (width - filled)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--connectome', default='src/config/connectome_v3.yaml')
	ap.add_argument('--stage', default='N0', choices=['F','N0','N1'])
	ap.add_argument('--sleep_period', type=int, default=None)
	ap.add_argument('--sleep_length', type=int, default=None)
	ap.add_argument('--ticks_per_week', type=int, default=50)
	ap.add_argument('--max_ticks', type=int, default=0, help='0=unbounded (until sleep)')
	ap.add_argument('--interval', type=float, default=0.05, help='seconds between frames')
	args = ap.parse_args()

	cfg = load_connectome(args.connectome)
	# Optionally override AA sleep params in config
	cfg.setdefault('architecture_agent', {})
	if args.sleep_period is not None:
		cfg['architecture_agent']['sleep_period'] = int(args.sleep_period)
	if args.sleep_length is not None:
		cfg['architecture_agent']['sleep_length'] = int(args.sleep_length)

	cur_cfg = cfg.get('curriculum', {})
	schedule = cur_cfg.get('schedule', [])
	tpw = int(args.ticks_per_week)
	curriculum = Curriculum(schedule, tpw) if schedule else None

	brain = Brain(cfg, stage=args.stage, curriculum=curriculum, log_csv=None, dot_every=0, dot_dir='graphs')

	# Live loop until first sleep
	os.system('')  # enable ANSI on some terminals
	count = 0
	print('\033[2J\033[H', end='')
	print(f'=== Live Brain Simulation (stage={args.stage}) — will stop on sleep ===')
	while True:
		tel = brain.step(tpw)
		mods = tel['mods']
		state = tel['state']
		mode = tel['mode']
		pfc = tel.get('pfc',{}).get('confidence',0.0)
		wm = tel.get('working_memory',{}).get('confidence',0.0)
		bg = tel.get('basal_ganglia',{}).get('confidence',0.0)
		thal = tel.get('thalamus',{}).get('confidence',0.0)
		att = tel.get('attention',{}).get('task_bias',0.0)

		# Clear screen and render
		print('\033[2J\033[H', end='')
		print(f"[t={tel['t']:04d}] state={state:<5} mode={mode:<13}  fat={tel['fatigue']:.2f}")
		print(f" DA [{bar(mods['DA'])}] {mods['DA']:.2f}   NE [{bar(mods['NE'])}] {mods['NE']:.2f}   ACh [{bar(mods['ACh'])}] {mods['ACh']:.2f}")
		print(f" PFC  [{bar(pfc)}] {pfc:.2f}   WM  [{bar(wm)}] {wm:.2f}   BG  [{bar(bg)}] {bg:.2f}   Thal [{bar(thal)}] {thal:.2f}")
		print(f" AttBias [{bar(att)}] {att:.2f}")
		print('\nCtrl+C to exit early.\n')

		count += 1
		if state == 'sleep':
			print('💤 Entered sleep — exiting live run.')
			break
		if args.max_ticks and count >= args.max_ticks:
			print('Reached max_ticks — exiting live run.')
			break
		time.sleep(max(0.0, args.interval))

if __name__ == '__main__':
	main()
