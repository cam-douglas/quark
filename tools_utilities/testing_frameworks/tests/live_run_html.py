#!/usr/bin/env python3
"""
Live HTML Brain Simulation (fixed duration)
- Runs brain_launcher_v3 for a wall-clock duration and saves Plotly HTML
- Captures modulators, confidences, state/mode over time

Usage:
  python tests/live_run_html.py --connectome src/config/connectome_v3.yaml --stage N0 --duration_sec 30 --interval 0.001
"""
import os, sys
import time
import argparse
import signal
from pathlib import Path

# Ensure src on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.brain_launcher_v3 import Brain, load_connectome, Curriculum
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def timeout_handler(signum, frame):
    raise TimeoutError("Simulation timed out")

def run_html(connectome: str, stage: str, duration_sec: float, interval: float, ticks_per_week: int = 50, sleep_period: int | None = None, sleep_length: int | None = None) -> Path:
    print(f"🚀 Starting brain simulation for {duration_sec}s (stage: {stage}, interval: {interval*1000:.1f}ms)...")
    
    cfg = load_connectome(connectome)
    cfg.setdefault('architecture_agent', {})
    if sleep_period is not None:
        cfg['architecture_agent']['sleep_period'] = int(sleep_period)
    if sleep_length is not None:
        cfg['architecture_agent']['sleep_length'] = int(sleep_length)
    cur_cfg = cfg.get('curriculum', {})
    schedule = cur_cfg.get('schedule', [])
    curriculum = Curriculum(schedule, ticks_per_week) if schedule else None
    brain = Brain(cfg, stage=stage, curriculum=curriculum, log_csv=None, dot_every=0, dot_dir='graphs')

    t_vals = []
    state_vals = []
    mode_vals = []
    da_vals = []
    ne_vals = []
    ach_vals = []
    pfc_vals = []
    wm_vals = []
    bg_vals = []
    thal_vals = []
    att_vals = []

    start = time.time()
    tick_count = 0
    last_progress_time = start
    
    # Set up timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(duration_sec + 5))  # Add 5 seconds buffer
    
    try:
        while True:
            current_time = time.time() - start
            if current_time >= duration_sec:
                print(f"⏱️ Duration reached ({duration_sec}s)")
                break
                
            # Run brain step
            tel = brain.step(ticks_per_week)
            tick_count += 1
            
            # Collect data
            mods = tel['mods']
            t_vals.append(tel['t'])
            state_vals.append(tel['state'])
            mode_vals.append(tel['mode'])
            da_vals.append(mods['DA'])
            ne_vals.append(mods['NE'])
            ach_vals.append(mods['ACh'])
            pfc_vals.append(tel.get('pfc',{}).get('confidence',0.0))
            wm_vals.append(tel.get('working_memory',{}).get('confidence',0.0))
            bg_vals.append(tel.get('basal_ganglia',{}).get('confidence',0.0))
            thal_vals.append(tel.get('thalamus',{}).get('confidence',0.0))
            att_vals.append(tel.get('attention',{}).get('task_bias',0.0))
            
            # Progress output every 2 seconds or every 100 ticks (whichever comes first)
            if (current_time - last_progress_time >= 2.0) or (tick_count % 100 == 0):
                print(f"📊 Tick {tick_count}, t={tel['t']}, state={tel['state']}, elapsed={current_time:.1f}s, DA={mods['DA']:.2f}, PFC={tel.get('pfc',{}).get('confidence',0.0):.2f}")
                last_progress_time = current_time
            
            # Check if we should sleep
            if interval > 0:
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except TimeoutError:
        print(f"\n⏰ Timeout after {duration_sec}s")
    except Exception as e:
        print(f"\n💥 Error during simulation: {e}")
    finally:
        signal.alarm(0)  # Cancel timeout
        print(f"✅ Simulation completed: {tick_count} ticks, {len(t_vals)} data points")

    # Guard: ensure we have at least one point
    if not t_vals:
        print("⚠️ No data collected, creating dummy data")
        t_vals = [0]
        da_vals = [0.0]; ne_vals = [0.0]; ach_vals = [0.0]
        pfc_vals = [0.0]; wm_vals = [0.0]; bg_vals = [0.0]; thal_vals = [0.0]
        att_vals = [0.0]
        state_vals = ['wake']
        mode_vals = ['internal']

    print("📊 Creating HTML visualization...")
    
    # Build Plotly HTML
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("Modulators (DA/NE/ACh)", "Module Confidences (PFC/WM/BG/Thal)", "Attention Bias + State/Mode"))

    fig.add_trace(go.Scatter(x=t_vals, y=da_vals, name='DA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_vals, y=ne_vals, name='NE', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_vals, y=ach_vals, name='ACh', line=dict(color='green')), row=1, col=1)

    fig.add_trace(go.Scatter(x=t_vals, y=pfc_vals, name='PFC.conf', line=dict(color='#8e44ad')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_vals, y=wm_vals, name='WM.conf', line=dict(color='#2ecc71')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_vals, y=bg_vals, name='BG.conf', line=dict(color='#e74c3c')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_vals, y=thal_vals, name='Thal.conf', line=dict(color='#3498db')), row=2, col=1)

    fig.add_trace(go.Scatter(x=t_vals, y=att_vals, name='AttBias', line=dict(color='#34495e')), row=3, col=1)

    # Shade sleep period if present
    sleep_indices = [i for i,s in enumerate(state_vals) if s == 'sleep']
    if sleep_indices:
        first_sleep_t = t_vals[sleep_indices[0]]
        x1 = max(t_vals) if t_vals else first_sleep_t
        fig.add_vrect(x0=first_sleep_t-0.5, x1=x1+0.5, fillcolor="#f7d794", opacity=0.25, line_width=0, row=1, col=1)
        fig.add_vrect(x0=first_sleep_t-0.5, x1=x1+0.5, fillcolor="#f7d794", opacity=0.25, line_width=0, row=2, col=1)
        fig.add_vrect(x0=first_sleep_t-0.5, x1=x1+0.5, fillcolor="#f7d794", opacity=0.25, line_width=0, row=3, col=1)

    fig.update_layout(title=f"Live Brain Simulation (stage={stage}) — {tick_count} ticks, {len(t_vals)} data points, {interval*1000:.1f}ms intervals",
        height=900, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0))
    fig.update_xaxes(title_text='t (ticks)', row=3, col=1)
    fig.update_yaxes(title_text='value', row=1, col=1)
    fig.update_yaxes(title_text='confidence', row=2, col=1)
    fig.update_yaxes(title_text='att_bias', row=3, col=1)

    out_dir = Path('tests/outputs'); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'live_run.html'
    fig.write_html(str(out_path))
    print(f"📁 HTML saved to: {out_path}")
    return out_path

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--connectome', default='src/config/connectome_v3.yaml')
    ap.add_argument('--stage', default='N0', choices=['F','N0','N1'])
    ap.add_argument('--duration_sec', type=float, default=30.0)
    ap.add_argument('--interval', type=float, default=0.001)  # Default to 1ms intervals
    ap.add_argument('--ticks_per_week', type=int, default=50)
    ap.add_argument('--sleep_period', type=int, default=None)
    ap.add_argument('--sleep_length', type=int, default=None)
    args = ap.parse_args()
    
    path = run_html(args.connectome, args.stage, args.duration_sec, args.interval, args.ticks_per_week, args.sleep_period, args.sleep_length)
    print(f"✅ Live HTML saved: {path}")
