import io
import threading
import time
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from flask import Flask, Response, send_file, render_template_string
from brian2 import *  # noqa: F401,F403

try:
    # Module execution
    from .thalamus_model import create_thalamic_nucleus  # type: ignore
except Exception:
    # Script execution
    from thalamus_model import create_thalamic_nucleus

app = Flask(__name__)

_image_lock = threading.Lock()
_image_bytes: Optional[bytes] = None
_stop_event = threading.Event()
_running = False

HTML = """
<!doctype html>
<title>Thalamic Relay - Live Viewer</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #0b0f1a; color: #e6eefc; }
  header { padding: 12px 16px; background: #111729; border-bottom: 1px solid #22304a; }
  main { padding: 12px 16px; }
  img { width: 100%; max-width: 960px; border: 1px solid #22304a; border-radius: 6px; }
  .meta { color: #99a6c4; font-size: 14px; margin: 8px 0 16px; }
  button { background: #1e2a44; color: #e6eefc; border: 1px solid #2b3b5c; padding: 6px 10px; border-radius: 6px; cursor: pointer; }
  button:hover { background: #243456; }
</style>
<header>
  <strong>Thalamic Relay - Live Viewer</strong>
</header>
<main>
  <div class="meta">Live raster updates every ~100ms. Use the buttons to start/stop the simulation.</div>
  <p>
    <button onclick="fetch('/start')">Start</button>
    <button onclick="fetch('/stop')">Stop</button>
  </p>
  <img id="frame" src="/frame.png?t=0" />
</main>
<script>
const img = document.getElementById('frame');
let t = 0;
setInterval(() => { t++; img.src = '/frame.png?t=' + t; }, 250);
</script>
"""


def _render_raster(mon_in: SpikeMonitor, mon_th: SpikeMonitor, t_ms_window: float = 1000.0) -> bytes:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_in, ax_th = axes

    times_in = mon_in.t / ms
    idx_in = mon_in.i
    times_th = mon_th.t / ms
    idx_th = mon_th.i

    if len(times_in):
        t_max = float(times_in[-1])
    else:
        t_max = 0.0
    t_min = max(0.0, t_max - t_ms_window)

    ax_in.plot(times_in, idx_in, '.k', markersize=2)
    ax_th.plot(times_th, idx_th, '.r', markersize=2)

    ax_in.set_xlim(t_min, t_max)
    ax_th.set_xlim(t_min, t_max)
    ax_in.set_ylabel('Sensory i')
    ax_th.set_ylabel('Thalamus i')
    ax_th.set_xlabel('Time (ms)')
    ax_in.set_title('Sensory Input (live)')
    ax_th.set_title('Thalamic Nucleus (relayed, live)')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _sim_loop():
    global _image_bytes, _running
    defaultclock.dt = 0.1*ms

    # Build network
    th = create_thalamic_nucleus(n_neurons=100)
    th.thresh = -55*mV
    th.tau = 30*ms

    n_inputs = 100
    input_rate = 100*Hz
    inp = PoissonGroup(n_inputs, rates=input_rate)
    syn = Synapses(inp, th, on_pre='v_post += 12*mV')
    syn.connect(j='i')

    mon_in = SpikeMonitor(inp)
    mon_th = SpikeMonitor(th)

    step_ms = 20.0
    _running = True
    while not _stop_event.is_set():
        run(step_ms*ms)
        img = _render_raster(mon_in, mon_th, t_ms_window=1500.0)
        with _image_lock:
            _image_bytes = img
        time.sleep(0.05)
    _running = False


@app.get('/')
def index():
    return render_template_string(HTML)


@app.get('/frame.png')
def frame():
    with _image_lock:
        data = _image_bytes
    if not data:
        # Empty placeholder
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, 'Waiting for simulation…', ha='center', va='center')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return Response(data, mimetype='image/png')


@app.get('/start')
def start():
    if not _running:
        _stop_event.clear()
        threading.Thread(target=_sim_loop, daemon=True).start()
        return 'started', 200
    return 'already running', 200


@app.get('/stop')
def stop():
    if _running:
        _stop_event.set()
        return 'stopping', 200
    return 'not running', 200


if __name__ == '__main__':
    # Auto-start simulation
    threading.Thread(target=_sim_loop, daemon=True).start()
    app.run(host='127.0.0.1', port=8010, debug=False)

