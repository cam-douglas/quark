import io
import threading
import time
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response, send_file, render_template_string, request, stream_with_context
from brian2 import *  # noqa: F401,F403

import numpy as np  # new
import os
import json

try:
    from .thalamus.thalamus_model import create_thalamic_nucleus  # type: ignore
except Exception:
    from thalamus.thalamus_model import create_thalamic_nucleus

from testing.testing_frameworks.scientific_validation import ScientificValidator  # type: ignore

# MLflow logger (optional)
try:
    from tools_utilities.automation.experiment_logger import ExperimentLogger
except Exception:
    ExperimentLogger = None  # type: ignore

# Optional MLflow client for replay
try:
    from mlflow.tracking import MlflowClient
except Exception:
    MlflowClient = None  # type: ignore

app = Flask(__name__)

_image_lock = threading.Lock()
_ctx_lock = threading.Lock()
_image_bytes: Optional[bytes] = None
_stop_event = threading.Event()
_running = False
_mode = 'thalamus'  # 'thalamus', 'hippocampus', 'ca3r'
_thread: Optional[threading.Thread] = None
_ctx: dict = {}
_last_sweep_start_s: float = 0.0
_replay_paths: dict = {}

HTML = """
<!doctype html>
<title>QUARK Live Viewer</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #0b0f1a; color: #e6eefc; }
  header { padding: 12px 16px; background: #111729; border-bottom: 1px solid #22304a; display:flex; gap:12px; align-items:center; }
  a.tab { color:#99a6c4; text-decoration:none; padding:6px 10px; border:1px solid #2b3b5c; border-radius:6px; }
  a.tab.active { background:#1e2a44; color:#e6eefc; }
  .panel { margin-top:16px; padding:12px 16px; background:#111729; border:1px solid #22304a; border-radius:6px; max-width:600px; }
  .panel label { display:block; margin:6px 0; font-size:14px; }
  .panel input[type=range] { width:70%; }
  .panel span.val { margin-left:8px; color:#4cc9f0; }
  main { padding: 12px 16px; }
  img { width: 100%; max-width: 1080px; border: 1px solid #22304a; border-radius: 6px; }
  .meta { color: #99a6c4; font-size: 14px; margin: 8px 0 16px; }
  button { background: #1e2a44; color: #e6eefc; border: 1px solid #2b3b5c; padding: 6px 10px; border-radius: 6px; cursor: pointer; }
  button:hover { background: #243456; }
</style>
<header>
  <strong>QUARK Live Viewer</strong>
  <a class="tab {{ 'active' if mode=='thalamus' else '' }}" href="/set_mode?mode=thalamus">Thalamus</a>
  <a class="tab {{ 'active' if mode=='hippocampus' else '' }}" href="/set_mode?mode=hippocampus">Hippocampus</a>
  <a class="tab {{ 'active' if mode=='ca3r' else '' }}" href="/set_mode?mode=ca3r">Hippocampus+CA3R</a>
  <a class="tab" style="margin-left:auto" href="http://127.0.0.1:8050" target="_blank">📊 Runs</a>
</header>
<main>
  <div class="meta">Mode: <strong>{{ mode }}</strong> • Live updates every ~150ms. <span id="val"></span> <span id="sweep"></span></div>
  <p>
    <button onclick="fetch('/start')">Start</button>
    <button onclick="fetch('/stop')">Stop</button>
    <button onclick="fetch('/log')">Log Run</button>
    <button onclick="setGoal()">Set Goal Priority</button>
    <button onclick="fetch('/validate').then(r=>r.text().then(t=>alert(t)))">Run Validation</button>
    <button onclick="startSweep()">Start CA3 Sweep</button>
    <label style="margin-left:12px">Replay run_id: <input id="runid" size="36"> <button onclick="startReplay()">Replay</button></label>
  </p>
  <div class="panel">
    <strong>Live Controls</strong>
    <label>Input rate: <input type="range" id="rate" min="10" max="300" step="10" value="100" oninput="document.getElementById('rateVal').textContent=this.value;"> <span id="rateVal" class="val">100</span> Hz <button onclick="setRate()">Set</button></label>
    <label>Threshold: <input type="range" id="thresh" min="-70" max="-40" step="1" value="-55" oninput="document.getElementById('thrVal').textContent=this.value;"> <span id="thrVal" class="val">-55</span> mV <button onclick="setThresh()">Set</button></label>
    <label>A<sub>+</sub>: <input type="range" id="aplus" min="0" max="1" step="0.05" value="0.5" oninput="document.getElementById('aplusVal').textContent=this.value;"> <span id="aplusVal" class="val">0.5</span> <button onclick="setAplus()">Set</button></label>
    <label>A<sub>-</sub>: <input type="range" id="aminus" min="0" max="1" step="0.05" value="0.5" oninput="document.getElementById('aminusVal').textContent=this.value;"> <span id="aminusVal" class="val">0.5</span> <button onclick="setAminus()">Set</button></label>
    <label>Motivational bias: <input type="range" id="bias" min="0" max="1" step="0.05" value="0.8" oninput="document.getElementById('biasVal').textContent=this.value;"> <span id="biasVal" class="val">0.8</span> <button onclick="setBias()">Set</button></label>
  </div>
  <img id="frame" src="/frame.png?t=0" />
  {% if mode == 'hippocampus' %}
  <div class="meta">CA3→CA1 weights (live)</div>
  <img id="weights" src="/weights.png?t=0" />
  {% endif %}
  <div class="meta">Replay (if available)</div>
  <img id="replay_frame" src="/replay_frame.png?t=0" />
  <img id="replay_weights" src="/replay_weights.png?t=0" />
</main>
<script>
const img = document.getElementById('frame');
let t = 0;
setInterval(() => { t++; img.src = '/frame.png?t=' + t; }, 300);
const weights = document.getElementById('weights');
if (weights) {
  let tw = 0; setInterval(() => { tw++; weights.src = '/weights.png?t=' + tw; }, 600);
}
const valSpan = document.getElementById('val');
setInterval(()=>{ fetch('/status').then(r=>r.json()).then(j=>{ if(j.last_validation){ valSpan.innerText=`Brain-Score: ${j.last_validation.score.toFixed(3)}`;} });}, 2000);
const sweepSpan = document.getElementById('sweep');
setInterval(()=>{ fetch('/sweep_status').then(r=> r.ok ? r.json(): Promise.resolve(null)).then(j=>{ if(j){ sweepSpan.innerText = `• Sweep: ${j.state} ${j.completed||0}/${j.total||0}`; } });}, 3000);
function startSweep(){ fetch('/sweep_start'); }
function setRate(){ const v=document.getElementById('rate').value; fetch('/set_param?name=rate_hz&value='+v); }
function setThresh(){ const v=document.getElementById('thresh').value; fetch('/set_param?name=thresh_mV&value='+v); }
function setAplus(){ const v=document.getElementById('aplus').value; fetch('/set_param?name=stdp_A_plus&value='+v); }
function setAminus(){ const v=document.getElementById('aminus').value; fetch('/set_param?name=stdp_A_minus&value='+v); }
function setBias(){ const v=document.getElementById('bias').value; fetch('/goal?priority='+v); }
function setGoal() {
  const p = prompt('Enter task priority (0–1):', '0.8');
  if (p !== null) { fetch('/goal?priority=' + p); }
}
function startReplay(){ const v=document.getElementById('runid').value; if(!v) return; fetch('/replay_start?run_id='+encodeURIComponent(v)); }
// Optional SSE for low-latency status
try {
  const es = new EventSource('/events');
  es.onmessage = (e)=>{ try{ const j = JSON.parse(e.data); if(j.last_validation&&valSpan){ valSpan.innerText=`Brain-Score: ${(+j.last_validation.score).toFixed(3)}`;} if(j.sweep&&sweepSpan){ sweepSpan.innerText = `• Sweep: ${j.sweep.state} ${j.sweep.completed||0}/${j.sweep.total||0}`;} }catch(_){} };
} catch (_) {}
</script>
"""


def _render_raster(panels: Tuple[SpikeMonitor, SpikeMonitor], labels: Tuple[str, str], t_ms_window: float = 1500.0) -> bytes:
    mon_a, mon_b = panels
    label_a, label_b = labels
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax_a, ax_b = axes

    t_a = mon_a.t / ms
    i_a = mon_a.i
    t_b = mon_b.t / ms
    i_b = mon_b.i

    t_max = float(t_a[-1]) if len(t_a) else 0.0
    t_min = max(0.0, t_max - t_ms_window)

    ax_a.plot(t_a, i_a, '.k', markersize=2)
    ax_b.plot(t_b, i_b, '.r', markersize=2)

    ax_a.set_xlim(t_min, t_max)
    ax_b.set_xlim(t_min, t_max)
    ax_a.set_ylabel(label_a)
    ax_b.set_ylabel(label_b)
    ax_b.set_xlabel('Time (ms)')
    ax_a.set_title(f'{label_a} (live)')
    ax_b.set_title(f'{label_b} (live)')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_thalamus():
    defaultclock.dt = 0.1*ms
    th = create_thalamic_nucleus(n_neurons=100)
    th.thresh = -55*mV
    th.tau = 30*ms
    inp = PoissonGroup(100, rates=100*Hz)
    syn = Synapses(inp, th, on_pre='v_post += 12*mV')
    syn.connect(j='i')
    mon_in = SpikeMonitor(inp)
    mon_th = SpikeMonitor(th)
    net = Network(th, inp, syn, mon_in, mon_th)

    panels = (mon_in, mon_th)
    labels = ("Sensory i", "Thalamus i")
    base_rate = 100*Hz

    def step_fn(step_ms: float):
        # Apply rate scaling if set
        rate_scale = _ctx.get('rate_scale', 1.0)
        try:
            inp.rates = max(1*Hz, base_rate * rate_scale)
        except Exception:
            pass
        net.run(step_ms*ms)

    objs = [th, inp, syn, mon_in, mon_th, net]
    return panels, labels, step_fn, objs


def _build_hippocampus():
    defaultclock.dt = 0.1*ms
    # Parameters
    El0 = -65*mV
    tau0 = 20*ms
    thresh0 = -55*mV
    reset0 = -70*mV
    # Equations with parameters as state variables
    eqs = '''
    dv/dt = (El - v)/tau : volt (unless refractory)
    El : volt
    tau : second
    thresh : volt
    reset : volt
    '''
    # DG layer (granule cells)
    DG = NeuronGroup(100, eqs, threshold='v>thresh', reset='v=reset', refractory=5*ms, method='exact')
    DG.v = El0; DG.El = El0; DG.tau = tau0; DG.thresh = thresh0; DG.reset = reset0

    CA3 = NeuronGroup(100, eqs, threshold='v>thresh', reset='v=reset', refractory=5*ms, method='exact')
    CA1 = NeuronGroup(100, eqs, threshold='v>thresh', reset='v=reset', refractory=5*ms, method='exact')
    # Initialize membrane params
    CA3.v = El0; CA3.El = El0; CA3.tau = tau0; CA3.thresh = thresh0; CA3.reset = reset0
    # Make CA1 slightly more excitable
    CA1.v = El0; CA1.El = El0; CA1.tau = tau0; CA1.thresh = -60*mV; CA1.reset = reset0

    # External input to DG
    inp = PoissonGroup(100, rates=80*Hz)
    syn_in = Synapses(inp, DG, on_pre='v_post += 10*mV')
    syn_in.connect(j='i')

    # DG -> CA3 relay
    syn_dg_ca3 = Synapses(DG, CA3, on_pre='v_post += 10*mV')
    syn_dg_ca3.connect(j='i')

    # STDP CA3->CA1
    stdp_model = '''
    w : volt
    dapre/dt = -apre/taupre : 1 (event-driven)
    dapost/dt = -apost/taupost : 1 (event-driven)
    taupre : second (shared)
    taupost : second (shared)
    A_plus : 1 (shared)
    A_minus : 1 (shared)
    w_max : volt (shared)
    '''
    on_pre = 'v_post += w; apre += 1; w = clip(w + A_plus*apost*mV, 0*mV, w_max)'
    on_post = 'apost += 1; w = clip(w - A_minus*apre*mV, 0*mV, w_max)'
    syn_c3131 = Synapses(CA3, CA1, model=stdp_model, on_pre=on_pre, on_post=on_post, method='euler')
    syn_c3131.connect(j='i')
    syn_c3131.w = 10*mV
    syn_c3131.taupre = 20*ms
    syn_c3131.taupost = 20*ms
    syn_c3131.A_plus = 0.5
    syn_c3131.A_minus = 0.5
    syn_c3131.w_max = 20*mV

    mon_ca3 = SpikeMonitor(CA3)
    mon_ca1 = SpikeMonitor(CA1)
    net = Network(DG, CA3, CA1, inp, syn_in, syn_dg_ca3, syn_c3131, mon_ca3, mon_ca1)

    panels = (mon_ca3, mon_ca1)
    labels = ("CA3 i", "CA1 i")
    base_rate = 80*Hz

    def step_fn(step_ms: float):
        # Dynamic rate scaling (PFC bias)
        rate_scale = _ctx.get('rate_scale', 1.0)
        try:
            inp.rates = max(1*Hz, base_rate * rate_scale)
        except Exception:
            pass
        net.run(step_ms*ms)

    objs = [DG, CA3, CA1, inp, syn_in, syn_dg_ca3, syn_c3131, mon_ca3, mon_ca1, net]
    return panels, labels, step_fn, objs


def _sim_loop():
    global _image_bytes, _running, _ctx
    step_ms = 25.0

    if _mode == 'thalamus':
        panels, labels, step_fn, objs = _build_thalamus()
    else:
        if _mode == 'hippocampus':
            panels, labels, step_fn, objs = _build_hippocampus()
        else:
            from hippocampus.ca3_recurrent import build_ca3_recurrent  # type: ignore
            panels, labels, step_fn, objs = build_ca3_recurrent()

    _ctx = {'panels': panels, 'labels': labels, 'step_fn': step_fn, 'objs': objs}
    # Ensure validator is available for autonomous checks
    try:
        _ctx['scientific_validator'] = ScientificValidator()
    except Exception:
        # If validator cannot be constructed, proceed without automatic validation
        pass

    _running = True
    last_val = time.time()
    try:
        while not _stop_event.is_set():
            step_fn(step_ms)
            img = _render_raster(panels, labels)
            with _image_lock:
                _image_bytes = img
            # autonomous validation every 5s
            if 'scientific_validator' in _ctx and time.time() - last_val > 5.0:
                try:
                    mon_a, mon_b = panels
                    agi_data = {
                        'activity': np.random.rand(100),  # placeholder
                    }
                    res = _ctx['scientific_validator'].run_validation(agi_data, 'brain_score')
                    _ctx['last_validation'] = res
                except Exception:
                    pass
                last_val = time.time()
            time.sleep(0.12)
    finally:
        _running = False


@app.get('/')
def index():
    return render_template_string(HTML, mode=_mode)


@app.get('/set_mode')
def set_mode():
    global _mode, _thread
    mode = request.args.get('mode', 'thalamus')
    if mode not in ('thalamus', 'hippocampus', 'ca3r'):
        mode = 'thalamus'
    _mode = mode
    # restart sim if running
    if _running or (_thread and _thread.is_alive()):
        _stop_event.set()
        if _thread:
            _thread.join(timeout=2.0)
        _stop_event.clear()
        _thread = threading.Thread(target=_sim_loop, daemon=True)
        _thread.start()
    return ('', 302, {'Location': '/'})


@app.get('/set_param')
def set_param():
    """Set runtime parameters.
    Examples:
      /set_param?name=rate_hz&value=120
      /set_param?name=thresh_mV&value=-52
      /set_param?name=stdp_A_plus&value=0.6
      /set_param?name=stdp_A_minus&value=0.4
    """
    name = request.args.get('name')
    value = request.args.get('value')
    if name is None or value is None:
        return 'missing name/value', 400
    try:
        with _ctx_lock:
            if name == 'rate_hz':
                scale = float(value) / (100.0 if _mode == 'thalamus' else 80.0)
                _ctx['rate_scale'] = max(0.1, min(5.0, scale))
                return 'ok', 200
            if name == 'thresh_mV':
                thr = float(value)*mV
                for o in _ctx.get('objs', []):
                    if isinstance(o, NeuronGroup) and hasattr(o, 'thresh'):
                        o.thresh = thr
                return 'ok', 200
            if name == 'stdp_A_plus' or name == 'stdp_A_minus':
                a = float(value)
                for o in _ctx.get('objs', []):
                    if isinstance(o, Synapses) and hasattr(o, 'A_plus'):
                        if name == 'stdp_A_plus':
                            o.A_plus = a
                        else:
                            o.A_minus = a
                return 'ok', 200
        return 'no-op', 200
    except Exception as e:
        return f'error: {e}', 500


@app.get('/seed')
def set_seed():
    s = request.args.get('value')
    if s is None:
        return 'missing value', 400
    try:
        sval = int(s)
        import numpy as _np
        _np.random.seed(sval)
        seed(sval)
        return 'ok', 200
    except Exception as e:
        return f'error: {e}', 500


@app.get('/pfc_bias')
def pfc_bias():
    v = request.args.get('value')
    if v is None:
        return 'missing value', 400
    try:
        scale = float(v)
        with _ctx_lock:
            _ctx['rate_scale'] = max(0.1, min(5.0, scale))
        return 'ok', 200
    except Exception as e:
        return f'error: {e}', 500


@app.get('/goal')
def goal():
    """Set motivational bias from a task priority (0–1).

    Example: /goal?priority=0.8  → scales input rates proportionally.
    Internally this sets the same _ctx['rate_scale'] used by /pfc_bias.
    """
    p = request.args.get('priority')
    if p is None:
        return 'missing priority', 400
    try:
        pr = max(0.0, min(1.0, float(p)))
        # Map priority 0→1 to scale 0.5→2.0 (tuned empirically)
        scale = 0.5 + 1.5 * pr
        with _ctx_lock:
            _ctx['rate_scale'] = scale
        return {'scale': scale}, 200
    except Exception as e:
        return f'error: {e}', 500


@app.get('/frame.png')
def frame():
    with _image_lock:
        data = _image_bytes
    if not data:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, f'Waiting for simulation… (mode={_mode})', ha='center', va='center')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return Response(data, mimetype='image/png')


@app.get('/weights.png')
def weights_png():
    if _mode != 'hippocampus' or not _ctx:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, 'Weights available in Hippocampus mode', ha='center', va='center')
        ax.axis('off')
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return send_file(buf, mimetype='image/png')
    # Extract synapse weights
    try:
        syn = None
        for o in _ctx.get('objs', []):
            if isinstance(o, Synapses) and hasattr(o, 'w'):
                syn = o
                break
        if syn is None:
            raise RuntimeError('No STDP synapse found')
        w = syn.w[:] / mV
        fig, ax = plt.subplots(figsize=(10, 3))
        if len(w):
            ax.hist(w, bins=20, color='#4cc9f0', edgecolor='#22304a')
            ax.set_title('CA3→CA1 weights (mV)')
            ax.set_xlabel('w (mV)'); ax.set_ylabel('count')
        else:
            ax.text(0.5, 0.5, 'No weights', ha='center', va='center')
            ax.axis('off')
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center'); ax.axis('off')
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return send_file(buf, mimetype='image/png')


@app.get('/log')
def log_run():
    if ExperimentLogger is None:
        return 'ExperimentLogger not available', 501
    try:
        # Collect metrics
        mode = _mode
        panels = _ctx.get('panels') if _ctx else None
        labels = _ctx.get('labels') if _ctx else None
        # weights
        w_mv = []
        syn = None
        if mode == 'hippocampus':
            for o in _ctx.get('objs', []):
                if isinstance(o, Synapses) and hasattr(o, 'w'):
                    syn = o; break
            if syn is not None:
                w_mv = list((syn.w[:] / mV))
        # params
        params = { 'mode': mode }
        metrics = {
            'num_weighted_synapses': int(len(w_mv)),
            'mean_w_mV': float(np.mean(w_mv)) if len(w_mv) else 0.0,
            'std_w_mV': float(np.std(w_mv)) if len(w_mv) else 0.0,
        }
        # artifact: save current frames
        img_buf = io.BytesIO()
        if panels and labels:
            img = _render_raster(panels, labels)
            img_buf.write(img); img_buf.seek(0)
        # save + log
        logger = ExperimentLogger()
        with logger.start_run(run_name=f"live_{mode}", experiment_name="QuarkBrainSimulation"):
            logger.log_params(params)
            logger.log_metrics(metrics)
            # Save weights hist as file
            if len(w_mv):
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(w_mv, bins=20, color='#4cc9f0', edgecolor='#22304a')
                ax.set_title('CA3→CA1 weights (mV)')
                ax.set_xlabel('w (mV)'); ax.set_ylabel('count')
                tmp = 'weights_hist.png'; fig.savefig(tmp); plt.close(fig)
                logger.log_artifact(tmp)
                os.remove(tmp)
            # Save raster snapshot
            tmp2 = 'raster_snapshot.png'; open(tmp2, 'wb').write(img_buf.getvalue())
            logger.log_artifact(tmp2); os.remove(tmp2)
        return 'logged', 200
    except Exception as e:
        return f'log error: {e}', 500


@app.get('/validate')
def validate():
    if 'scientific_validator' not in _ctx:
        try:
            _ctx['scientific_validator'] = ScientificValidator()
        except Exception:
            return 'ScientificValidator not available', 501
    validator = _ctx['scientific_validator']
    # Mock: extract simple activity vector from monitors
    try:
        panels = _ctx.get('panels')
        if not panels:
            return 'no data', 400
        mon_a, mon_b = panels
        agi_data = {
            'spikes_a': list(zip(mon_a.t/ms, mon_a.i)),
            'spikes_b': list(zip(mon_b.t/ms, mon_b.i)),
        }
        res = validator.run_validation(agi_data, 'brain_score')
        return res, 200
    except Exception as e:
        return f'validate error: {e}', 500


@app.get('/start')
def start():
    global _thread
    if not _running and not (_thread and _thread.is_alive()):
        _stop_event.clear()
        _thread = threading.Thread(target=_sim_loop, daemon=True)
        _thread.start()
        return 'started', 200
    # Attempt to recover if thread died but flag not set
    if _thread and not _thread.is_alive():
        _stop_event.clear()
        _thread = threading.Thread(target=_sim_loop, daemon=True)
        _thread.start()
        return 'restarted', 200
    return 'already running', 200


@app.get('/stop')
def stop():
    global _thread
    if _running or (_thread and _thread.is_alive()):
        _stop_event.set()
        if _thread:
            _thread.join(timeout=3.0)
        return 'stopped', 200
    return 'not running', 200


@app.get('/status')
def status():
    state = 'running' if _running else 'stopped'
    payload = {'mode': _mode, 'state': state}
    if 'last_validation' in _ctx:
        payload['last_validation'] = _ctx['last_validation']
    return payload, 200


@app.get('/healthz')
def healthz():
    return 'ok', 200


@app.get('/sweep_start')
def sweep_start():
    from .hippocampus.ca3_sweep import run_sweep  # type: ignore
    def _bg():
        try:
            run_sweep(status_path='logs/ca3_sweep_status.json', parallel=1, sim_ms=2000.0)
            # On completion, aggregate results if any and persist CSV
            try:
                import json as _json, csv as _csv
                st = _json.loads(open('logs/ca3_sweep_status.json').read())
                hist = st.get('history', [])
                if hist:
                    os.makedirs('logs', exist_ok=True)
                    with open('logs/ca3_sweep_results.csv', 'w', newline='') as f:
                        w = _csv.DictWriter(f, fieldnames=sorted(hist[0].keys()))
                        w.writeheader()
                        for row in hist:
                            w.writerow(row)
            except Exception:
                pass
        except Exception:
            pass
    global _last_sweep_start_s
    now = time.time()
    if now - _last_sweep_start_s < 10.0:
        return 'cooldown', 429
    _last_sweep_start_s = now
    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return 'started', 200


@app.get('/sweep_status')
def sweep_status():
    try:
        import json as _json
        if not os.path.exists('logs/ca3_sweep_status.json'):
            return {'state': 'idle', 'total': 0, 'completed': 0}, 200
        return _json.loads(open('logs/ca3_sweep_status.json').read()), 200
    except Exception:
        return {'state': 'error'}, 200


@app.get('/replay_start')
def replay_start():
    if MlflowClient is None:
        return 'mlflow client not available', 501
    run_id = request.args.get('run_id')
    if not run_id:
        return 'missing run_id', 400
    try:
        client = MlflowClient()
        dst = os.path.join('logs', 'replay', run_id)
        os.makedirs(dst, exist_ok=True)
        # Candidates based on our logger usage
        candidates = ['raster_snapshot.png', 'weights_hist.png', 'ca3_weights_hist.png']
        found = {}
        for name in candidates:
            try:
                p = client.download_artifacts(run_id, name, dst)
                if os.path.exists(p):
                    if 'raster' not in found and 'raster' in name:
                        found['raster'] = p
                    if 'weights' in name:
                        found['weights'] = p
            except Exception:
                continue
        global _replay_paths
        _replay_paths = found
        return {'found': list(found.keys())}, 200
    except Exception as e:
        return f'error: {e}', 500


@app.get('/replay_frame.png')
def replay_frame_png():
    path = _replay_paths.get('raster')
    if not path or not os.path.exists(path):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, 'No replay raster available', ha='center', va='center'); ax.axis('off')
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return send_file(path, mimetype='image/png')


@app.get('/replay_weights.png')
def replay_weights_png():
    path = _replay_paths.get('weights')
    if not path or not os.path.exists(path):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, 'No replay weights available', ha='center', va='center'); ax.axis('off')
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return send_file(path, mimetype='image/png')


@app.get('/events')
def events():
    def gen():
        while True:
            try:
                payload = {'last_validation': _ctx.get('last_validation')}
                try:
                    import json as _json
                    st = _json.loads(open('logs/ca3_sweep_status.json').read()) if os.path.exists('logs/ca3_sweep_status.json') else {'state': 'idle'}
                except Exception:
                    st = {'state': 'idle'}
                payload['sweep'] = st
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(2.0)
            except GeneratorExit:
                break
            except Exception:
                time.sleep(2.0)
    return Response(stream_with_context(gen()), mimetype='text/event-stream')


if __name__ == '__main__':
    _thread = threading.Thread(target=_sim_loop, daemon=True)
    _thread.start()
    app.run(host='127.0.0.1', port=8011, debug=False)
