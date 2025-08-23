#!/usr/bin/env python3
"""
brain_launcher_v3.py — cognitive brain scaffold
Adds:
- CSV metrics logger (via --log_csv)
- Graphviz DOT export every N ticks (via --dot_every and --dot_dir)
- Curriculum scheduler (ticks→weeks) that adjusts WM slots and MoE k per schedule in YAML

Usage:
  pip install pyyaml
  python brain_launcher_v3.py --connectome connectome_v3.yaml --steps 200 --stage F \
      --log_csv metrics.csv --dot_every 25 --dot_dir ./graphs
"""

import argparse
import random
import os
from 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclasses import 📊_DATA_KNOWLEDGE.01_DATA_REPOSITORYclass, field
from typing import Dict, Any, List, Tuple
import yaml

# Import neural components for Pillar 1
from .neural_components import SpikingNeuron, HebbianSynapse, STDP, NeuralPopulation, calculate_synchrony, calculate_oscillation_power

# ---------------------------
# Message types
# ---------------------------
@dataclass
class Message:
    kind: str              # Observation | Plan | Command | Reward | Modulation | Telemetry | Replay
    src: str
    dst: str
    priority: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

def msg(kind, src, dst, **payload):
    return Message(kind=kind, src=src, dst=dst, payload=payload)

# ---------------------------
# Base Module
# ---------------------------
class Module:
    def __init__(self, name: str, spec: Dict[str, Any]):
        self.name = name
        self.spec = spec
        self.state: Dict[str, Any] = {}

    def step(self, inbox: List[Message], ctx: Dict[str, Any]) -> Tuple[List[Message], Dict[str, Any]]:
        raise NotImplementedError

# ---------------------------
# Utils
# ---------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ---------------------------
# Module Implementations
# ---------------------------
class PFC(Module):
    def __init__(self, name: str, spec: Dict[str, Any]):
        super().__init__(name, spec)
        # Initialize neural population for PFC
        num_neurons = spec.get("num_neurons", 100)
        self.neural_population = NeuralPopulation(
            population_id=f"pfc_{name}",
            num_neurons=num_neurons,
            neuron_type="regular_spiking",
            connectivity=0.15
        )
        self.current_time = 0.0
        
    def step(self, inbox, ctx):
        wm_conf = ctx.get("wm_confidence", 0.5)
        arousal = ctx["global"].get("arousal", 0.5)
        mode = ctx["global"].get("mode", "internal")
        
        # Generate external inputs based on context
        external_inputs = []
        for i in range(self.neural_population.num_neurons):
            # Base input from arousal and working memory confidence
            base_input = (arousal + wm_conf) * 15.0  # Increased scale for more spiking
            
            # Add some noise and variation
            noise = random.uniform(-2.0, 2.0)
            task_modulation = 5.0 if mode == "task-positive" else 3.0
            
            input_current = base_input + noise + task_modulation
            external_inputs.append(input_current)
        
        # Step the neural population
        spike_events = self.neural_population.step(
            external_inputs=external_inputs,
            dt=1.0,
            current_time=self.current_time
        )
        
        # Update time
        self.current_time += 1.0
        
        # Generate plan based on neural activity
        avg_firing_rate = self.neural_population.get_population_firing_rate()
        plan = {
            "seq": [("think", 1), ("rehearse", 1 if mode=="task-positive" else 0)], 
            "goal": "homeostasis",
            "neural_activity": avg_firing_rate
        }
        
        out = [msg("Command", self.name, "working_memory", action="rehearse", plan=plan)]
        
        # Enhanced telemetry with neural dynamics
        telemetry = {
            "confidence": clamp01(0.35 + 0.45*wm_conf + 0.2*arousal),
            "demand": 0.10,
            "firing_rate": avg_firing_rate,
            "spike_count": sum(spike_events),
            "neural_synchrony": calculate_synchrony(self.neural_population.spike_times),
            "alpha_power": calculate_oscillation_power(self.neural_population.spike_times, 10.0),  # Alpha band
            "beta_power": calculate_oscillation_power(self.neural_population.spike_times, 20.0),   # Beta band
            "gamma_power": calculate_oscillation_power(self.neural_population.spike_times, 40.0),  # Gamma band
            "membrane_potentials": self.neural_population.get_membrane_potentials()[:10]  # First 10 neurons
        }
        return out, telemetry

class BasalGanglia(Module):
    def step(self, inbox, ctx):
        da = ctx["modulators"]["DA"]
        ach = ctx["modulators"]["ACh"]
        fatigue = ctx["global"]["fatigue"]
        moe_k = ctx["policy"]["moe_k"]
        candidates = ["pfc", "working_memory", "dmn", "hippocampus"]
        if "cerebellum" in ctx["topology"]["modules"]:
            candidates.append("cerebellum")
        scores = []
        for c in candidates:
            base = 0.5
            if c == "working_memory":
                base += 0.2*ach
            if c == "dmn":
                base += 0.2*(1.0 - ctx["attention"]["task_bias"])
            if c == "cerebellum":
                base += 0.1
            score = base + 0.4*da - 0.2*fatigue + random.uniform(-0.05, 0.05)
            scores.append((c, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = [c for c,_ in scores[:moe_k]]
        out = [msg("Command", self.name, "thalamus", activate=chosen)]
        telemetry = {"confidence": clamp01(0.5 + 0.4*da - 0.2*fatigue), "demand": 0.05, "chosen": chosen}
        return out, telemetry

class Thalamus(Module):
    def step(self, inbox, ctx):
        out = []
        for m in inbox:
            if m.kind == "Command" and "activate" in m.payload:
                out.append(msg("Telemetry", self.name, "architecture_agent", activated=m.payload["activate"]))
            if m.kind in ("Command", "Observation", "Plan"):
                if "working_memory" in ctx["topology"]["modules"]:
                    out.append(msg(m.kind, self.name, "working_memory", **m.payload))
        telemetry = {"confidence": clamp01(0.6 * ctx["attention"].get("task_bias", 0.6)), "demand": 0.06}
        return out, telemetry

class WorkingMemory(Module):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        slots = spec.get("slots", 3)
        self.slots = [{"vec": [0.0]*8, "age": 0, "precision": 0.45} for _ in range(slots)]
        
        # Initialize neural population for working memory
        num_neurons = spec.get("num_neurons", 50)  # Smaller than PFC
        self.neural_population = NeuralPopulation(
            population_id=f"wm_{name}",
            num_neurons=num_neurons,
            neuron_type="regular_spiking",
            connectivity=0.2  # Higher connectivity for memory maintenance
        )
        self.current_time = 0.0

    def configure(self, slots: int):
        current = len(self.slots)
        if slots > current:
            for _ in range(slots - current):
                self.slots.append({"vec": [0.0]*8, "age": 0, "precision": 0.45})
        elif slots < current:
            self.slots = self.slots[:slots]

    def step(self, inbox, ctx):
        rehearsed = False
        for m in inbox:
            if m.kind == "Command" and m.payload.get("action") == "rehearse":
                for s in self.slots:
                    s["precision"] = min(1.0, s["precision"] + 0.05)
                    s["age"] = 0
                rehearsed = True
            if m.kind == "Replay":
                for s in self.slots:
                    s["precision"] = min(1.0, s["precision"] + 0.02)
        
        for s in self.slots:
            s["age"] += 1
            s["precision"] = max(0.1, s["precision"] - 0.01)
        
        wm_conf = sum(s["precision"] for s in self.slots) / max(1, len(self.slots))
        
        # Generate external inputs based on memory state
        external_inputs = []
        for i in range(self.neural_population.num_neurons):
            # Base input from memory confidence and rehearsal
            base_input = wm_conf * 12.0  # Increased scale for more spiking
            
            # Add rehearsal boost
            rehearsal_boost = 6.0 if rehearsed else 0.0
            
            # Add noise and variation
            noise = random.uniform(-1.0, 1.0)
            
            input_current = base_input + rehearsal_boost + noise
            external_inputs.append(input_current)
        
        # Step the neural population
        spike_events = self.neural_population.step(
            external_inputs=external_inputs,
            dt=1.0,
            current_time=self.current_time
        )
        
        # Update time
        self.current_time += 1.0
        
        # Enhanced telemetry with neural dynamics
        avg_firing_rate = self.neural_population.get_population_firing_rate()
        telemetry = {
            "confidence": wm_conf, 
            "demand": 0.05, 
            "rehearsed": rehearsed, 
            "slots": len(self.slots),
            "firing_rate": avg_firing_rate,
            "spike_count": sum(spike_events),
            "neural_synchrony": calculate_synchrony(self.neural_population.spike_times),
            "persistent_activity": avg_firing_rate > 5.0,  # Indicates persistent activity
            "membrane_potentials": self.neural_population.get_membrane_potentials()[:5]  # First 5 neurons
        }
        out = [msg("Telemetry", self.name, "architecture_agent", wm=telemetry)]
        return out, telemetry

class Hippocampus(Module):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.buffer: List[Dict[str, Any]] = []

    def step(self, inbox, ctx):
        if ctx["global"]["state"] == "wake":
            self.buffer.append({"t": ctx["global"]["t"], "mode": ctx["global"]["mode"]})
            if len(self.buffer) > 512:
                self.buffer = self.buffer[-512:]
        telemetry = {"confidence": 0.6, "demand": 0.04, "episodes": len(self.buffer)}
        out = [msg("Telemetry", self.name, "architecture_agent", hc=len(self.buffer))]
        return out, telemetry

class DMN(Module):
    def step(self, inbox, ctx):
        mode = ctx["global"].get("mode", "internal")
        conf = 0.6 if mode == "internal" else 0.3
        out = []
        if mode == "internal" and ctx["global"]["state"] == "wake":
            out.append(msg("Plan", self.name, "pfc", idea="self-talk"))
        telemetry = {"confidence": conf, "demand": 0.05, "mode": mode}
        return out, telemetry

class SalienceSwitch(Module):
    def step(self, inbox, ctx):
        wm_conf = ctx.get("wm_confidence", 0.5)
        ne = ctx["modulators"]["NE"]
        fatigue = ctx["global"]["fatigue"]
        util_task = 0.5*ne + 0.5*wm_conf - 0.3*fatigue
        util_internal = 0.4*(1-wm_conf) + 0.2 + 0.2*fatigue
        mode = "task-positive" if util_task >= util_internal else "internal"
        out = [msg("Telemetry", self.name, "architecture_agent", suggest_mode=mode)]
        telemetry = {"confidence": clamp01(0.6 + 0.2*ne - 0.2*fatigue), "demand": 0.02, "suggest_mode": mode}
        return out, telemetry

class Attention(Module):
    def step(self, inbox, ctx):
        ne = ctx["modulators"]["NE"]
        bias = 0.5 + 0.4*(ne - 0.5)
        bias = clamp01(bias)
        out = [msg("Telemetry", self.name, "architecture_agent", task_bias=bias)]
        telemetry = {"confidence": bias, "demand": 0.02, "task_bias": bias}
        return out, telemetry

class Cerebellum(Module):
    def step(self, inbox, ctx):
        mode = ctx["global"]["mode"]
        delta = 0.03 if mode == "task-positive" else 0.0
        out = [msg("Telemetry", self.name, "architecture_agent", timing_boost=delta)]
        telemetry = {"confidence": 0.55 + delta, "demand": 0.03}
        return out, telemetry

class SleepConsolidationEngine(Module):
    def step(self, inbox, ctx):
        phase = ctx["sleep"].get("phase", "NREM")
        out = [
            msg("Replay", self.name, "working_memory", phase=phase),
            msg("Replay", self.name, "dmn", phase=phase),
        ]
        telemetry = {"confidence": 0.7 if phase=="REM" else 0.6, "demand": 0.04, "phase": phase}
        return out, telemetry

class ArchitectureAgent(Module):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.modulators = {"DA": 0.3, "NE": 0.45, "ACh": 0.5, "5HT": 0.5}
        self.mode = "internal"      # internal vs task-positive
        self.state = "wake"         # wake vs sleep
        self.fatigue = 0.15
        self.sleep_timer = 0
        self.sleep_period = spec.get("sleep_period", 25)
        self.sleep_length = spec.get("sleep_length", 6)

    def step(self, inbox, ctx):
        for m in inbox:
            if "task_bias" in m.payload:
                ctx["attention"]["task_bias"] = m.payload["task_bias"]
            if "suggest_mode" in m.payload:
                self.mode = m.payload["suggest_mode"]
            if "activated" in m.payload:
                ctx["activated"] = m.payload["activated"]

        # fatigue dynamics
        if self.state == "wake":
            self.fatigue = clamp01(self.fatigue + 0.01)
        else:
            self.fatigue = clamp01(self.fatigue - 0.06)

        # schedule sleep
        self.sleep_timer += 1
        if self.state == "wake" and (self.sleep_timer >= self.sleep_period or self.fatigue > 0.65):
            self.state = "sleep"
            self.sleep_timer = 0
            ctx["sleep"]["ticks_left"] = self.sleep_length
            ctx["sleep"]["phase"] = "NREM"

        # sleep phase management
        if self.state == "sleep":
            ticks_left = ctx["sleep"].get("ticks_left", 0)
            ctx["sleep"]["ticks_left"] = max(0, ticks_left-1)
            if ticks_left % 2 == 0:
                ctx["sleep"]["phase"] = "REM" if ctx["sleep"].get("phase")=="NREM" else "NREM"
            if ctx["sleep"]["phase"] == "NREM":
                self.modulators["NE"] = clamp01(self.modulators["NE"] - 0.05)
                self.modulators["ACh"] = clamp01(self.modulators["ACh"] - 0.02)
            else:
                self.modulators["NE"] = clamp01(self.modulators["NE"] - 0.03)
                self.modulators["ACh"] = clamp01(self.modulators["ACh"] + 0.04)
            if ctx["sleep"]["ticks_left"] == 0:
                self.state = "wake"
        else:
            self.modulators["DA"] = clamp01(self.modulators["DA"] + random.uniform(-0.02, 0.02))
            self.modulators["NE"] = clamp01(self.modulators["NE"] + (0.015 if self.mode=="task-positive" else -0.005))

        out = [
            msg("Modulation", self.name, "attention", NE=self.modulators["NE"]),
            msg("Modulation", self.name, "basal_ganglia", DA=self.modulators["DA"], ACh=self.modulators["ACh"]),
            msg("Command", self.name, "pfc", mode=self.mode),
        ]
        telemetry = {
            "confidence": 0.7,
            "demand": 0.03,
            "modulators": self.modulators.copy(),
            "mode": self.mode,
            "state": self.state,
            "fatigue": self.fatigue,
        }
        return out, telemetry

# ---------------------------
# Curriculum Scheduler
# ---------------------------
class Curriculum:
    def __init__(self, schedule: List[Dict[str, Any]], ticks_per_week: int):
        self.schedule = sorted(schedule, key=lambda s: s.get("at_week", 0))
        self.ticks_per_week = max(1, int(ticks_per_week))
        self.idx = 0

    def update(self, t: int, brain: "Brain", ctx: Dict[str, Any]):
        week = t // self.ticks_per_week
        while self.idx < len(self.schedule) and week >= self.schedule[self.idx].get("at_week", 0):
            step = self.schedule[self.idx]
            stage = step.get("stage")
            if stage:
                brain.stage = stage
            wm_slots = step.get("wm_slots")
            if wm_slots is not None and "working_memory" in brain.modules:
                wm = brain.modules["working_memory"]
                if isinstance(wm, WorkingMemory):
                    wm.configure(int(wm_slots))
            moe_k = step.get("moe_k")
            if moe_k is not None:
                ctx["policy"]["moe_k"] = int(moe_k)
            self.idx += 1

# ---------------------------
# Loader & Orchestrator
# ---------------------------
MODULE_IMPLS = {
    "architecture_agent": ArchitectureAgent,
    "pfc": PFC,
    "basal_ganglia": BasalGanglia,
    "thalamus": Thalamus,
    "working_memory": WorkingMemory,
    "hippocampus": Hippocampus,
    "dmn": DMN,
    "attention": Attention,
    "salience": SalienceSwitch,
    "sleeper": SleepConsolidationEngine,
    "cerebellum": Cerebellum,
}

class Brain:
    def __init__(self, config: Dict[str, Any], stage: str, curriculum: Curriculum=None, log_csv: str=None, dot_every: int=0, dot_dir: str="graphs"):
        self.config = config
        self.stage = stage
        self.curriculum = curriculum
        self.modules: Dict[str, Module] = {}
        self.mailboxes: Dict[str, List[Message]] = {}
        self.t = 0
        # logging
        self.log_csv = log_csv
        if self.log_csv:
            with open(self.log_csv, "w", newline="", encoding="utf-8") as f:
                import csv
                w = csv.writer(f)
                w.writerow(["t","week","state","mode","fatigue","DA","NE","ACh","PFC","WM","WM_slots","BG","Thal","DMN","SN","AttBias","SCE_phase","CB"])
        # dot export
        self.dot_every = max(0, int(dot_every))
        self.dot_dir = dot_dir
        if self.dot_every > 0:
            os.makedirs(self.dot_dir, exist_ok=True)

        # Instantiate modules by baseline stage
        base = ["architecture_agent", "pfc", "basal_ganglia", "thalamus", "working_memory", "hippocampus", "dmn", "attention"]
        stage_add = {
            "F": [],
            "N0": ["salience", "sleeper"],
            "N1": ["salience", "sleeper", "cerebellum"],
        }
        for name in base + stage_add.get(stage, []):
            spec = self._get_spec(name)
            impl = MODULE_IMPLS[name]
            mod = impl(name, spec)
            self.modules[name] = mod
            self.mailboxes[name] = []

        # Configure WM per stage default
        wm: WorkingMemory = self.modules["working_memory"]  # type: ignore
        if isinstance(wm, WorkingMemory):
            slots = {"F": 3, "N0": 3, "N1": 4}.get(stage, 3)
            wm.configure(slots)

    def _get_spec(self, name: str) -> Dict[str, Any]:
        if name == "attention":
            return self.config.get("attention", {})
        if name == "architecture_agent":
            return self.config.get("architecture_agent", {})
        return self.config.get("modules", {}).get(name, {})

    def export_dot(self, t: int):
        nodes = list(self.modules.keys())
        lines = ["digraph Brain {", '  rankdir=LR;', '  node [shape=box, style="rounded,filled", fillcolor="#eef7ff"];']
        for n in nodes:
            lines.append(f'  "{n}";')
        edges = [
            ("architecture_agent","pfc"), ("pfc","working_memory"), ("pfc","basal_ganglia"),
            ("basal_ganglia","thalamus"), ("thalamus","working_memory"), ("working_memory","architecture_agent"),
            ("dmn","pfc"), ("hippocampus","dmn"), ("attention","architecture_agent")
        ]
        if "salience" in nodes:
            edges.append(("salience","architecture_agent"))
        if "cerebellum" in nodes:
            edges.append(("cerebellum","architecture_agent"))
        for a,b in edges:
            if a in nodes and b in nodes:
                lines.append(f'  "{a}" -> "{b}";')
        lines.append("}")
        path = os.path.join(self.dot_dir, f"brain_t{t:05d}.dot")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def log_metrics(self, t: int, tel: Dict[str, Any], ticks_per_week: int):
        if not self.log_csv:
            return
        week = t // ticks_per_week if ticks_per_week>0 else 0
        mods = tel["mods"]
        row = [
            tel["t"], week, tel["state"], tel["mode"], tel["fatigue"],
            mods["DA"], mods["NE"], mods["ACh"],
            tel.get("pfc",{}).get("confidence",0.0),
            tel.get("working_memory",{}).get("confidence",0.0),
            tel.get("working_memory",{}).get("slots","-"),
            tel.get("basal_ganglia",{}).get("confidence",0.0),
            tel.get("thalamus",{}).get("confidence",0.0),
            tel.get("dmn",{}).get("confidence",0.0),
            tel.get("salience",{}).get("confidence",0.0),
            tel.get("attention",{}).get("task_bias",0.0),
            tel.get("sleeper",{}).get("phase","-"),
            tel.get("cerebellum",{}).get("confidence","-"),
        ]
        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            w.writerow(row)

    def step(self, ticks_per_week: int):
        self.t += 1
        ctx = {
            "global": {"t": self.t, "arousal": 0.55, "mode": "internal", "state": "wake", "fatigue": 0.2},
            "modulators": {"DA": 0.3, "NE": 0.45, "ACh": 0.5, "5HT": 0.5},
            "attention": {"task_bias": 0.6},
            "wm_confidence": 0.5,
            "policy": {"moe_k": {"F":1, "N0":2, "N1":2}.get(self.stage, 1)},
            "sleep": {"ticks_left": 0, "phase": "NREM"},
            "topology": {"modules": list(self.modules.keys())},
        }

        if self.curriculum:
            self.curriculum.update(self.t, self, ctx)

        aa = self.modules["architecture_agent"]
        aa_out, aa_tel = aa.step(self.mailboxes["architecture_agent"], ctx)
        self.mailboxes["architecture_agent"].clear()
        ctx["modulators"] = aa_tel["modulators"]
        ctx["global"]["mode"] = aa_tel["mode"]
        ctx["global"]["state"] = aa_tel["state"]
        ctx["global"]["fatigue"] = aa_tel["fatigue"]

        if ctx["global"]["state"] == "sleep" and "sleeper" in self.modules:
            sl = self.modules["sleeper"]
            sl_out, sl_tel = sl.step(self.mailboxes["sleeper"], ctx)
            self.mailboxes["sleeper"].clear()
            for m in sl_out:
                if m.dst in self.mailboxes:
                    self.mailboxes[m.dst].append(m)

        for m in aa_out:
            if m.dst in self.mailboxes:
                self.mailboxes[m.dst].append(m)

        order = [k for k in self.modules.keys() if k != "architecture_agent"]
        telemetry_log = {
            "t": self.t,
            "mode": ctx["global"]["mode"],
            "state": ctx["global"]["state"],
            "fatigue": ctx["global"]["fatigue"],
            "mods": ctx["modulators"],
        }
        for name in order:
            mod = self.modules[name]
            inbox = self.mailboxes[name]
            self.mailboxes[name] = []
            out, tel = mod.step(inbox, ctx)
            if name == "working_memory":
                ctx["wm_confidence"] = tel.get("confidence", ctx["wm_confidence"])
            if name == "attention":
                ctx["attention"]["task_bias"] = tel.get("task_bias", ctx["attention"]["task_bias"])
            if name == "salience" and "suggest_mode" in tel:
                ctx["global"]["mode"] = tel["suggest_mode"]
            telemetry_log[name] = tel
            for m in out:
                if m.dst in self.mailboxes:
                    self.mailboxes[m.dst].append(m)

        tpw = self.curriculum.ticks_per_week if self.curriculum else 100
        self.log_metrics(self.t, telemetry_log, tpw)

        if self.dot_every > 0 and self.t % self.dot_every == 0:
            self.export_dot(self.t)

        return telemetry_log

def load_connectome(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", required=True, help="Path to connectome_v3.yaml")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--stage", choices=["F", "N0", "N1"], default="F")
    ap.add_argument("--log_csv", default=None, help="CSV file to write metrics (default: data/metrics/brain_metrics.csv)")
    ap.add_argument("--dot_every", type=int, default=0, help="Export DOT every N ticks (0=off)")
    ap.add_argument("--dot_dir", default="graphs", help="Directory for DOT files")
    args = ap.parse_args()

    cfg = load_connectome(args.connectome)

    # Ensure metrics directory exists
    if args.log_csv:
        import os
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)

    cur_cfg = cfg.get("curriculum", {})
    schedule = cur_cfg.get("schedule", [])
    tpw = int(cur_cfg.get("ticks_per_week", 50))
    curriculum = Curriculum(schedule, tpw) if schedule else None

    brain = Brain(cfg, stage=args.stage, curriculum=curriculum, log_csv=args.log_csv, dot_every=args.dot_every, dot_dir=args.dot_dir)

    print(f"=== brain scaffold v3 (stage={args.stage}) ===")
    for _ in range(args.steps):
        tel = brain.step(tpw)
        mods = tel["mods"]
        print(
            f"[t={tel['t']:03d}] {tel['state']:<5} {tel['mode']:<13} fat={tel['fatigue']:.2f}  "
            f"DA={mods['DA']:.2f} NE={mods['NE']:.2f} ACh={mods['ACh']:.2f}  "
            f"PFC={tel.get('pfc',{}).get('confidence',0):.2f}  "
            f"WM={tel.get('working_memory',{}).get('confidence',0):.2f}({tel.get('working_memory',{}).get('slots','-')})  "
            f"BG={tel.get('basal_ganglia',{}).get('confidence',0):.2f}  "
            f"Thal={tel.get('thalamus',{}).get('confidence',0):.2f}  "
            f"DMN={tel.get('dmn',{}).get('confidence',0):.2f}  "
            f"SN={tel.get('salience',{}).get('confidence',0):.2f}  "
            f"AttBias={tel.get('attention',{}).get('task_bias',0):.2f}  "
            f"SCE={tel.get('sleeper',{}).get('phase','-')}  "
            f"CB={tel.get('cerebellum',{}).get('confidence','-')}"
        )

if __name__ == "__main__":
    main()
