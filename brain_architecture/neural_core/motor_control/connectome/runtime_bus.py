# connectome/runtime_bus.py
# Lightweight asyncio message bus + gating hooks you can embed or extend.
import asyncio
from typing import Dict, Any, Callable
import json
import os
import time

class Neuromodulators:
    def __init__(self, dopamine=1.0, serotonin=1.0, norepinephrine=1.0, acetylcholine=1.0):
        self.levels = {
            "dopamine": dopamine,
            "serotonin": serotonin,
            "norepinephrine": norepinephrine,
            "acetylcholine": acetylcholine,
        }

    def as_gain(self) -> float:
        # Simple multiplicative gain; customize as needed.
        g = 1.0
        for v in self.levels.values():
            g *= max(0.25, min(1.75, v))
        return g ** 0.25

class ConnectomeBus:
    """In-process pub/sub with gating & sleep hooks; replace with ZMQ/HTTP as needed."""
    def __init__(self):
        self.channels: Dict[str, asyncio.Queue] = {}
        self.gating: Dict[str, float] = {}  # channel -> [0..1] gate
        self.nmods = Neuromodulators()
        self.sleeping = False

    def get_channel(self, name: str) -> asyncio.Queue:
        if name not in self.channels:
            self.channels[name] = asyncio.Queue(maxsize=1024)
        return self.channels[name]

    def set_gate(self, name: str, value: float):
        self.gating[name] = max(0.0, min(1.0, value))

    def set_sleep(self, asleep: bool):
        self.sleeping = asleep

    async def publish(self, name: str, payload: Dict[str, Any]):
        if self.sleeping:
            return
        gain = self.nmods.as_gain()
        gate = self.gating.get(name, 1.0)
        eff = gate * gain
        if eff <= 0.05:
            return
        q = self.get_channel(name)
        item = {"payload": payload, "gain": gain, "gate": gate, "ts": time.time()}
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # Backpressure: drop oldest to keep up, like synaptic depression
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            await q.put(item)

    async def subscribe(self, name: str, handler: Callable[[Dict[str, Any]], None]):
        q = self.get_channel(name)
        while True:
            msg = await q.get()
            handler(msg)

def read_telemetry_sleep_flag(path="runtime/telemetry.json",
                              cog=85, mem=80, err=15) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            t = json.load(f)
        return (t.get("cognitive_load_pct", 0) > cog or
                t.get("memory_usage_pct", 0) > mem or
                t.get("error_rate_pct", 0) > err)
    except Exception:
        return False
