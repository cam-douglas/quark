# connectome/cli.py
# Simple CLI: build / validate once, or watch repo and auto-rebuild on changes.

import argparse
import os, sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from connectome_manager import compile_connectome, validate_connectome, apply_sleep_gating
from maintenance_agent import start_maintenance_agent, stop_maintenance_agent, get_maintenance_status
from advanced_monitoring_agent import start_advanced_monitoring, stop_advanced_monitoring, get_advanced_monitoring_status, rollback_connectome

WATCH_PATTERNS = [
    "connectome/connectome.yaml",
    ".cursor/rules",
    "database",
    "cognition",
    "learning",
    "self_org",
    "sensors",
    "morpho",
    "connectome",
]

class RebuildOnChange(FileSystemEventHandler):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.last_build = 0.0

    def on_any_event(self, event):
        now = time.time()
        # debounce
        if now - self.last_build < 0.5:
            return
        self.last_build = now
        try:
            print("[connectome] change detected → recompiling…")
            summary = compile_connectome(self.config_path)
            report = validate_connectome(self.config_path)
            apply_sleep_gating(self.config_path)
            print("[connectome] build:", summary)
            print("[connectome] validation:", report)
        except Exception as e:
            print("[connectome] ERROR during rebuild:", e, file=sys.stderr)

def cmd_build(args):
    summary = compile_connectome(args.config)
    print("BUILD SUMMARY:", summary)

def cmd_validate(args):
    report = validate_connectome(args.config)
    print("VALIDATION:", report)

def cmd_sleep(args):
    apply_sleep_gating(args.config)
    print("SLEEP STATE UPDATED (from telemetry)")

def cmd_maintenance_start(args):
    start_maintenance_agent()
    print("MAINTENANCE AGENT STARTED")

def cmd_maintenance_stop(args):
    stop_maintenance_agent()
    print("MAINTENANCE AGENT STOPPED")

def cmd_maintenance_status(args):
    status = get_maintenance_status()
    print("MAINTENANCE STATUS:")
    import json
    print(json.dumps(status, indent=2))

def cmd_monitoring_start(args):
    start_advanced_monitoring()
    print("ADVANCED MONITORING STARTED")

def cmd_monitoring_stop(args):
    stop_advanced_monitoring()
    print("ADVANCED MONITORING STOPPED")

def cmd_monitoring_status(args):
    status = get_advanced_monitoring_status()
    print("ADVANCED MONITORING STATUS:")
    import json
    print(json.dumps(status, indent=2))

def cmd_rollback(args):
    success = rollback_connectome(args.version)
    if success:
        print(f"Successfully rolled back to version {args.version}")
    else:
        print(f"Failed to rollback to version {args.version}")

def cmd_versions(args):
    status = get_advanced_monitoring_status()
    versions = status.get("version_count", 0)
    print(f"Total versions: {versions}")
    if versions > 0:
        print("Recent versions:")
        # This would be enhanced to show actual version list

def cmd_watch(args):
    # initial build
    summary = compile_connectome(args.config)
    report = validate_connectome(args.config)
    apply_sleep_gating(args.config)
    print("[connectome] initial build:", summary)
    print("[connectome] initial validation:", report)

    # Start maintenance and advanced monitoring agents
    start_maintenance_agent()
    start_advanced_monitoring()
    print("[connectome] maintenance and advanced monitoring agents started")

    handler = RebuildOnChange(args.config)
    observer = Observer()
    root = os.getcwd()
    for p in WATCH_PATTERNS:
        path = os.path.join(root, p)
        if os.path.exists(path):
            observer.schedule(handler, path=path, recursive=True)
    observer.start()
    print("[connectome] watching for changes… (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[connectome] stopping maintenance and monitoring agents...")
        stop_maintenance_agent()
        stop_advanced_monitoring()
        observer.stop()
    observer.join()

if __name__ == "__main__":
    ap = argparse.ArgumentParser("connectome")
    sub = ap.add_subparsers()

    a = sub.add_parser("build")
    a.add_argument("--config", default="connectome/connectome.yaml")
    a.set_defaults(func=cmd_build)

    v = sub.add_parser("validate")
    v.add_argument("--config", default="connectome/connectome.yaml")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("sleep")
    s.add_argument("--config", default="connectome/connectome.yaml")
    s.set_defaults(func=cmd_sleep)

    w = sub.add_parser("watch")
    w.add_argument("--config", default="connectome/connectome.yaml")
    w.set_defaults(func=cmd_watch)

    # Maintenance commands
    m = sub.add_parser("maintenance")
    m_sub = m.add_subparsers()
    
    m_start = m_sub.add_parser("start")
    m_start.set_defaults(func=cmd_maintenance_start)
    
    m_stop = m_sub.add_parser("stop")
    m_stop.set_defaults(func=cmd_maintenance_stop)
    
    m_status = m_sub.add_parser("status")
    m_status.set_defaults(func=cmd_maintenance_status)

    # Advanced monitoring commands
    mon = sub.add_parser("monitoring")
    mon_sub = mon.add_subparsers()
    
    mon_start = mon_sub.add_parser("start")
    mon_start.set_defaults(func=cmd_monitoring_start)
    
    mon_stop = mon_sub.add_parser("stop")
    mon_stop.set_defaults(func=cmd_monitoring_stop)
    
    mon_status = mon_sub.add_parser("status")
    mon_status.set_defaults(func=cmd_monitoring_status)

    # Version management commands
    ver = sub.add_parser("versions")
    ver.set_defaults(func=cmd_versions)
    
    rb = sub.add_parser("rollback")
    rb.add_argument("version", help="Version ID to rollback to")
    rb.set_defaults(func=cmd_rollback)

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()
