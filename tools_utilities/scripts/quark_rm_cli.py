#!/usr/bin/env python3
"""Quark Resource Manager CLI

Usage:
  quark rm scan                # list unapproved resources detected in /data
  quark rm approve <RID>       # integrate a resource by id
  quark rm reject  <RID>       # blacklist a resource by id

This thin CLI delegates to ResourceManager.

Integration: Not simulator-integrated; repository tooling for indexing, validation, or CI.
Rationale: Executed by developers/CI to maintain repo health; not part of runtime simulator loop.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # add repo root

from brain.architecture.neural_core.cognitive_systems.resource_manager import ResourceManager


def main():
    parser = argparse.ArgumentParser(prog="quark rm", description="Resource Manager CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan", help="List unapproved resources in /data")

    p_appr = sub.add_parser("approve", help="Approve resource by id")
    p_appr.add_argument("rid", help="Resource id (first 16 hex chars)")

    p_rej = sub.add_parser("reject", help="Reject resource by id")
    p_rej.add_argument("rid", help="Resource id (first 16 hex chars)")

    args = parser.parse_args()
    rm = ResourceManager(auto_scan=False)

    if args.cmd == "scan":
        pending = [ (rid, meta) for rid, meta in rm.registry.items() if not meta.get("approved", False) ]
        if not pending:
            print("✅ No pending resources.")
        else:
            for rid, meta in pending:
                print(f"{rid}  {meta['path']}  {meta['size_bytes']/1e6:.1f} MB  {meta.get('resource_type','?')}")
    elif args.cmd == "approve":
        rm.approve(args.rid)
    elif args.cmd == "reject":
        rm.reject(args.rid)

if __name__ == "__main__":
    main()
