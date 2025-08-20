from __future__ import annotations
import os, pathlib, json, textwrap
from typing import Dict, List
from .....................................................config import ROOT

def ensure_agent_hub():
    """Create basic agent_hub structure."""
    agent_hub_dir = ROOT / "agent_hub"
    agent_hub_dir.mkdir(exist_ok=True)
    
    # Create basic __init__.py
    init_file = agent_hub_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# agent_hub package\n")
    
    # Create basic cli.py
    cli_file = agent_hub_dir / "cli.py"
    if not cli_file.exists():
        cli_file.write_text(textwrap.dedent("""\
            import argparse
            def cmd_list(a): 
                print("No models configured yet")
            def main():
                p = argparse.ArgumentParser()
                sub = p.add_subparsers(dest='cmd')
                s = sub.add_parser('list')
                s.set_defaults(func=cmd_list)
                a = p.parse_args()
                if getattr(a, 'func', None):
                    a.func(a)
                else:
                    p.print_help()
            if __name__ == '__main__': main()
        """))
    
    return True
