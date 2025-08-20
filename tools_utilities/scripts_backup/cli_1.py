from __future__ import annotations
import argparse, json, pathlib, sys
from .....................................................config import ROOT, DEFAULT_MODELS
from .....................................................scanners import scan_files, read_text_safe
from .....................................................analyzers import analyze_file
from .....................................................connectome import build_connectome, detect_communities, record_coactivation
from .....................................................composer import compose_and_write
from .....................................................actions import ensure_agent_hub
from .....................................................organizer import suggest_organization, organize_files, generate_file_titles, ORGANIZATION_RULES

def cmd_scan(args):
    files = scan_files()
    out = [{"path": str(p), "size": p.stat().st_size if p.exists() else 0} for p in files]
    print(json.dumps(out, indent=2))

def cmd_analyze(args):
    files = scan_files()
    rep = []
    for p in files:
        text = read_text_safe(p)
        analysis = analyze_file(p, text)
        titles = generate_file_titles(p) if args.titles else []
        rep.append({"path": str(p), "titles": titles, **analysis})
    print(json.dumps(rep, indent=2))

def cmd_organize(args):
    suggestions = suggest_organization(dry_run=args.dry_run)
    
    if args.dry_run:
        print("=== FILE ORGANIZATION SUGGESTIONS ===")
        for category, files in suggestions.items():
            desc = ORGANIZATION_RULES[category]["description"]
            print(f"\n📁 {category}/ - {desc}")
            for file_path, _, score in files:
                print(f"  • {file_path.name} (confidence: {score:.1f})")
        
        total_files = sum(len(files) for files in suggestions.values())
        print(f"\nTotal files to organize: {total_files}")
        print("Run with --execute to perform the organization.")
    else:
        results = organize_files(suggestions, dry_run=False)
        print("=== ORGANIZATION RESULTS ===")
        
        if results["created_dirs"]:
            print("\n📁 Created directories:")
            for dir_path in results["created_dirs"]:
                print(f"  • {dir_path}")
                
        if results["moved"]:
            print("\n📦 Moved files:")
            for move_info in results["moved"]:
                print(f"  • {move_info}")
                
        if results["errors"]:
            print("\n❌ Errors:")
            for error in results["errors"]:
                print(f"  • {error}")

def cmd_connectome(args):
    try:
        G = build_connectome()
        comms = detect_communities(G)
        payload = {
            "nodes": G["nodes"],
            "edges": G["edges"], 
            "communities": comms,
        }
        print(json.dumps(payload, indent=2))
    except ImportError as e:
        print(f"Missing dependency for connectome: {e}")
        print("Install with: pip install networkx")

def cmd_compose(args):
    ensure_agent_hub()
    res = compose_and_write()
    if args.learn and res["new_agents"]:
        # Treat new agent cluster as a co-activation episode
        for cl in res["clusters"]:
            record_coactivation(cl)
    print(json.dumps({
        "models_yaml": str(DEFAULT_MODELS),
        "new_agents": res["new_agents"],
        "clusters": res["clusters"]
    }, indent=2))

def main():
    ap = argparse.ArgumentParser(prog="neuro")
    sub = ap.add_subparsers(dest="cmd")

    s=sub.add_parser("scan"); s.set_defaults(func=cmd_scan)
    
    s=sub.add_parser("analyze"); s.add_argument("--titles", action="store_true", help="Extract file titles"); s.set_defaults(func=cmd_analyze)
    
    s=sub.add_parser("organize")
    s.add_argument("--dry-run", action="store_true", default=True, help="Show suggestions without moving files")
    s.add_argument("--execute", dest="dry_run", action="store_false", help="Actually organize the files")
    s.set_defaults(func=cmd_organize)
    
    s=sub.add_parser("connectome"); s.set_defaults(func=cmd_connectome)
    s=sub.add_parser("compose"); s.add_argument("--learn", action="store_true"); s.set_defaults(func=cmd_compose)

    a = ap.parse_args()
    if not getattr(a,"func",None):
        ap.print_help(); return 1
    return a.func(a)

if __name__ == "__main__":
    sys.exit(main() or 0)
