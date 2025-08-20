import argparse, os, sys, json

def cmd_list(a): 
    print("Available models:")
    print("  neuro.scanner    - File system scanner")
    print("  neuro.connectome - Build neural connectome")

def cmd_describe(a):
    print(f"Model: {a.model_id}")
    print("Type: neuro")
    print("Capabilities: scanning, analysis")

def cmd_ask(a):
    print(f"Processing: {a.prompt}")
    if "scan" in a.prompt.lower():
        print("Running file scan...")
        os.system("python3 -m neuro.cli scan")
    else:
        print("Use 'scan' in your prompt to scan files")

def cmd_run(a):
    print(f"Running task: {' '.join(a.task or [])}")

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd')
    s=sub.add_parser('list'); s.set_defaults(func=cmd_list)
    s=sub.add_parser('describe'); s.add_argument('model_id'); s.set_defaults(func=cmd_describe)
    s=sub.add_parser('ask'); s.add_argument('prompt'); s.add_argument('--tools',default=''); s.add_argument('--allow-shell',action='store_true'); s.add_argument('--sudo-ok',action='store_true'); s.set_defaults(func=cmd_ask)
    s=sub.add_parser('run'); s.add_argument('--model',required=True); s.add_argument('--allow-shell',action='store_true'); s.add_argument('--sudo-ok',action='store_true'); s.add_argument('task', nargs='*'); s.set_defaults(func=cmd_run)
    a=p.parse_args(); 
    if not getattr(a,'func',None): return p.print_help()
    a.func(a)
if __name__=='__main__': main()
