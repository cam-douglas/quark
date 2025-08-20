import argparse, os, json, sys
from typing import List
from ................................................registry import ModelRegistry
from ................................................planner import infer_needs, auto_route_request
from ................................................router import choose_model
from ................................................runner import run_model

def _parse_tools(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def cmd_list(args):
    reg = ModelRegistry()
    for m in reg.list():
        print(f"{m['id']:20s}  type={m['type']}  caps={','.join(m.get('capabilities',[]))}")

def cmd_describe(args):
    reg = ModelRegistry()
    m = reg.get(args.model_id)
    print(json.dumps(m, indent=2))

def cmd_auto(args):
    """Automatically route requests without explicit commands."""
    reg = ModelRegistry()
    
    # Auto-detect intent and route
    auto_result = auto_route_request(args.prompt)
    
    print(f"🤖 Auto-detected intent: {auto_result['intent']['primary_intent']}")
    print(f"🎯 Recommended action: {auto_result['routing']['action']}")
    print(f"🧠 Model type: {auto_result['routing']['model_type']}")
    print(f"⚡ Priority: {auto_result['routing']['priority']}")
    print(f"📝 Response format: {auto_result['response_config']['format']}")
    
    # Get tools if specified
    tools = _parse_tools(args.tools) if args.tools else []
    
    # Merge auto-detected needs with explicit tools
    needs = auto_result['needs']
    if tools:
        needs['need'].extend(tools)
        needs['need'] = list(set(needs['need']))  # Remove duplicates
    
    # Choose best model
    model_id = choose_model(needs, reg.routing, reg)
    m = reg.get(model_id)
    
    print(f"🚀 Selected model: {model_id}")
    print(f"🔧 Capabilities: {', '.join(m.get('capabilities', []))}")
    
    # Execute with auto-detected configuration
    out = run_model(m, args.prompt, 
                   allow_shell=args.allow_shell, 
                   sudo_ok=args.sudo_ok,
                   timeout=auto_result['routing'].get('timeout', 300))
    
    # Format response based on auto-detected config
    response = _format_response(out, auto_result['response_config'])
    sys.stdout.write(response)
    
    if args.show_run_dir:
        sys.stderr.write(f"\n[run_dir] {out['run_dir']}\n")

def cmd_ask(args):
    reg = ModelRegistry()
    tools = _parse_tools(args.tools) if args.tools else []
    needs = infer_needs(args.prompt, tools)
    model_id = choose_model(needs, reg.routing, reg)
    m = reg.get(model_id)
    out = run_model(m, args.prompt, allow_shell=args.allow_shell, sudo_ok=args.sudo_ok)
    sys.stdout.write(out["result"].get("stdout",""))
    if args.show_run_dir:
        sys.stderr.write(f"\n[run_dir] {out['run_dir']}\n")

def cmd_plan(args):
    reg = ModelRegistry()
    needs = {"need":["planning"]}
    model_id = choose_model(needs, reg.routing, reg)
    m = reg.get(model_id)
    out = run_model(m, args.goal, allow_shell=False, sudo_ok=False)
    sys.stdout.write(out["result"].get("stdout",""))

def cmd_run(args):
    reg = ModelRegistry()
    m = reg.get(args.model)
    prompt = " ".join(args.task) if args.task else ""
    out = run_model(m, prompt, allow_shell=args.allow_shell, sudo_ok=args.sudo_ok)
    sys.stdout.write(out["result"].get("stdout",""))

def cmd_analyze(args):
    """Analyze and investigate requests."""
    reg = ModelRegistry()
    needs = infer_needs(args.prompt, ["reasoning", "analysis"])
    model_id = choose_model(needs, reg.routing, reg)
    m = reg.get(model_id)
    out = run_model(m, args.prompt, allow_shell=False, sudo_ok=False)
    sys.stdout.write(out["result"].get("stdout",""))

def cmd_create(args):
    """Create and generate content."""
    reg = ModelRegistry()
    needs = infer_needs(args.prompt, ["creation", "code"])
    model_id = choose_model(needs, reg.routing, reg)
    m = reg.get(model_id)
    out = run_model(m, args.prompt, allow_shell=args.allow_shell, sudo_ok=args.sudo_ok)
    sys.stdout.write(out["result"].get("stdout",""))

def cmd_optimize(args):
    """Optimize and improve existing solutions."""
    reg = ModelRegistry()
    needs = infer_needs(args.prompt, ["optimization", "planning"])
    model_id = choose_model(needs, reg.routing, reg)
    m = reg.get(model_id)
    out = run_model(m, args.prompt, allow_shell=args.allow_shell, sudo_ok=args.sudo_ok)
    sys.stdout.write(out["result"].get("stdout",""))

def cmd_parallel(args):
    # delegate to zsh-level agent-run for UI; here we just print helper
    cmds = args.commands
    if not cmds: print("smctl parallel \"cmd1\" \"cmd2\" ..."); return 1
    quoted = " ".join([f"\"{c}\"" for c in cmds])
    print(f"Run in shell: agent-run {quoted}")

def cmd_shell(args):
    os.execvp("zsh", ["zsh","-lc","rescap; openinterpreter || zsh"])

def _format_response(output: dict, response_config: dict) -> str:
    """Format response based on auto-detected configuration."""
    result = output.get("result", {})
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    
    # Apply response formatting based on config
    if response_config["format"] == "structured":
        return f"📋 **Structured Response**\n\n{stdout}\n"
    elif response_config["format"] == "detailed":
        return f"🔍 **Detailed Analysis**\n\n{stdout}\n"
    elif response_config["format"] == "creative":
        return f"🎨 **Creative Solution**\n\n{stdout}\n"
    elif response_config["format"] == "step_by_step":
        return f"📝 **Step-by-Step Process**\n\n{stdout}\n"
    else:
        return stdout

def main():
    p = argparse.ArgumentParser(prog="agent_hub.cli")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("list"); sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("describe")
    sp.add_argument("model_id"); sp.set_defaults(func=cmd_describe)

    # Auto-routing command (no explicit action needed)
    sp = sub.add_parser("auto")
    sp.add_argument("prompt")
    sp.add_argument("--tools", default="")
    sp.add_argument("--allow-shell", action="store_true")
    sp.add_argument("--sudo-ok", action="store_true")
    sp.add_argument("--show-run-dir", action="store_true")
    sp.set_defaults(func=cmd_auto)

    sp = sub.add_parser("ask")
    sp.add_argument("prompt")
    sp.add_argument("--tools", default="")
    sp.add_argument("--allow-shell", action="store_true")
    sp.add_argument("--sudo-ok", action="store_true")
    sp.add_argument("--show-run-dir", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("plan")
    sp.add_argument("goal"); sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run")
    sp.add_argument("--model", required=True)
    sp.add_argument("--allow-shell", action="store_true")
    sp.add_argument("--sudo-ok", action="store_true")
    sp.add_argument("task", nargs=argparse.REMAINDER)
    sp.set_defaults(func=cmd_run)

    # New intent-specific commands
    sp = sub.add_parser("analyze")
    sp.add_argument("prompt"); sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("create")
    sp.add_argument("prompt")
    sp.add_argument("--allow-shell", action="store_true")
    sp.add_argument("--sudo-ok", action="store_true")
    sp.set_defaults(func=cmd_create)

    sp = sub.add_parser("optimize")
    sp.add_argument("prompt")
    sp.add_argument("--allow-shell", action="store_true")
    sp.add_argument("--sudo-ok", action="store_true")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("parallel")
    sp.add_argument("commands", nargs="*")
    sp.set_defaults(func=cmd_parallel)

    sp = sub.add_parser("shell"); sp.set_defaults(func=cmd_shell)

    args = p.parse_args()
    if not getattr(args, "func", None): p.print_help(); return 1
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main() or 0)
