#!/usr/bin/env python3
"""
Test MCP Server Connectivity and Status
"""

import json
import subprocess
import sys
import os
from pathlib import Path
import time

def test_mcp_server(name: str, config: dict) -> dict:
    """Test individual MCP server connectivity"""
    
    print(f"\n🔍 Testing {name}...")
    result = {
        "name": name,
        "status": "unknown",
        "command": config.get("command", ""),
        "error": None,
        "recommendation": None
    }
    
    # Check if command exists
    command = config.get("command", "")
    if not command:
        result["status"] = "❌ No command configured"
        return result
    
    # Test different command types
    if command == "npm":
        # Check if npm package exists
        try:
            # Check if the prefix directory exists
            args = config.get("args", [])
            for i, arg in enumerate(args):
                if arg == "--prefix":
                    if i + 1 < len(args):
                        prefix_dir = args[i + 1]
                        if not Path(prefix_dir).exists():
                            result["status"] = "❌ Directory not found"
                            result["error"] = f"Directory {prefix_dir} does not exist"
                            result["recommendation"] = f"Install MCP server: cd {Path(prefix_dir).parent} && git clone [repo] && npm install"
                            return result
        except Exception as e:
            result["error"] = str(e)
            
    elif command == "npx":
        # Check if package is available via npx
        args = config.get("args", [])
        if "-y" in args and len(args) > 1:
            package_name = args[args.index("-y") + 1] if args.index("-y") + 1 < len(args) else args[1]
            
            # Test if package exists
            test_cmd = ["npm", "view", package_name, "version"]
            try:
                output = subprocess.run(test_cmd, capture_output=True, text=True, timeout=5)
                if output.returncode == 0:
                    result["status"] = "✅ Package available"
                    result["version"] = output.stdout.strip()
                else:
                    result["status"] = "⚠️ Package not found on npm"
                    result["error"] = output.stderr
            except subprocess.TimeoutExpired:
                result["status"] = "⚠️ NPM timeout"
                result["error"] = "npm view command timed out"
            except Exception as e:
                result["status"] = "❌ NPM error"
                result["error"] = str(e)
                
    elif command == "node":
        # Check if Node.js script exists
        args = config.get("args", [])
        if args:
            script_path = Path(args[0])
            if not script_path.exists():
                result["status"] = "❌ Script not found"
                result["error"] = f"Script {script_path} does not exist"
                result["recommendation"] = f"Build or install the MCP server at {script_path}"
            else:
                result["status"] = "✅ Script exists"
                
    elif command.endswith("python") or command.endswith("python3"):
        # Check if Python module/script exists
        args = config.get("args", [])
        
        # Check if it's a virtual environment python
        if "/.venv/bin/python" in command:
            venv_dir = Path(command).parent.parent
            if not Path(command).exists():
                result["status"] = "❌ Virtual environment not found"
                result["error"] = f"Venv at {venv_dir} does not exist"
                result["recommendation"] = f"Create venv: cd {venv_dir} && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
                return result
            else:
                result["status"] = "✅ Virtual environment exists"
                
            # Check if the module/script is available
            if "-m" in args:
                module_idx = args.index("-m") + 1
                if module_idx < len(args):
                    module_name = args[module_idx]
                    # Try to check if module is installed
                    check_cmd = [command, "-c", f"import {module_name.replace('-', '_').replace('.', '_')}"]
                    try:
                        output = subprocess.run(check_cmd, capture_output=True, timeout=2)
                        if output.returncode != 0:
                            result["status"] = "⚠️ Module not installed"
                            result["recommendation"] = f"Install module: {command.replace('python', 'pip')} install {module_name}"
                    except:
                        pass
            elif args and Path(args[0]).suffix == ".py":
                if not Path(args[0]).exists():
                    result["status"] = "❌ Script not found"
                    result["error"] = f"Script {args[0]} does not exist"
    
    # Check for required environment variables
    env = config.get("env", {})
    for key, value in env.items():
        if value.startswith("$(") and value.endswith(")"):
            # This is a shell command substitution
            cmd = value[2:-1]  # Extract command
            if "cat" in cmd:
                # Check if file exists
                file_path = cmd.replace("cat ", "").strip()
                file_path = os.path.expanduser(file_path)
                if not Path(file_path).exists():
                    result["status"] = "⚠️ Missing credentials"
                    result["error"] = f"Credential file {file_path} not found"
                    result["recommendation"] = f"Create {file_path} with your API key"
    
    return result

def main():
    """Test all MCP servers"""
    
    # Load MCP configuration
    config_path = Path.home() / ".cursor" / "mcp.json"
    
    if not config_path.exists():
        print("❌ MCP configuration not found at ~/.cursor/mcp.json")
        return 1
    
    with open(config_path) as f:
        config = json.load(f)
    
    servers = config.get("mcpServers", {})
    
    print("=" * 60)
    print("MCP SERVER STATUS CHECK")
    print("=" * 60)
    print(f"Found {len(servers)} MCP servers configured")
    
    results = []
    
    for name, server_config in servers.items():
        result = test_mcp_server(name, server_config)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = []
    needs_install = []
    needs_credentials = []
    errors = []
    
    for r in results:
        if "✅" in r["status"]:
            working.append(r["name"])
        elif "not found" in r["status"] or "not exist" in str(r.get("error", "")):
            needs_install.append(r)
        elif "credentials" in r["status"].lower() or "Missing credentials" in str(r.get("error", "")):
            needs_credentials.append(r)
        else:
            errors.append(r)
    
    print(f"\n✅ Working servers ({len(working)}/{len(results)}):")
    for name in working:
        print(f"  - {name}")
    
    if needs_install:
        print(f"\n📦 Servers needing installation ({len(needs_install)}):")
        for r in needs_install:
            print(f"  - {r['name']}: {r['status']}")
            if r.get("recommendation"):
                print(f"    💡 {r['recommendation']}")
    
    if needs_credentials:
        print(f"\n🔑 Servers needing credentials ({len(needs_credentials)}):")
        for r in needs_credentials:
            print(f"  - {r['name']}: {r['status']}")
            if r.get("recommendation"):
                print(f"    💡 {r['recommendation']}")
    
    if errors:
        print(f"\n❌ Servers with errors ({len(errors)}):")
        for r in errors:
            print(f"  - {r['name']}: {r['status']}")
            if r.get("error"):
                print(f"    Error: {r['error']}")
    
    # Generate fix script
    if needs_install or needs_credentials:
        print("\n" + "=" * 60)
        print("AUTOMATED FIX SCRIPT")
        print("=" * 60)
        
        script_lines = ["#!/bin/bash", "# MCP Server Installation Script", ""]
        
        if needs_install:
            script_lines.append("# Install missing servers")
            for r in needs_install:
                if "npm" in r["command"] or "node" in r["command"]:
                    # Guess the installation based on common patterns
                    script_lines.append(f"# Install {r['name']}")
                    if r.get("recommendation"):
                        script_lines.append(r["recommendation"])
                elif "python" in r["command"]:
                    script_lines.append(f"# Install {r['name']}")
                    if r.get("recommendation"):
                        script_lines.append(r["recommendation"])
                script_lines.append("")
        
        if needs_credentials:
            script_lines.append("# Add missing credentials")
            for r in needs_credentials:
                script_lines.append(f"# {r['name']}")
                if r.get("recommendation"):
                    script_lines.append(f"echo 'YOUR_API_KEY_HERE' > {r['recommendation'].split()[-1]}")
            script_lines.append("")
        
        script_path = Path("/Users/camdouglas/quark/fix_mcp_servers.sh")
        with open(script_path, "w") as f:
            f.write("\n".join(script_lines))
        
        print(f"Fix script generated: {script_path}")
        print("Review and run: chmod +x fix_mcp_servers.sh && ./fix_mcp_servers.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
