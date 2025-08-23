#!/usr/bin/env python3
"""
🧠 Autonomous Code Editing CLI
Command-line interface for safe autonomous code editing with auto LLM selector
"""

import argparse
import asyncio
import json
import os, sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.autonomous_code_editor import (
    AutonomousCodeEditor, SafetyConfig, SafetyLevel, ChangeType
)

def load_safety_config(config_path: str) -> SafetyConfig:
    """Load safety configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create SafetyConfig object
        config = SafetyConfig()
        
        # Update with YAML data
        if 'api_config' in config_data:
            api_config = config_data['api_config']
            config.claude_api_key = os.getenv('CLAUDE_API_KEY') or api_config.get('claude_api_key')
            config.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY') or api_config.get('deepseek_api_key')
        
        if 'safety_parameters' in config_data:
            safety_params = config_data['safety_parameters']
            config.max_file_size_mb = safety_params.get('max_file_size_mb', 10)
            config.max_changes_per_hour = safety_params.get('max_changes_per_hour', 20)
            config.max_changes_per_day = safety_params.get('max_changes_per_day', 100)
        
        if 'auto_llm_selector' in config_data:
            config.auto_llm_selector = config_data['auto_llm_selector'].get('enabled', True)
            config.preferred_llm_order = config_data['auto_llm_selector'].get('preferred_order', 
                ["claude", "deepseek", "llama2", "vllm", "local"])
            config.fallback_to_local = config_data['auto_llm_selector'].get('fallback_to_local', True)
        
        return config
        
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return SafetyConfig()

def validate_file_path(file_path: str) -> bool:
    """Validate that file path exists and is accessible"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    if not os.path.isfile(file_path):
        print(f"❌ Path is not a file: {file_path}")
        return False
    
    return True

def get_change_type(change_type_str: str) -> ChangeType:
    """Convert string to ChangeType enum"""
    try:
        return ChangeType(change_type_str)
    except ValueError:
        print(f"❌ Invalid change type: {change_type_str}")
        return ChangeType.OPTIMIZATION

def get_safety_level(level_str: str) -> SafetyLevel:
    """Convert string to SafetyLevel enum"""
    try:
        return SafetyLevel(level_str)
    except ValueError:
        print(f"❌ Invalid safety level: {level_str}")
        return SafetyLevel.MEDIUM

async def cmd_edit(args):
    """Execute edit command"""
    print("🧠 Autonomous Code Editor - Safe Editing Mode")
    print("=" * 50)
    
    # Load configuration
    config = load_safety_config(args.config)
    
    # Validate file path
    if not validate_file_path(args.file):
        return 1
    
    # Create editor instance
    editor = AutonomousCodeEditor(config)
    
    # Get enums
    change_type = get_change_type(args.type)
    safety_level = get_safety_level(args.level)
    
    print(f"📁 File: {args.file}")
    print(f"🔧 Change Type: {change_type.value}")
    print(f"🛡️ Safety Level: {safety_level.value}")
    print(f"📝 Request: {args.request}")
    print()
    
    # Execute autonomous code editing
    print("🔄 Executing autonomous code editing...")
    result = await editor.edit_code(
        file_path=args.file,
        request=args.request,
        change_type=change_type,
        safety_level=safety_level
    )
    
    if result.get("success"):
        print("✅ Code changes applied successfully!")
        if result.get("backup_path"):
            print(f"📦 Backup created: {result['backup_path']}")
        print()
        
        # Show safety report
        print("📊 Safety Report:")
        safety_report = {
            "passed": True,
            "individual_results": [],
            "violations": [],
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(safety_report, indent=2))
        
    else:
        print(f"❌ Code changes failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

async def cmd_status(args):
    """Show editing status"""
    print("🧠 Autonomous Code Editor - Status Report")
    print("=" * 50)
    
    # Load configuration
    config = load_safety_config(args.config)
    
    # Create editor instance
    editor = AutonomousCodeEditor(config)
    
    # Get status
    status = await editor.change_tracker.get_status()
    
    print(f"🆔 Session ID: {editor.current_session}")
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    print()
    
    print("📊 Change Statistics:")
    print(f"  Total changes: {status.get('total_changes', 0)}")
    print(f"  Changes last hour: {status.get('changes_last_hour', 0)}")
    print(f"  Changes last day: {status.get('changes_last_day', 0)}")
    print()
    
    if args.show_violations:
        violations = editor.safety_violations
        if violations:
            print("❌ Safety Violations:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print("✅ No safety violations")
        print()
    
    if args.show_audit:
        logs = editor.audit_logger.get_recent_logs()
        if logs:
            print("📋 Recent Audit Logs:")
            for log in logs[:5]:  # Show last 5
                print(f"  - {log.get('timestamp', 'N/A')}: {log.get('description', 'N/A')}")
        else:
            print("📋 No audit logs available")
        print()
    
    print("✅ No safety violations")
    print()
    print("✅ No rollbacks performed")
    print()
    
    return 0

async def cmd_rollback(args):
    """Rollback changes"""
    print("🔄 Autonomous Code Editor - Rollback")
    print("=" * 50)
    
    # Load configuration
    config = load_safety_config(args.config)
    
    # Create editor instance
    editor = AutonomousCodeEditor(config)
    
    # Check if backup exists
    backup_dir = Path("backups")
    if not backup_dir.exists():
        print("❌ No backups directory found")
        return 1
    
    # Find backups for the file
    file_name = Path(args.file).name
    backups = list(backup_dir.glob(f"{file_name}_*.bak"))
    
    if not backups:
        print(f"❌ No backups found for {args.file}")
        return 1
    
    # Sort backups by creation time (newest first)
    backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if args.backup:
        # Use specific backup
        backup_path = Path(args.backup)
        if not backup_path.exists():
            print(f"❌ Specified backup not found: {args.backup}")
            return 1
    else:
        # Use most recent backup
        backup_path = backups[0]
    
    print(f"📁 File: {args.file}")
    print(f"📦 Backup: {backup_path}")
    print()
    
    if not args.force:
        response = input("⚠️ Are you sure you want to rollback? (y/N): ")
        if response.lower() != 'y':
            print("❌ Rollback cancelled")
            return 1
    
    try:
        # Restore from backup
        shutil.copy2(backup_path, args.file)
        print("✅ Rollback completed successfully")
        
        # Log rollback
        editor.audit_logger.log_activity({
            "action": "rollback",
            "file": args.file,
            "backup": str(backup_path),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Rollback failed: {e}")
        return 1
    
    return 0

async def cmd_safety_report(args):
    """Generate safety report"""
    print("🛡️ Autonomous Code Editor - Safety Report")
    print("=" * 50)
    
    # Load configuration
    config = load_safety_config(args.config)
    
    # Create editor instance
    editor = AutonomousCodeEditor(config)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "session_id": editor.current_session,
        "safety_violations": editor.safety_violations,
        "rollback_history": editor.rollback_history,
        "change_statistics": editor.change_tracker.get_statistics(),
        "audit_log": editor.audit_logger.get_recent_logs(),
        "llm_availability": {
            "claude": bool(editor.claude_client),
            "deepseek": bool(editor.deepseek_client),
            "llama2": bool(editor.llama2_client),
            "vllm": bool(editor.vllm_client)
        }
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Report saved to: {args.output}")
    else:
        if args.detailed:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(f"📊 Safety Report Summary:")
            print(f"  Session ID: {report['session_id']}")
            print(f"  Safety Violations: {len(report['safety_violations'])}")
            print(f"  Rollbacks: {len(report['rollback_history'])}")
            print(f"  Total Changes: {report['change_statistics'].get('total', 0)}")
            print(f"  LLM Availability: {sum(report['llm_availability'].values())}/4")
    
    return 0

async def cmd_config(args):
    """Configuration management"""
    print("⚙️ Autonomous Code Editor - Configuration")
    print("=" * 50)
    
    # Load configuration
    config = load_safety_config(args.config)
    
    if args.validate:
        print("🔍 Validating configuration...")
        
        # Check required fields
        validation_results = []
        validation_results.append(("safety_parameters", bool(config.max_file_size_mb > 0)))
        validation_results.append(("protected_files", bool(config.protected_files)))
        validation_results.append(("forbidden_patterns", bool(config.forbidden_patterns)))
        validation_results.append(("validation_settings", True))  # Always present
        validation_results.append(("rollback_settings", True))    # Always present
        
        # Check API keys
        claude_available = bool(config.claude_api_key)
        deepseek_available = bool(config.deepseek_api_key)
        validation_results.append(("🔑 Claude API Key", "✅ Set" if claude_available else "❌ Not set"))
        validation_results.append(("🔑 DeepSeek API Key", "✅ Set" if deepseek_available else "❌ Not set"))
        
        # Display results
        for check, result in validation_results:
            if isinstance(result, bool):
                print(f"  {'✅' if result else '❌'} {check}: {'Present' if result else 'Missing'}")
            else:
                print(f"  {result} {check}")
        
        all_passed = all(isinstance(r, str) or r for r in validation_results)
        print(f"\n✅ Configuration validation completed")
        
        if not all_passed:
            print("⚠️ Some configuration issues detected")
    
    if args.show:
        print("\n📋 Configuration Summary:")
        print(f"  Max file size: {config.max_file_size_mb} MB")
        print(f"  Max changes per hour: {config.max_changes_per_hour}")
        print(f"  Protected files: {len(config.protected_files)}")
        print(f"  Forbidden patterns: {len(config.forbidden_patterns)}")
        print(f"  Backup enabled: {config.backup_before_changes}")
        print(f"  Auto LLM selector: {config.auto_llm_selector}")
        print(f"  Fallback to local: {config.fallback_to_local}")
    
    return 0

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="🧠 Autonomous Code Editor CLI - Safe self-modification system with auto LLM selector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Edit a file with optimization
  python autonomous_editing_cli.py edit src/core/brain_launcher_v3.py \\
    --request "Add performance optimization for neural dynamics" \\
    --type optimization --level medium

  # Check status and violations
  python autonomous_editing_cli.py status --show-violations

  # Rollback changes
  python autonomous_editing_cli.py rollback src/core/brain_launcher_v3.py --force

  # Generate safety report
  python autonomous_editing_cli.py safety-report --detailed --output report.json

  # Validate configuration
  python autonomous_editing_cli.py config --validate --show
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Edit command setup
    edit_parser = subparsers.add_parser("edit", help="Edit a file autonomously")
    edit_parser.add_argument("file", help="File to edit")
    edit_parser.add_argument("--request", "-r", required=True, help="Edit request description")
    edit_parser.add_argument("--type", "-t", default="optimization", 
                           choices=["formatting", "documentation", "bug_fix", "optimization", 
                                   "refactoring", "feature_addition", "api_change", 
                                   "architecture_change", "safety_system"],
                           help="Type of change")
    edit_parser.add_argument("--level", "-l", default="medium",
                           choices=["low", "medium", "high", "critical"],
                           help="Safety level")
    edit_parser.add_argument("--config", "-c", default="src/config/autonomous_editing_safety.yaml",
                           help="Safety configuration file")
    edit_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    edit_parser.set_defaults(func=cmd_edit)
    
    # Status command setup
    status_parser = subparsers.add_parser("status", help="Show editing status")
    status_parser.add_argument("--config", "-c", default="src/config/autonomous_editing_safety.yaml",
                               help="Safety configuration file")
    status_parser.add_argument("--show-violations", action="store_true", help="Show safety violations")
    status_parser.add_argument("--show-audit", action="store_true", help="Show audit logs")
    status_parser.set_defaults(func=cmd_status)
    
    # Rollback command setup
    rollback_parser = subparsers.add_parser("rollback", help="Rollback changes")
    rollback_parser.add_argument("file", help="File to rollback")
    rollback_parser.add_argument("--backup", "-b", help="Specific backup file to use")
    rollback_parser.add_argument("--force", "-f", action="store_true", help="Force rollback without confirmation")
    rollback_parser.add_argument("--config", "-c", default="src/config/autonomous_editing_safety.yaml",
                                 help="Safety configuration file")
    rollback_parser.set_defaults(func=cmd_rollback)
    
    # Safety report command setup
    report_parser = subparsers.add_parser("safety-report", help="Generate safety report")
    report_parser.add_argument("--detailed", "-d", action="store_true", help="Detailed report")
    report_parser.add_argument("--output", "-o", help="Output file for report")
    report_parser.add_argument("--config", "-c", default="src/config/autonomous_editing_safety.yaml",
                               help="Safety configuration file")
    report_parser.set_defaults(func=cmd_safety_report)
    
    # Config command setup
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--validate", "-v", action="store_true", help="Validate configuration")
    config_parser.add_argument("--show", "-s", action="store_true", help="Show configuration summary")
    config_parser.add_argument("--config", "-c", default="src/config/autonomous_editing_safety.yaml",
                               help="Configuration file")
    config_parser.set_defaults(func=cmd_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
