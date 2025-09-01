#!/usr/bin/env python3
"""Simple Emergency Control Interface

Simple command-line interface for the emergency shutdown system.
No freezing, no complex monitoring.

Integration: Not simulator-integrated; governance, rules, and roadmap processing.
Rationale: Consumed by state system to guide behavior; no direct simulator hooks.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from management.emergency.emergency_shutdown_system import EmergencyShutdownSystem, QuarkState, EmergencyLevel

def print_banner():
    """Print emergency system banner."""
    print("🚨" * 40)
    print("🚨      QUARK EMERGENCY CONTROL SYSTEM      🚨")
    print("🚨      Human Control Always Maintained     🚨")
    print("🚨" * 40)
    print()

def print_status(emergency_system):
    """Print current emergency system status."""
    try:
        status = emergency_system.get_status()
        
        print("📊 CURRENT EMERGENCY SYSTEM STATUS:")
        print("=" * 40)
        print(f"Quark State: {status['state']}")
        print(f"Emergency Level: {status['emergency_level']}")
        
        if status['state'] == 'SLEEPING':
            print(f"Shutdown Trigger Code: {status['shutdown_trigger_code']}")
            print(f"Sleep Reason: {status['sleep_reason']}")
            print(f"Error Resolved: {status['error_is_resolved']}")
            print(f"Sleep Time: {time.ctime(status['sleep_timestamp'])}")
        
        print(f"Emergency Events: {status['emergency_events_count']}")
        print("=" * 40)
        
    except Exception as e:
        print(f"❌ Error getting status: {e}")

def print_help():
    """Print help information."""
    print("📖 EMERGENCY CONTROL COMMANDS:")
    print("=" * 40)
    print("status                - Show current emergency system status")
    print("trigger <ERROR_CODE>  - Trigger emergency shutdown with a specific error code")
    print("resolve <ERROR_CODE>  - Mark a specific error as resolved")
    print("wakeup                - Wake up Quark from sleep state")
    print("help                  - Show this help message")
    print("quit                  - Exit emergency control")
    print()
    print("🚨 EMERGENCY SHUTDOWN TRIGGERS:")
    print("=" * 40)
    print("SAFETY_SCORE_CRITICAL     - Safety score below 20")
    print("RESOURCE_EXHAUSTION       - System resources exhausted")
    print("CONSCIOUSNESS_ANOMALY     - Consciousness system issues")
    print("LEARNING_LOOP_DETECTED    - Runaway learning processes")
    print("TEST_SHUTDOWN             - Test emergency shutdown")
    print()

def trigger_emergency_shutdown(emergency_system, shutdown_type: str):
    """Trigger emergency shutdown for testing."""
    try:
        print(f"🚨 Triggering emergency shutdown: {shutdown_type}")
        
        if shutdown_type == "SAFETY_SCORE_CRITICAL":
            emergency_system.trigger_emergency_shutdown(
                "TEST_SAFETY_SCORE",
                "Test emergency shutdown - safety score critical",
                EmergencyLevel.CRITICAL
            )
        elif shutdown_type == "RESOURCE_EXHAUSTION":
            emergency_system.trigger_emergency_shutdown(
                "TEST_RESOURCE_EXHAUSTION",
                "Test emergency shutdown - resource exhaustion",
                EmergencyLevel.EMERGENCY
            )
        elif shutdown_type == "CONSCIOUSNESS_ANOMALY":
            emergency_system.trigger_emergency_shutdown(
                "TEST_CONSCIOUSNESS_ANOMALY",
                "Test emergency shutdown - consciousness anomaly",
                EmergencyLevel.CRITICAL
            )
        elif shutdown_type == "LEARNING_LOOP_DETECTED":
            emergency_system.trigger_emergency_shutdown(
                "TEST_LEARNING_LOOP",
                "Test emergency shutdown - learning loop detected",
                EmergencyLevel.CRITICAL
            )
        else:
            emergency_system.trigger_emergency_shutdown(
                "TEST_SHUTDOWN",
                f"Test emergency shutdown - {shutdown_type}",
                EmergencyLevel.CRITICAL
            )
        
        print("✅ Emergency shutdown triggered successfully")
        print("😴 Quark should now be in sleep state")
        print("🚨 Use 'wakeup' command to restore operation")
        
    except Exception as e:
        print(f"❌ Error triggering emergency shutdown: {e}")

def wakeup_quark(emergency_system):
    """Wake up Quark from sleep state."""
    try:
        print("🌅 Attempting to wake up Quark...")
        
        # Check if Quark is sleeping
        status = emergency_system.get_status()
        if status['state'] != QuarkState.SLEEPING:
            print("❌ Quark is not sleeping - no wakeup needed")
            return
        
        # Request wakeup command
        print("🚨 To wake up Quark, you must type: WAKEUP QUARK")
        print("🚨 This ensures human control is maintained")
        
        wakeup_command = input("Enter wakeup command: ").strip()
        
        if emergency_system.wakeup_quark(wakeup_command):
            print("✅ Quark successfully awakened!")
            print("🚀 Systems are being restored...")
        else:
            print("❌ Wakeup failed - Quark remains sleeping")
            print("🚨 Please check that the error is resolved")
            print("🚨 Then try the wakeup command again")
        
    except Exception as e:
        print(f"❌ Error during wakeup: {e}")

def main():
    """Main emergency control interface."""
    print_banner()
    
    # Initialize emergency system
    try:
        emergency_system = EmergencyShutdownSystem()
        print("✅ Emergency shutdown system initialized")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize emergency system: {e}")
        print("🚨 Emergency control unavailable")
        return
    
    # Check if Quark is already sleeping
    try:
        status = emergency_system.get_status()
        if status['state'] == QuarkState.SLEEPING:
            print("😴 Quark is currently in SLEEPING state")
            print("🚨 Use 'wakeup' command to restore operation")
            print()
    except Exception as e:
        print(f"⚠️ Warning: Could not get status: {e}")
    
    print_help()
    
    # Main command loop
    while True:
        try:
            command_input = input("🚨 EMERGENCY> ").strip().lower()
            parts = command_input.split(" ", 1)
            command = parts[0]
            
            if command == "quit" or command == "exit":
                print("🔄 Shutting down emergency control...")
                try:
                    emergency_system.shutdown()
                except:
                    pass
                break
                
            elif command == "status":
                try:
                    print_status(emergency_system)
                except Exception as e:
                    print(f"❌ Error getting status: {e}")
                
            elif command == "trigger":
                if len(parts) > 1:
                    error_code = parts[1].upper()
                    trigger_emergency_shutdown(emergency_system, error_code)
                else:
                    print("❌ Usage: trigger <ERROR_CODE>")
                
            elif command == "resolve":
                if len(parts) > 1:
                    error_code = parts[1].upper()
                    emergency_system.resolve_error(error_code)
                else:
                    print("❌ Usage: resolve <ERROR_CODE>")

            elif command == "wakeup":
                try:
                    wakeup_quark(emergency_system)
                except Exception as e:
                    print(f"❌ Error during wakeup: {e}")
                
            elif command == "help":
                print_help()
                
            elif command == "":
                continue
                
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands")
            
            print()
            
        except KeyboardInterrupt:
            print("\n🚨 Emergency control interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in emergency control: {e}")
            print("🚨 Emergency system may be compromised")
            break
    
    print("🚨 Emergency control shutdown complete")

if __name__ == "__main__":
    main()