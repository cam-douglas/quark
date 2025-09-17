#!/usr/bin/env python3
"""Simplified Emergency Shutdown System for Quark

This is a simplified version that won't freeze.
Human control is ALWAYS maintained.

Integration: Not simulator-integrated; governance, rules, and roadmap processing.
Rationale: Consumed by state system to guide behavior; no direct simulator hooks.
"""

import time
import json
from pathlib import Path

class EmergencyLevel:
    """Emergency severity levels."""
    NONE = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

class QuarkState:
    """Quark's operational states."""
    ACTIVE = "ACTIVE"
    SLEEPING = "SLEEPING"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"

class EmergencyEvent:
    """Emergency event record."""
    def __init__(self, timestamp, level, source, description, error_code):
        self.timestamp = timestamp
        self.level = level
        self.source = source
        self.description = description
        self.error_code = error_code

class EmergencyShutdownSystem:
    """Simplified emergency shutdown system - no freezing."""

    def __init__(self):
        self.state = QuarkState.ACTIVE
        self.emergency_level = EmergencyLevel.NONE
        self.emergency_events = []
        self.sleep_reason = ""
        self.sleep_timestamp = 0.0
        self.shutdown_trigger_code = None # Stores the specific error code
        self.error_is_resolved = False # Flag to track resolution

        # Emergency log file
        self.emergency_log_file = Path("management/emergency/emergency_log.json")
        self.emergency_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Load previous state
        self.load_emergency_state()

        print("🚨 Simple Emergency Shutdown System initialized")
        print("🚨 Human control is ALWAYS maintained")

    def load_emergency_state(self):
        """Load previous emergency state from file."""
        try:
            if self.emergency_log_file.exists():
                with open(self.emergency_log_file, 'r') as f:
                    data = json.load(f)
                    self.state = data.get('state', 'ACTIVE')
                    self.emergency_level = data.get('emergency_level', 0)
                    self.sleep_reason = data.get('sleep_reason', '')
                    self.sleep_timestamp = data.get('sleep_timestamp', 0.0)
                    self.shutdown_trigger_code = data.get('shutdown_trigger_code', None)

                    if self.state == QuarkState.SLEEPING:
                        print(f"🚨 Quark is in SLEEPING state due to: {self.shutdown_trigger_code}")
                        print("🚨 The error must be resolved before wakeup is allowed.")
        except Exception as e:
            print(f"Failed to load emergency state: {e}")

    def save_emergency_state(self):
        """Save current emergency state to file."""
        try:
            data = {
                'state': self.state,
                'emergency_level': self.emergency_level,
                'sleep_reason': self.sleep_reason,
                'sleep_timestamp': self.sleep_timestamp,
                'shutdown_trigger_code': self.shutdown_trigger_code,
                'last_updated': time.time()
            }
            with open(self.emergency_log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save emergency state: {e}")

    def trigger_emergency_shutdown(self, source, description, level):
        """Trigger emergency shutdown sequence."""
        if self.state == QuarkState.EMERGENCY_SHUTDOWN:
            return

        print("🚨 EMERGENCY SHUTDOWN TRIGGERED")
        print(f"🚨 Source: {source}")
        print(f"🚨 Description: {description}")

        self.shutdown_trigger_code = source # Save the specific error code
        self.error_is_resolved = False # Reset resolution flag on new error

        # Record emergency event
        event = EmergencyEvent(
            timestamp=time.time(),
            level=level,
            source=source,
            description=description,
            error_code=f"EMG_{source}_{int(time.time())}"
        )
        self.emergency_events.append(event)

        # Update emergency level
        if level > self.emergency_level:
            self.emergency_level = level

        # Execute shutdown sequence
        self.execute_emergency_shutdown(description)

        # Save state
        self.save_emergency_state()

    def execute_emergency_shutdown(self, reason):
        """Execute the emergency shutdown sequence."""
        try:
            print("🚨 Executing emergency shutdown sequence...")

            # Put Quark to sleep
            self.put_quark_to_sleep(reason)

            # For this simplified model, we'll go to SLEEPING first,
            # allowing for wakeup. A more complex system might go
            # straight to EMERGENCY_SHUTDOWN.

            print("🚨 Emergency shutdown sequence completed")
            print("🚨 Quark is now in a safe SLEEPING state.")
            print("🚨 Human intervention is required to restore operation.")

        except Exception as e:
            print(f"Error in emergency shutdown sequence: {e}")

    def put_quark_to_sleep(self, reason):
        """Put Quark into safe sleep state."""
        print("😴 Putting Quark to sleep for safety...")

        self.state = QuarkState.SLEEPING
        self.sleep_reason = reason
        self.sleep_timestamp = time.time()

        print("😴 Quark is now sleeping")
        print("🚨 To wake Quark, use command: WAKEUP QUARK")
        print("🚨 Error must be resolved before wakeup is allowed")

    def wakeup_quark(self, human_command):
        """Wake up Quark from sleep state."""
        if self.state != QuarkState.SLEEPING:
            print(f"Quark is not sleeping (current state: {self.state}). Cannot wake up.")
            return False

        if not self.error_is_resolved:
            print(f"❌ WAKEUP FAILED: The critical error '{self.shutdown_trigger_code}' has not been resolved.")
            print("🚨 Please resolve the issue and use the 'resolve' command before attempting wakeup.")
            return False

        # Verify human command
        if human_command.strip().upper() != "WAKEUP QUARK":
            print("Invalid wakeup command")
            return False

        # Perform wakeup sequence
        print("🌅 Waking up Quark...")

        try:
            # Reset emergency state
            self.reset_emergency_state()

            print("🌅 Quark is now awake and operational")
            return True

        except Exception as e:
            print(f"Error during wakeup: {e}")
            print("🚨 Wakeup failed - Quark remains sleeping")
            return False

    def resolve_error(self, trigger_code):
        """Mark a specific error as resolved."""
        if self.state != QuarkState.SLEEPING:
            print(f"Quark is not sleeping (current state: {self.state}). No errors to resolve.")
            return

        if trigger_code.upper() == self.shutdown_trigger_code:
            self.error_is_resolved = True
            print(f"✅ Error '{self.shutdown_trigger_code}' has been marked as resolved.")
            print("You may now attempt to wake Quark.")
        else:
            print(f"❌ Resolution failed. The active error is '{self.shutdown_trigger_code}', not '{trigger_code}'.")

    def reset_emergency_state(self):
        """Reset emergency state."""
        self.state = QuarkState.ACTIVE
        self.emergency_level = EmergencyLevel.NONE
        self.sleep_reason = ""
        self.sleep_timestamp = 0.0
        self.shutdown_trigger_code = None
        self.error_is_resolved = False

        # Clear emergency events
        self.emergency_events.clear()

        # Save state
        self.save_emergency_state()

    def get_status(self):
        """Get current emergency system status."""
        return {
            'state': self.state,
            'emergency_level': self.emergency_level,
            'sleep_reason': self.sleep_reason,
            'sleep_timestamp': self.sleep_timestamp,
            'shutdown_trigger_code': self.shutdown_trigger_code,
            'error_is_resolved': self.error_is_resolved,
            'emergency_events_count': len(self.emergency_events)
        }

    def shutdown(self):
        """Graceful shutdown of emergency system."""
        print("🔄 Shutting down emergency system...")
        print("🔄 Emergency system shutdown complete")

def main():
    """Main function for simple emergency shutdown system."""
    print("🚨 Simple Emergency Shutdown System - Safety First!")
    print("🚨 Human control is ALWAYS maintained")
    print("🚨 Quark will sleep if critical errors occur")
    print("🚨 Use 'WAKEUP QUARK' command to restore operation")

    # Initialize emergency system
    emergency_system = EmergencyShutdownSystem()

    try:
        print("✅ Emergency system initialized successfully")
        print("🚨 Use the emergency control interface for full functionality")
        print("🚨 Command: python management/emergency/emergency_control_simple.py")

        # Simple status check
        status = emergency_system.get_status()
        print(f"📊 Current status: {status['state']}")

    except Exception as e:
        print(f"🚨 Emergency system error: {e}")

if __name__ == "__main__":
    main()
