#!/usr/bin/env python3
"""Sync Module - Main interface for QUARK state synchronization.

Provides unified interface to state synchronization functionality.

Integration: State sync interface for QuarkDriver and AutonomousAgent.
Rationale: Clean API abstraction over state synchronization modules.
"""

from .state_synchronizer import QuarkStateSynchronizer

# Export main interface for backward compatibility
__all__ = ['QuarkStateSynchronizer']

# Main function for CLI compatibility
def main():
    """Main function for state synchronization."""
    synchronizer = QuarkStateSynchronizer()
    results = synchronizer.synchronize_all_state_files()

    print("🔄 QUARK STATE SYNCHRONIZATION")
    print("=" * 40)
    print(f"Files processed: {results['files_processed']}")
    print(f"Files updated: {results['files_updated']}")

    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  ❌ {error}")

    if results['sync_log']:
        print("\nSync log:")
        for entry in results['sync_log']:
            print(f"  {entry}")

if __name__ == "__main__":
    main()
