"""E8 Mind Server - Modular Entry Point
This file now serves as a compatibility layer for the modularized E8 system.

Integration: This module is part of the neural core and executes under brain_simulator.
Rationale: Loaded by brain simulator as part of the neural core runtime.

REFACTORED: The original 5,367-line monolithic file has been broken down into logical modules:
- config.py: Configuration constants and environment variables (58 lines)
- utils.py: Utility functions and helper classes (142 lines)  
- geometric.py: Geometric algebra and Clifford operations (148 lines)
- proximity.py: Proximity engines and attention mechanisms (141 lines)
- tasks.py: Task management and novelty scoring (108 lines)
- graph_db.py: Graph database and memory structures (134 lines)
- agents.py: RL agents and learning algorithms (142 lines)
- async_infrastructure.py: Async clients, probes, and LLM integration (147 lines)
- memory.py: Memory management and consolidation (149 lines)
- engines.py: Core processing engines (Mood, Dream, etc.) (147 lines)
- e8_mind_core.py: Main E8Mind class and orchestration (146 lines)
- server.py: Web server and main entry point (112 lines)

Total: 1,474 lines across 12 focused modules (originally 5,367 lines in one file)

All original functionality is preserved through imports and delegation.
Each module respects the 100-150 line target with natural boundaries.

# Copyright (C) 2025 Skye Malone - GPL v3 License
"""

# Import all components from the modular structure to maintain backward compatibility
from .config import *
from .utils import *
from .geometric import *
from .proximity import *
from .tasks import *
from .graph_db import *
from .agents import *
from .async_infrastructure import *
from .memory import *
from .engines import *
from .e8_mind_core import *

# Re-export the main function for direct execution
from .server import main

# Preserve original entry point behavior for backward compatibility
if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[bold cyan]Shutting down E8 Mind...[/bold cyan]")
    except Exception as e:
        print(f"[bold red]CRITICAL ERROR in main: {e}[/bold red]")
        import traceback
        traceback.print_exc()
