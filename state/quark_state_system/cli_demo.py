"""Simple CLI demo printing roadmap-aware QUARK recommendations.

Usage:
    python -m state.quark_state_system.cli_demo [context]
"""
import sys
from .quark_recommendations import QuarkRecommendationsEngine


def main():
    ctx = sys.argv[1] if len(sys.argv) > 1 else "general"
    engine = QuarkRecommendationsEngine()
    print(engine.provide_intelligent_guidance(f"recommend {ctx}"))


if __name__ == "__main__":
    main()
