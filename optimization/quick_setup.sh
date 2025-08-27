#!/bin/bash
# Quick Setup for Cursor Optimization

echo "🚀 Setting up Cursor optimization..."

# Ensure we're in the right directory
cd /Users/camdouglas/quark

# Create alias for integrated optimization + pruning
echo '
# Quark Integrated Optimization & Pruning Aliases
alias quark-optimize="python brain_modules/integrated_optimization_pruning.py --optimize-only"
alias quark-prune="python brain_modules/integrated_optimization_pruning.py --workflow"
alias quark-status="python brain_modules/integrated_optimization_pruning.py --status"
alias quark-health="python optimization/system_optimizer.py status"
' >> ~/.zshrc

echo "✅ Aliases added to ~/.zshrc"
echo "🔄 Restart your terminal or run: source ~/.zshrc"
echo ""
echo "📋 Available commands:"
echo "   quark-optimize  - Run system optimization only"
echo "   quark-prune     - Run pruning with automatic optimization"
echo "   quark-status    - Check integrated system status"
echo "   quark-health    - Quick system health check"
echo ""
echo "🎯 To run integrated workflow: quark-prune"
