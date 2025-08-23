#!/usr/bin/env python3
"""Visual demo: Proto-cortex LayerSheet homeostatic convergence.

Run this script locally to open a Matplotlib window showing the mean
activity of the `LayerSheet` over 1 000 steps.  You should see the trace
approach the target firing rate (≈0.1), confirming the homeostatic
plasticity rule.

Usage
-----
python testing/visualizations/layer_sheet_demo.py
"""

import numpy as np
from testing.visualizations.visual_utils import plot_series, save_fig, live_series

# Make sure import works when launched from project root
from brain_architecture.neural_core.proto_cortex.layer_sheet import LayerSheet


def main():
    sheet = LayerSheet(n=100)
    steps = 1000
    history = np.zeros(steps)
    
    # Import live streaming if enabled
    try:
        from testing.visualizations.visual_utils import live_series
        use_live = True
        print("🎥 Live streaming enabled - open testing/visualizations/live_dashboard.html")
    except ImportError:
        use_live = False
        print("📊 Static mode - saving final plot only")

    for t in range(steps):
        sheet.step()
        history[t] = sheet.mean_activity()
        
        # Stream live data if enabled
        if use_live:
            live_series("layer_sheet_activity", history[t])

    # Create and save final plot
    fig = plot_series("LayerSheet Convergence", history, target=0.1)
    save_fig(fig, "layer_sheet_live")
    print("✅ Live dashboard updated - check your browser!")


if __name__ == "__main__":
    main()
