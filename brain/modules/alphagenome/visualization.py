"""
AlphaGenome visualization module - provides visualization tools for genomic predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_components(data: Union[Dict[str, np.ndarray], 'PredictionOutputs'],
                   title: str = "AlphaGenome Predictions",
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (14, 10),
                   show_legend: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple AlphaGenome prediction tracks.
    
    Args:
        data: Dictionary of track names to arrays or PredictionOutputs object
        title: Main title for the plot
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        show_legend: Whether to show legend
        
    Returns:
        fig, axes: Matplotlib figure and axes
    """

    # Convert PredictionOutputs to dict if needed
    if hasattr(data, 'to_dict'):
        plot_data = {}
        interval = getattr(data, 'interval', None)

        # Extract available tracks
        for track_name in ['rna_seq', 'atac', 'histone_h3k27ac', 'histone_h3k27me3',
                          'histone_h3k9me3', 'dnase', 'cage', 'conservation',
                          'regulatory_score']:
            track_data = getattr(data, track_name, None)
            if track_data is not None:
                plot_data[track_name] = track_data
    else:
        plot_data = data
        interval = None

    if not plot_data:
        logger.warning("No data to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=16)
        return fig, ax

    # Create figure with subplots for each track
    n_tracks = len(plot_data)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_tracks + 1, 1, height_ratios=[0.5] + [1] * n_tracks,
                  hspace=0.3)

    # Add title and interval info
    title_ax = fig.add_subplot(gs[0])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, title, ha='center', va='center',
                  fontsize=18, fontweight='bold')
    if interval:
        title_ax.text(0.5, 0.1, str(interval), ha='center', va='center',
                      fontsize=12, style='italic')

    axes = []

    # Define track properties
    track_props = {
        'rna_seq': {'color': 'darkblue', 'label': 'RNA-seq (Gene Expression)',
                    'ylabel': 'Expression'},
        'atac': {'color': 'darkgreen', 'label': 'ATAC-seq (Chromatin Accessibility)',
                 'ylabel': 'Accessibility'},
        'histone_h3k27ac': {'color': 'orange', 'label': 'H3K27ac (Active Enhancer)',
                           'ylabel': 'Signal'},
        'histone_h3k27me3': {'color': 'purple', 'label': 'H3K27me3 (Repressive)',
                            'ylabel': 'Signal'},
        'histone_h3k9me3': {'color': 'darkred', 'label': 'H3K9me3 (Heterochromatin)',
                           'ylabel': 'Signal'},
        'dnase': {'color': 'teal', 'label': 'DNase Hypersensitivity',
                  'ylabel': 'Signal'},
        'cage': {'color': 'brown', 'label': 'CAGE (TSS Activity)',
                'ylabel': 'Signal'},
        'conservation': {'color': 'gray', 'label': 'Evolutionary Conservation',
                        'ylabel': 'Score'},
        'regulatory_score': {'color': 'red', 'label': 'Regulatory Potential',
                            'ylabel': 'Score'}
    }

    # Plot each track
    for i, (track_name, track_data) in enumerate(plot_data.items()):
        ax = fig.add_subplot(gs[i + 1])
        axes.append(ax)

        # Get track properties
        props = track_props.get(track_name,
                               {'color': 'black', 'label': track_name, 'ylabel': 'Value'})

        # Create position array
        positions = np.arange(len(track_data))

        # Plot based on track type
        if track_name in ['rna_seq', 'cage']:
            # Bar plot for expression data
            ax.bar(positions, track_data, color=props['color'], alpha=0.7, width=1.0)
        elif track_name == 'conservation':
            # Filled area plot for conservation
            ax.fill_between(positions, 0, track_data, color=props['color'], alpha=0.5)
            ax.plot(positions, track_data, color=props['color'], linewidth=1)
        else:
            # Line plot with fill for other tracks
            ax.fill_between(positions, 0, track_data, color=props['color'], alpha=0.3)
            ax.plot(positions, track_data, color=props['color'], linewidth=1.5)

        # Styling
        ax.set_ylabel(props['ylabel'], fontsize=10)
        ax.set_xlim(0, len(track_data) - 1)
        ax.set_ylim(0, max(track_data.max() * 1.1, 0.1))  # 10% padding

        # Add track name on the right
        ax.text(1.02, 0.5, props['label'], transform=ax.transAxes,
                fontsize=10, va='center', ha='left')

        # Only show x-axis label on bottom plot
        if i < n_tracks - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Position (bins)', fontsize=12)

        # Add grid
        ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    return fig, axes


def plot_variant_effects(reference_data: Dict[str, np.ndarray],
                        variant_data: Dict[str, np.ndarray],
                        variant_info: Optional[Dict[str, Any]] = None,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
    """Plot variant effects by comparing reference and alternate predictions.
    
    Args:
        reference_data: Reference allele predictions
        variant_data: Variant allele predictions  
        variant_info: Optional variant information
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        fig, axes: Matplotlib figure and axes
    """

    # Find common tracks
    common_tracks = set(reference_data.keys()) & set(variant_data.keys())
    n_tracks = len(common_tracks)

    if n_tracks == 0:
        logger.warning("No common tracks between reference and variant")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No common tracks to compare', ha='center', va='center')
        return fig, ax

    # Create figure
    fig, axes = plt.subplots(n_tracks, 2, figsize=figsize, sharey='row')
    if n_tracks == 1:
        axes = axes.reshape(1, -1)

    # Title
    title = "Variant Effect Analysis"
    if variant_info:
        variant_str = variant_info.get('variant', 'Unknown variant')
        title += f"\n{variant_str}"
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot each track
    for i, track_name in enumerate(sorted(common_tracks)):
        ref_data = reference_data[track_name]
        var_data = variant_data[track_name]

        # Reference plot
        axes[i, 0].plot(ref_data, 'b-', alpha=0.7, label='Reference')
        axes[i, 0].fill_between(range(len(ref_data)), 0, ref_data, alpha=0.3)
        axes[i, 0].set_ylabel(track_name.replace('_', ' ').title())
        axes[i, 0].set_title('Reference Allele' if i == 0 else '')
        axes[i, 0].grid(True, alpha=0.3)

        # Variant plot
        axes[i, 1].plot(var_data, 'r-', alpha=0.7, label='Variant')
        axes[i, 1].fill_between(range(len(var_data)), 0, var_data, alpha=0.3)
        axes[i, 1].set_title('Variant Allele' if i == 0 else '')
        axes[i, 1].grid(True, alpha=0.3)

        # Calculate and show difference
        diff = var_data - ref_data
        max_diff_idx = np.argmax(np.abs(diff))
        max_diff_val = diff[max_diff_idx]

        # Add difference annotation
        axes[i, 1].text(0.95, 0.95, f'Max Δ: {max_diff_val:.2f}',
                       transform=axes[i, 1].transAxes,
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Add x-labels to bottom plots only
    for ax in axes[-1, :]:
        ax.set_xlabel('Position (bins)')

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved variant plot to {save_path}")

    return fig, axes


def plot_regulatory_elements(regulatory_scores: np.ndarray,
                           threshold: float = 0.7,
                           min_width: int = 5,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """Identify and plot potential regulatory elements.
    
    Args:
        regulatory_scores: Array of regulatory potential scores
        threshold: Score threshold for calling elements
        min_width: Minimum width for regulatory elements (in bins)
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        fig, ax: Matplotlib figure and axis
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Plot regulatory scores
    positions = np.arange(len(regulatory_scores))
    ax.plot(positions, regulatory_scores, 'k-', linewidth=1, alpha=0.8)
    ax.fill_between(positions, 0, regulatory_scores, alpha=0.2, color='gray')

    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5,
               label=f'Threshold ({threshold})')

    # Find regulatory elements
    above_threshold = regulatory_scores > threshold

    # Find contiguous regions
    elements = []
    start = None

    for i in range(len(above_threshold)):
        if above_threshold[i] and start is None:
            start = i
        elif not above_threshold[i] and start is not None:
            if i - start >= min_width:
                elements.append((start, i))
            start = None

    # Handle last element
    if start is not None and len(above_threshold) - start >= min_width:
        elements.append((start, len(above_threshold)))

    # Highlight regulatory elements
    for i, (start, end) in enumerate(elements):
        ax.add_patch(patches.Rectangle((start, 0), end - start,
                                      regulatory_scores[start:end].max(),
                                      alpha=0.3, facecolor='orange',
                                      edgecolor='red', linewidth=2))

        # Add element label
        center = (start + end) / 2
        ax.text(center, regulatory_scores[start:end].max() + 0.05,
                f'E{i+1}', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Styling
    ax.set_xlabel('Position (bins)', fontsize=12)
    ax.set_ylabel('Regulatory Score', fontsize=12)
    ax.set_title(f'Regulatory Elements (n={len(elements)})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add summary text
    summary = f"Found {len(elements)} regulatory elements\n"
    summary += f"Threshold: {threshold}, Min width: {min_width} bins"
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved regulatory elements plot to {save_path}")

    return fig, ax
