"""
Visualization Utilities for Paper Figures

Generate publication-ready plots for:
- Training curves
- Sample distributions
- Mode coverage heatmaps
- Entropy evolution
- Comparison plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves (losses and entropy).

    Args:
        history: Dictionary with 'g_loss', 'd_loss', 'entropy', 'alpha'
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    epochs = np.arange(1, len(history['g_loss']) + 1)

    # Generator & Discriminator loss
    ax = axes[0, 0]
    ax.plot(epochs, history['g_loss'], label='Generator', linewidth=2)
    ax.plot(epochs, history['d_loss'], label='Discriminator', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Adversarial Losses')
    ax.legend()
    ax.grid(alpha=0.3)

    # Entanglement entropy
    ax = axes[0, 1]
    ax.plot(epochs, history['entropy'], color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy S(ρ_A)')
    ax.set_title('Mean Entanglement Entropy')
    ax.grid(alpha=0.3)

    # Alpha (regularization weight)
    ax = axes[1, 0]
    ax.plot(epochs, history['alpha'], color='orange', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('α')
    ax.set_title('Entropy Regularization Weight')
    ax.grid(alpha=0.3)

    # Loss ratio
    ax = axes[1, 1]
    ratio = np.array(history['g_loss']) / (np.array(history['d_loss']) + 1e-6)
    ax.plot(epochs, ratio, color='purple', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Balance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('G Loss / D Loss')
    ax.set_title('Training Balance')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_2d_distribution(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    true_modes: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Generated vs Real Distribution",
    show: bool = True
):
    """
    Plot 2D distribution comparison (for toy datasets).

    Args:
        real_samples: Real samples [n_samples, 2]
        generated_samples: Generated samples [n_samples, 2]
        true_modes: True mode centers [n_modes, 2]
        save_path: Path to save figure
        title: Plot title
        show: Whether to display
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Real samples
    ax = axes[0]
    ax.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, s=10, label='Real')
    if true_modes is not None:
        ax.scatter(true_modes[:, 0], true_modes[:, 1], c='red', marker='x',
                  s=200, linewidths=3, label='True Modes')
    ax.set_title('Real Samples')
    ax.legend()
    ax.set_aspect('equal')

    # Generated samples
    ax = axes[1]
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1],
              alpha=0.5, s=10, c='orange', label='Generated')
    if true_modes is not None:
        ax.scatter(true_modes[:, 0], true_modes[:, 1], c='red', marker='x',
                  s=200, linewidths=3, label='True Modes')
    ax.set_title('Generated Samples')
    ax.legend()
    ax.set_aspect('equal')

    # Overlay
    ax = axes[2]
    ax.scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.3, s=10,
              label='Real', c='blue')
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.3, s=10,
              label='Generated', c='orange')
    if true_modes is not None:
        ax.scatter(true_modes[:, 0], true_modes[:, 1], c='red', marker='x',
                  s=200, linewidths=3, label='True Modes')
    ax.set_title('Overlay')
    ax.legend()
    ax.set_aspect('equal')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_mode_coverage_comparison(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Bar plot comparing mode coverage across different methods.

    Args:
        results: Dictionary mapping method names to coverage scores
        save_path: Path to save figure
        show: Whether to display
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = list(results.keys())
    coverages = list(results.values())

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(methods, coverages, color=colors[:len(methods)])

    ax.set_ylabel('Mode Coverage (%)', fontsize=12)
    ax.set_title('Mode Coverage Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Coverage')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=10)

    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved mode coverage comparison to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_entropy_distribution(
    entropies: np.ndarray,
    max_entropy: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "Entanglement Entropy Distribution",
    show: bool = True
):
    """
    Plot distribution of entanglement entropies.

    Args:
        entropies: Entropy values [n_samples]
        max_entropy: Theoretical maximum entropy
        save_path: Path to save figure
        title: Plot title
        show: Whether to display
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    ax.hist(entropies, bins=30, alpha=0.7, color='green', edgecolor='black')
    if max_entropy is not None:
        ax.axvline(x=max_entropy, color='red', linestyle='--',
                  linewidth=2, label=f'Max Entropy: {max_entropy:.2f}')
    ax.axvline(x=entropies.mean(), color='blue', linestyle='--',
              linewidth=2, label=f'Mean: {entropies.mean():.2f}')
    ax.set_xlabel('Entanglement Entropy S(ρ_A)')
    ax.set_ylabel('Count')
    ax.set_title('Entropy Histogram')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot
    ax = axes[1]
    ax.boxplot(entropies, vert=True)
    if max_entropy is not None:
        ax.axhline(y=max_entropy, color='red', linestyle='--',
                  linewidth=2, label=f'Max: {max_entropy:.2f}')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('Entropy Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved entropy distribution to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_ablation_study(
    alphas: List[float],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot ablation study results (effect of α on various metrics).

    Args:
        alphas: List of alpha values tested
        metrics: Dictionary mapping metric names to lists of values
        save_path: Path to save figure
        show: Whether to display
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(alphas, values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('α (Entropy Weight)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Effect of α on {metric_name}')
        ax.grid(alpha=0.3)

    plt.suptitle('Ablation Study: Entropy Regularization Weight', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved ablation study to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_image_grid(
    images: torch.Tensor,
    n_rows: int = 8,
    n_cols: int = 8,
    save_path: Optional[str] = None,
    title: str = "Generated Samples",
    show: bool = True
):
    """
    Plot grid of generated images.

    Args:
        images: Image tensor [n_images, channels, height, width]
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        save_path: Path to save figure
        title: Plot title
        show: Whether to display
    """
    n_images = min(len(images), n_rows * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))

    for i, ax in enumerate(axes.flat):
        if i < n_images:
            img = images[i].cpu().detach()

            # Handle different image formats
            if img.shape[0] == 1:  # Grayscale
                ax.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
            else:  # RGB
                img = img.permute(1, 2, 0)
                ax.imshow(img)

        ax.axis('off')

    plt.suptitle(title, fontsize=14, y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved image grid to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_entropy_diversity_correlation(
    entropies: np.ndarray,
    diversities: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Scatter plot showing correlation between entropy and diversity.

    THIS IS A KEY FIGURE FOR OUR PAPER: demonstrating that higher entropy
    correlates with higher sample diversity.

    Args:
        entropies: Entanglement entropies [n_samples]
        diversities: Diversity scores [n_samples]
        save_path: Path to save figure
        show: Whether to display
    """
    from scipy.stats import pearsonr

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(entropies, diversities, alpha=0.5, s=20)

    # Fit line
    z = np.polyfit(entropies, diversities, 1)
    p = np.poly1d(z)
    x_line = np.linspace(entropies.min(), entropies.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Linear Fit')

    # Compute correlation
    corr, pval = pearsonr(entropies, diversities)
    ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {pval:.2e}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Entanglement Entropy S(ρ_A)', fontsize=12)
    ax.set_ylabel('Sample Diversity', fontsize=12)
    ax.set_title('Entropy-Diversity Correlation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved entropy-diversity correlation to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> str:
    """
    Create LaTeX-formatted comparison table for paper.

    Args:
        results: Nested dict {method_name: {metric_name: value}}
        save_path: Path to save LaTeX file

    Returns:
        LaTeX table string
    """
    methods = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    # Start table
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{l" + "c" * len(methods) + "}\n"
    latex += "\\hline\n"

    # Header
    latex += "Metric & " + " & ".join(methods) + " \\\\\n"
    latex += "\\hline\n"

    # Rows
    for metric in metrics:
        row = [metric.replace('_', ' ').title()]
        values = [results[method][metric] for method in methods]

        # Bold the best value
        best_idx = np.argmax(values) if 'coverage' in metric or 'diversity' in metric else np.argmin(values)

        for i, (method, value) in enumerate(zip(methods, values)):
            if i == best_idx:
                row.append(f"\\textbf{{{value:.3f}}}")
            else:
                row.append(f"{value:.3f}")

        latex += " & ".join(row) + " \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Performance comparison across different methods.}\n"
    latex += "\\label{tab:comparison}\n"
    latex += "\\end{table}"

    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"Saved LaTeX table to {save_path}")

    return latex
