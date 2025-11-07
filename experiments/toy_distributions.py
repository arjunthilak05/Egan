"""
Toy Distribution Experiments (Gaussian Mixtures)

This is THE KEY EXPERIMENT for demonstrating our contribution:
- Visualize mode coverage improvements
- Show that entropy regularization prevents mode collapse
- Compare against baselines
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src.models import create_qgan_model
from src.evaluation import (
    mode_coverage_score,
    sample_diversity_score,
    plot_training_curves,
    plot_2d_distribution,
    plot_mode_coverage_comparison,
    plot_entropy_distribution,
    MetricsTracker
)


def generate_gaussian_mixture(
    n_samples: int,
    n_modes: int = 8,
    mode_distance: float = 3.0,
    mode_std: float = 0.3,
    seed: int = 42
) -> tuple:
    """
    Generate 2D Gaussian mixture dataset.

    Args:
        n_samples: Number of samples to generate
        n_modes: Number of modes (Gaussians)
        mode_distance: Distance of modes from origin
        mode_std: Standard deviation of each Gaussian
        seed: Random seed

    Returns:
        (samples, mode_centers)
    """
    np.random.seed(seed)

    # Create mode centers in a circle
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    mode_centers = np.stack([
        mode_distance * np.cos(angles),
        mode_distance * np.sin(angles)
    ], axis=1)

    # Generate samples
    samples = []
    samples_per_mode = n_samples // n_modes

    for center in mode_centers:
        mode_samples = np.random.randn(samples_per_mode, 2) * mode_std + center
        samples.append(mode_samples)

    samples = np.concatenate(samples, axis=0)

    # Shuffle
    np.random.shuffle(samples)

    return samples.astype(np.float32), mode_centers.astype(np.float32)


def run_toy_experiment(
    model_type: str,
    n_modes: int = 8,
    n_epochs: int = 200,
    batch_size: int = 64,
    latent_dim: int = 4,
    n_qubits: int = 6,
    alpha: float = 1.0,
    device: str = "cpu",
    save_dir: str = "../results"
):
    """
    Run toy distribution experiment for a single model.

    Args:
        model_type: 'entropy_qgan', 'standard_qgan', or 'classical_gan'
        n_modes: Number of Gaussian modes
        n_epochs: Training epochs
        batch_size: Batch size
        latent_dim: Latent dimension
        n_qubits: Number of qubits (for quantum models)
        alpha: Entropy regularization weight
        device: 'cpu' or 'cuda'
        save_dir: Directory to save results

    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Running Toy Experiment: {model_type}")
    print(f"n_modes={n_modes}, alpha={alpha}, epochs={n_epochs}")
    print(f"{'='*60}\n")

    # Create save directory
    save_dir = Path(save_dir) / f"toy_{model_type}_modes{n_modes}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    real_samples, true_modes = generate_gaussian_mixture(
        n_samples=10000,
        n_modes=n_modes,
        mode_distance=3.0,
        mode_std=0.3
    )

    # Create dataloader
    dataset = TensorDataset(torch.from_numpy(real_samples))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = create_qgan_model(
        model_type=model_type,
        latent_dim=latent_dim,
        output_dim=2,  # 2D data
        n_qubits=n_qubits,
        n_layers=2,
        circuit_type="hardware_efficient",
        alpha=alpha,
        lr_g=0.001,
        lr_d=0.001,
        device=device
    )

    print(f"Model created: {type(model).__name__}")
    print(f"Generator: {type(model.generator).__name__}")
    print(f"Discriminator: {type(model.discriminator).__name__}\n")

    # Train
    print("Starting training...")
    history = model.train(
        dataloader=dataloader,
        n_epochs=n_epochs,
        n_disc_steps=1,
        verbose=True
    )

    # Generate samples
    print("\nGenerating samples...")
    generated_samples = model.generate(n_samples=1000).cpu().numpy()

    # Compute entropies
    print("Computing entanglement entropies...")
    entropies = model.compute_entropy_distribution(n_samples=1000).cpu().numpy()

    # Evaluate
    print("\nEvaluating performance...")

    # Mode coverage
    coverage, covered_modes = mode_coverage_score(
        generated_samples,
        true_modes,
        threshold=0.5
    )

    # Diversity
    diversity = sample_diversity_score(generated_samples, k=5)

    print(f"\nResults:")
    print(f"  Mode Coverage: {coverage:.2%}")
    print(f"  Sample Diversity: {diversity:.4f}")
    print(f"  Mean Entropy: {entropies.mean():.4f}")
    print(f"  Std Entropy: {entropies.std():.4f}")

    # Visualizations
    print("\nCreating visualizations...")

    # Training curves
    plot_training_curves(
        history,
        save_path=str(save_dir / "training_curves.png"),
        show=False
    )

    # Distribution comparison
    plot_2d_distribution(
        real_samples[:1000],
        generated_samples,
        true_modes,
        save_path=str(save_dir / "distribution.png"),
        title=f"{model_type}: Mode Coverage = {coverage:.2%}",
        show=False
    )

    # Entropy distribution
    plot_entropy_distribution(
        entropies,
        max_entropy=np.log(2 ** (n_qubits // 2)),  # Max entropy for subsystem
        save_path=str(save_dir / "entropy_dist.png"),
        title=f"Entanglement Entropy Distribution ({model_type})",
        show=False
    )

    print(f"\nResults saved to {save_dir}")

    # Return results
    results = {
        'model_type': model_type,
        'mode_coverage': coverage,
        'diversity': diversity,
        'mean_entropy': entropies.mean(),
        'std_entropy': entropies.std(),
        'history': history,
        'generated_samples': generated_samples,
        'entropies': entropies
    }

    return results


def run_comparison_experiment(
    n_modes: int = 8,
    n_epochs: int = 200,
    device: str = "cpu"
):
    """
    Run full comparison experiment across all methods.

    This generates THE KEY FIGURE for our paper: mode coverage comparison.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON EXPERIMENT")
    print("="*80)

    models_to_test = [
        ('classical_gan', 0.0),
        ('standard_qgan', 0.0),
        ('entropy_qgan', 0.5),
        ('entropy_qgan', 1.0),
        ('entropy_qgan', 2.0),
    ]

    all_results = {}

    for model_type, alpha in models_to_test:
        label = f"{model_type}_alpha{alpha}" if model_type == 'entropy_qgan' else model_type

        try:
            results = run_toy_experiment(
                model_type=model_type,
                n_modes=n_modes,
                n_epochs=n_epochs,
                alpha=alpha,
                device=device
            )

            all_results[label] = results

        except Exception as e:
            print(f"\nError running {label}: {e}")
            continue

    # Create comparison plots
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)

    save_dir = Path("../results") / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Mode coverage comparison
    coverage_dict = {
        label.replace('_', ' ').title(): results['mode_coverage']
        for label, results in all_results.items()
    }

    plot_mode_coverage_comparison(
        coverage_dict,
        save_path=str(save_dir / "mode_coverage_comparison.png"),
        show=False
    )

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Method':<30} {'Mode Coverage':<15} {'Diversity':<15} {'Mean Entropy':<15}")
    print("-"*80)

    for label, results in all_results.items():
        print(f"{label:<30} "
              f"{results['mode_coverage']:.4f}        "
              f"{results['diversity']:.4f}        "
              f"{results['mean_entropy']:.4f}")

    print("="*80)

    return all_results


if __name__ == "__main__":
    # Simple test: single experiment
    # results = run_toy_experiment(
    #     model_type='entropy_qgan',
    #     n_modes=8,
    #     n_epochs=50,  # Short for testing
    #     alpha=1.0
    # )

    # Full comparison
    all_results = run_comparison_experiment(
        n_modes=8,
        n_epochs=200,
        device='cpu'
    )

    print("\n✅ Toy distribution experiments completed!")
    print(f"Results saved to ../results/")
