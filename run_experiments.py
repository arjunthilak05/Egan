#!/usr/bin/env python3
"""
Main Experiment Runner

Run all experiments for the paper:
1. Toy distributions (mode coverage)
2. MNIST (image generation)
3. Ablation study (effect of alpha)
4. Comparison with baselines
"""

import argparse
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiments.toy_distributions import run_toy_experiment, run_comparison_experiment


def main():
    parser = argparse.ArgumentParser(description="Run Entanglement-Regularized QGAN Experiments")

    parser.add_argument(
        '--experiment',
        type=str,
        default='toy',
        choices=['toy', 'toy_comparison', 'mnist', 'ablation', 'all'],
        help='Which experiment to run'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='entropy_qgan',
        choices=['entropy_qgan', 'standard_qgan', 'classical_gan'],
        help='Model type'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Entropy regularization weight'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--n-modes',
        type=int,
        default=8,
        help='Number of modes for toy distributions'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    print("="*80)
    print("ENTANGLEMENT-REGULARIZED QGAN EXPERIMENTS")
    print("="*80)
    print(f"Experiment: {args.experiment}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("="*80)
    print()

    # Run experiments
    if args.experiment == 'toy':
        print("Running single toy distribution experiment...")
        results = run_toy_experiment(
            model_type=args.model,
            n_modes=args.n_modes,
            n_epochs=args.epochs,
            alpha=args.alpha,
            device=args.device
        )

        print("\n✅ Toy experiment completed!")
        print(f"Mode Coverage: {results['mode_coverage']:.2%}")
        print(f"Diversity: {results['diversity']:.4f}")
        print(f"Mean Entropy: {results['mean_entropy']:.4f}")

    elif args.experiment == 'toy_comparison':
        print("Running comprehensive toy distribution comparison...")
        all_results = run_comparison_experiment(
            n_modes=args.n_modes,
            n_epochs=args.epochs,
            device=args.device
        )

        print("\n✅ Comparison experiment completed!")
        print("See results in ./results/comparison/")

    elif args.experiment == 'mnist':
        print("MNIST experiment not yet implemented in runner.")
        print("Please run experiments/mnist_experiment.py directly.")

    elif args.experiment == 'ablation':
        print("Running ablation study...")
        print("Testing different alpha values: [0.0, 0.5, 1.0, 2.0, 5.0]")

        alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
        results = {}

        for alpha in alphas:
            print(f"\nTesting alpha = {alpha}...")
            result = run_toy_experiment(
                model_type='entropy_qgan',
                n_modes=args.n_modes,
                n_epochs=args.epochs,
                alpha=alpha,
                device=args.device
            )
            results[alpha] = result

        print("\n✅ Ablation study completed!")
        print("\nSummary:")
        print(f"{'Alpha':<10} {'Coverage':<15} {'Diversity':<15} {'Entropy':<15}")
        print("-"*55)
        for alpha, res in results.items():
            print(f"{alpha:<10} {res['mode_coverage']:<15.4f} {res['diversity']:<15.4f} {res['mean_entropy']:<15.4f}")

    elif args.experiment == 'all':
        print("Running ALL experiments (this will take a while)...")

        print("\n" + "="*80)
        print("1. TOY DISTRIBUTION COMPARISON")
        print("="*80)
        run_comparison_experiment(
            n_modes=8,
            n_epochs=200,
            device=args.device
        )

        print("\n" + "="*80)
        print("2. ABLATION STUDY")
        print("="*80)
        alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
        for alpha in alphas:
            run_toy_experiment(
                model_type='entropy_qgan',
                n_modes=8,
                n_epochs=200,
                alpha=alpha,
                device=args.device
            )

        print("\n✅ ALL experiments completed!")
        print("Results saved to ./results/")

    else:
        print(f"Unknown experiment: {args.experiment}")
        return 1

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
