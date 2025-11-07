"""
Complete QGAN Model with Entanglement Entropy Regularization

This module ties together:
- Quantum Generator
- Classical Discriminator
- Entropy-regularized loss
- Training loop

This is the main model for our paper.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import numpy as np

from .generator import QuantumGenerator, HybridQuantumGenerator, ClassicalGenerator
from .discriminator import Discriminator
from .losses import EntropyRegularizedGANLoss, StandardGANLoss, AdaptiveEntropyScheduler


class EntropyRegularizedQGAN:
    """
    Main QGAN model with entanglement entropy regularization.

    This is our NOVEL CONTRIBUTION: a QGAN that explicitly optimizes
    entanglement entropy to prevent mode collapse and ensure diversity.
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        latent_dim: int,
        alpha: float = 1.0,
        beta: float = 0.1,
        lr_g: float = 0.001,
        lr_d: float = 0.001,
        adversarial_loss: str = "bce",
        diversity_loss: str = "none",
        use_entropy_scheduler: bool = False,
        device: str = "cpu"
    ):
        """
        Args:
            generator: Generator network (quantum or classical)
            discriminator: Discriminator network
            latent_dim: Dimension of latent space
            alpha: Weight for entropy regularization
            beta: Weight for diversity loss
            lr_g: Generator learning rate
            lr_d: Discriminator learning rate
            adversarial_loss: Type of adversarial loss
            diversity_loss: Type of additional diversity loss
            use_entropy_scheduler: Whether to use adaptive alpha scheduling
            device: 'cpu' or 'cuda'
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_dim = latent_dim
        self.device = device

        # Loss function with entropy regularization
        self.loss_fn = EntropyRegularizedGANLoss(
            alpha=alpha,
            beta=beta,
            adversarial_loss=adversarial_loss,
            diversity_loss=diversity_loss
        )

        # Optimizers
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        # Entropy scheduler
        self.use_entropy_scheduler = use_entropy_scheduler
        if use_entropy_scheduler:
            self.entropy_scheduler = AdaptiveEntropyScheduler(
                initial_alpha=alpha,
                final_alpha=alpha * 10,
                strategy="linear_increase"
            )

        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'entropy': [],
            'alpha': []
        }

    def train_step(
        self,
        real_samples: torch.Tensor,
        n_disc_steps: int = 1
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            real_samples: Real data samples [batch_size, data_dim]
            n_disc_steps: Number of discriminator updates per generator update

        Returns:
            Dictionary of loss values
        """
        batch_size = real_samples.size(0)
        real_samples = real_samples.to(self.device)

        metrics = {}

        # ============================================
        # Train Discriminator
        # ============================================
        for _ in range(n_disc_steps):
            self.optimizer_d.zero_grad()

            # Generate fake samples
            latent = torch.randn(batch_size, self.latent_dim, device=self.device)

            with torch.no_grad():
                fake_samples = self.generator(latent)

            # Discriminator outputs
            real_logits = self.discriminator(real_samples)
            fake_logits = self.discriminator(fake_samples)

            # Discriminator loss
            d_loss, d_components = self.loss_fn.discriminator_loss(real_logits, fake_logits)

            d_loss.backward()
            self.optimizer_d.step()

            metrics.update(d_components)

        # ============================================
        # Train Generator (with entropy regularization)
        # ============================================
        self.optimizer_g.zero_grad()

        # Generate new fake samples
        latent = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Forward pass with entropy computation
        if hasattr(self.generator, 'forward_with_entropy'):
            fake_samples, entropies = self.generator.forward_with_entropy(latent)
        else:
            fake_samples = self.generator(latent)
            entropies = torch.zeros(batch_size, device=self.device)

        # Discriminator evaluation
        fake_logits = self.discriminator(fake_samples)

        # Generator loss with entropy regularization
        g_loss, g_components = self.loss_fn.generator_loss(
            fake_logits,
            entropies,
            fake_samples
        )

        g_loss.backward()
        self.optimizer_g.step()

        metrics.update(g_components)

        # Update entropy weight if using scheduler
        if self.use_entropy_scheduler:
            new_alpha = self.entropy_scheduler.step()
            self.loss_fn.alpha = new_alpha
            metrics['alpha'] = new_alpha
        else:
            metrics['alpha'] = self.loss_fn.alpha

        return metrics

    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        n_disc_steps: int = 1,
        verbose: bool = True,
        save_interval: int = 10
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            dataloader: DataLoader for real data
            n_epochs: Number of training epochs
            n_disc_steps: Discriminator steps per generator step
            verbose: Print training progress
            save_interval: Epochs between saving checkpoints

        Returns:
            Training history
        """
        for epoch in range(n_epochs):
            epoch_metrics = {
                'g_loss': [],
                'd_loss': [],
                'entropy': [],
                'alpha': []
            }

            # Progress bar
            if verbose:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
            else:
                pbar = dataloader

            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    real_samples = batch[0]
                else:
                    real_samples = batch

                # Training step
                metrics = self.train_step(real_samples, n_disc_steps)

                # Accumulate metrics
                epoch_metrics['g_loss'].append(metrics['g_total'])
                epoch_metrics['d_loss'].append(metrics['d_total'])
                epoch_metrics['entropy'].append(metrics['entropy_mean'])
                epoch_metrics['alpha'].append(metrics['alpha'])

                # Update progress bar
                if verbose:
                    pbar.set_postfix({
                        'G': f"{metrics['g_total']:.3f}",
                        'D': f"{metrics['d_total']:.3f}",
                        'S': f"{metrics['entropy_mean']:.3f}",
                        'α': f"{metrics['alpha']:.2f}"
                    })

            # Save epoch averages
            self.history['g_loss'].append(np.mean(epoch_metrics['g_loss']))
            self.history['d_loss'].append(np.mean(epoch_metrics['d_loss']))
            self.history['entropy'].append(np.mean(epoch_metrics['entropy']))
            self.history['alpha'].append(np.mean(epoch_metrics['alpha']))

            if verbose:
                print(f"Epoch {epoch+1} - G: {self.history['g_loss'][-1]:.3f}, "
                      f"D: {self.history['d_loss'][-1]:.3f}, "
                      f"Entropy: {self.history['entropy'][-1]:.3f}")

        return self.history

    def generate(self, n_samples: int = 100) -> torch.Tensor:
        """
        Generate samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Generated samples [n_samples, output_dim]
        """
        self.generator.eval()

        with torch.no_grad():
            latent = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(latent)

        self.generator.train()

        return samples

    def compute_entropy_distribution(self, n_samples: int = 100) -> torch.Tensor:
        """
        Compute entanglement entropy for a batch of random latent vectors.

        Useful for analyzing entropy distribution during/after training.

        Args:
            n_samples: Number of samples

        Returns:
            Entropy values [n_samples]
        """
        self.generator.eval()

        with torch.no_grad():
            latent = torch.randn(n_samples, self.latent_dim, device=self.device)

            if hasattr(self.generator, 'compute_entanglement_entropy'):
                entropies = self.generator.compute_entanglement_entropy(latent)
            else:
                entropies = torch.zeros(n_samples)

        self.generator.train()

        return entropies


class StandardQGAN(EntropyRegularizedQGAN):
    """
    Standard QGAN without entropy regularization (baseline).

    Identical architecture but without the entropy regularization term.
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        latent_dim: int,
        lr_g: float = 0.001,
        lr_d: float = 0.001,
        adversarial_loss: str = "bce",
        device: str = "cpu"
    ):
        # Initialize without entropy regularization
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            latent_dim=latent_dim,
            alpha=0.0,  # No entropy regularization
            beta=0.0,
            lr_g=lr_g,
            lr_d=lr_d,
            adversarial_loss=adversarial_loss,
            diversity_loss="none",
            use_entropy_scheduler=False,
            device=device
        )

        # Use standard GAN loss
        self.loss_fn = StandardGANLoss(adversarial_loss=adversarial_loss)


def create_qgan_model(
    model_type: str,
    latent_dim: int,
    output_dim: int,
    n_qubits: int = 8,
    n_layers: int = 3,
    circuit_type: str = "hardware_efficient",
    hidden_dim: int = 128,
    alpha: float = 1.0,
    device: str = "cpu",
    **kwargs
) -> EntropyRegularizedQGAN:
    """
    Factory function to create different QGAN variants.

    Args:
        model_type: 'entropy_qgan', 'standard_qgan', 'classical_gan'
        latent_dim: Latent dimension
        output_dim: Output dimension
        n_qubits: Number of qubits (for quantum models)
        n_layers: Number of quantum circuit layers
        circuit_type: Quantum circuit architecture
        hidden_dim: Hidden dimension for classical layers
        alpha: Entropy regularization weight
        device: 'cpu' or 'cuda'

    Returns:
        QGAN model instance
    """
    # Create generator
    if model_type in ['entropy_qgan', 'standard_qgan']:
        generator = QuantumGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type
        )
    elif model_type == 'classical_gan':
        generator = ClassicalGenerator(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create discriminator
    discriminator = Discriminator(
        input_dim=output_dim,
        hidden_dim=hidden_dim
    )

    # Create QGAN model
    if model_type == 'entropy_qgan':
        model = EntropyRegularizedQGAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=latent_dim,
            alpha=alpha,
            device=device,
            **kwargs
        )
    elif model_type in ['standard_qgan', 'classical_gan']:
        model = StandardQGAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=latent_dim,
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
