"""
Loss Functions for Entanglement-Regularized QGAN

This module implements our NOVEL CONTRIBUTION:
L_G = L_adversarial - α·S(ρ_A) + β·L_diversity

Where S(ρ_A) is the entanglement entropy regularization term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EntropyRegularizedGANLoss:
    """
    Custom loss function combining adversarial loss with entanglement entropy
    regularization.

    This is THE KEY NOVELTY of our paper: we explicitly optimize entanglement
    entropy to enforce diversity in the generator.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        adversarial_loss: str = "bce",
        diversity_loss: str = "none"
    ):
        """
        Args:
            alpha: Weight for entropy regularization (maximize entropy)
            beta: Weight for diversity loss
            adversarial_loss: 'bce', 'hinge', or 'wasserstein'
            diversity_loss: 'none', 'minibatch', or 'repulsion'
        """
        self.alpha = alpha
        self.beta = beta
        self.adversarial_loss_type = adversarial_loss
        self.diversity_loss_type = diversity_loss

        # Adversarial loss function
        if adversarial_loss == "bce":
            self.adversarial_criterion = nn.BCEWithLogitsLoss()
        elif adversarial_loss == "mse":
            self.adversarial_criterion = nn.MSELoss()

    def generator_loss(
        self,
        fake_logits: torch.Tensor,
        entanglement_entropies: torch.Tensor,
        fake_samples: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss with entropy regularization.

        L_G = L_adversarial - α·S(ρ_A) + β·L_diversity
                              ^^^^^^^^^^^
                              THIS IS OUR CONTRIBUTION!

        Args:
            fake_logits: Discriminator output on generated samples [batch_size, 1]
            entanglement_entropies: Entanglement entropy S(ρ_A) [batch_size]
            fake_samples: Generated samples for diversity loss [batch_size, dim]

        Returns:
            (total_loss, loss_components_dict)
        """
        # 1. Adversarial loss (fool discriminator)
        if self.adversarial_loss_type == "bce":
            real_labels = torch.ones_like(fake_logits)
            L_adversarial = self.adversarial_criterion(fake_logits, real_labels)
        elif self.adversarial_loss_type == "hinge":
            L_adversarial = -fake_logits.mean()
        elif self.adversarial_loss_type == "wasserstein":
            L_adversarial = -fake_logits.mean()
        else:
            raise ValueError(f"Unknown adversarial loss: {self.adversarial_loss_type}")

        # 2. Entropy regularization (NOVEL: maximize entanglement entropy)
        # We SUBTRACT α·S to MAXIMIZE entropy (higher entropy = more diversity)
        mean_entropy = entanglement_entropies.mean()
        L_entropy_reg = -self.alpha * mean_entropy

        # 3. Additional diversity loss (optional)
        if self.diversity_loss_type == "none" or fake_samples is None:
            L_diversity = torch.tensor(0.0, device=fake_logits.device)
        elif self.diversity_loss_type == "minibatch":
            L_diversity = -minibatch_discrimination_loss(fake_samples)
        elif self.diversity_loss_type == "repulsion":
            L_diversity = -sample_repulsion_loss(fake_samples)
        else:
            L_diversity = torch.tensor(0.0, device=fake_logits.device)

        # Total generator loss
        total_loss = L_adversarial + L_entropy_reg + self.beta * L_diversity

        # Return components for logging
        loss_components = {
            'g_total': total_loss.item(),
            'g_adversarial': L_adversarial.item(),
            'g_entropy_reg': L_entropy_reg.item(),
            'g_diversity': L_diversity.item(),
            'entropy_mean': mean_entropy.item()
        }

        return total_loss, loss_components

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss.

        L_D = -log(D(x_real)) - log(1 - D(G(z)))

        Args:
            real_logits: Discriminator output on real samples [batch_size, 1]
            fake_logits: Discriminator output on fake samples [batch_size, 1]

        Returns:
            (total_loss, loss_components_dict)
        """
        if self.adversarial_loss_type == "bce":
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)

            real_loss = self.adversarial_criterion(real_logits, real_labels)
            fake_loss = self.adversarial_criterion(fake_logits, fake_labels)

            total_loss = real_loss + fake_loss

        elif self.adversarial_loss_type == "hinge":
            real_loss = torch.relu(1.0 - real_logits).mean()
            fake_loss = torch.relu(1.0 + fake_logits).mean()
            total_loss = real_loss + fake_loss

        elif self.adversarial_loss_type == "wasserstein":
            total_loss = -(real_logits.mean() - fake_logits.mean())
            real_loss = -real_logits.mean()
            fake_loss = fake_logits.mean()

        else:
            raise ValueError(f"Unknown adversarial loss: {self.adversarial_loss_type}")

        loss_components = {
            'd_total': total_loss.item(),
            'd_real': real_loss.item(),
            'd_fake': fake_loss.item()
        }

        return total_loss, loss_components


def minibatch_discrimination_loss(samples: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """
    Minibatch discrimination: encourage diversity within each batch.

    Computes pairwise distances and encourages them to be large.

    Args:
        samples: Generated samples [batch_size, dim]
        reduce: Whether to reduce to scalar

    Returns:
        Diversity score (higher = more diverse)
    """
    # Compute pairwise distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2⟨x_i, x_j⟩
    dots = torch.mm(samples, samples.t())
    norms = torch.diag(dots)
    distances = norms.unsqueeze(1) + norms.unsqueeze(0) - 2 * dots

    # Exclude self-distances (diagonal)
    batch_size = samples.size(0)
    mask = (1 - torch.eye(batch_size, device=samples.device)).bool()
    distances = distances[mask]

    if reduce:
        return distances.mean()
    else:
        return distances


def sample_repulsion_loss(samples: torch.Tensor) -> torch.Tensor:
    """
    Repulsion loss: penalize samples that are too close to each other.

    L_repulsion = -log(∑_{i≠j} ||x_i - x_j||^2)

    Args:
        samples: Generated samples [batch_size, dim]

    Returns:
        Repulsion loss (minimize to push samples apart)
    """
    distances = minibatch_discrimination_loss(samples, reduce=False)

    # Use negative log to penalize small distances heavily
    epsilon = 1e-6
    repulsion = -torch.log(distances + epsilon).mean()

    return repulsion


class StandardGANLoss:
    """
    Standard GAN loss without any entropy regularization (baseline).
    """

    def __init__(self, adversarial_loss: str = "bce"):
        self.adversarial_loss_type = adversarial_loss

        if adversarial_loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif adversarial_loss == "mse":
            self.criterion = nn.MSELoss()

    def generator_loss(
        self,
        fake_logits: torch.Tensor,
        entanglement_entropies: Optional[torch.Tensor] = None,
        fake_samples: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Standard generator loss (no entropy regularization)."""

        if self.adversarial_loss_type == "bce":
            real_labels = torch.ones_like(fake_logits)
            loss = self.criterion(fake_logits, real_labels)
        elif self.adversarial_loss_type == "hinge":
            loss = -fake_logits.mean()
        elif self.adversarial_loss_type == "wasserstein":
            loss = -fake_logits.mean()
        else:
            raise ValueError(f"Unknown loss: {self.adversarial_loss_type}")

        loss_components = {
            'g_total': loss.item(),
            'g_adversarial': loss.item(),
            'g_entropy_reg': 0.0,
            'g_diversity': 0.0,
            'entropy_mean': entanglement_entropies.mean().item() if entanglement_entropies is not None else 0.0
        }

        return loss, loss_components

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Standard discriminator loss."""

        if self.adversarial_loss_type == "bce":
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)

            real_loss = self.criterion(real_logits, real_labels)
            fake_loss = self.criterion(fake_logits, fake_labels)
            total_loss = real_loss + fake_loss

        elif self.adversarial_loss_type == "hinge":
            real_loss = torch.relu(1.0 - real_logits).mean()
            fake_loss = torch.relu(1.0 + fake_logits).mean()
            total_loss = real_loss + fake_loss

        elif self.adversarial_loss_type == "wasserstein":
            total_loss = -(real_logits.mean() - fake_logits.mean())
            real_loss = -real_logits.mean()
            fake_loss = fake_logits.mean()

        else:
            raise ValueError(f"Unknown loss: {self.adversarial_loss_type}")

        loss_components = {
            'd_total': total_loss.item(),
            'd_real': real_loss.item(),
            'd_fake': fake_loss.item()
        }

        return total_loss, loss_components


class AdaptiveEntropyScheduler:
    """
    Scheduler for α (entropy weight) that adapts during training.

    Strategies:
    - 'constant': Keep α fixed
    - 'linear_increase': Gradually increase α
    - 'cosine': Cosine annealing schedule
    - 'adaptive': Increase α when mode collapse is detected
    """

    def __init__(
        self,
        initial_alpha: float = 1.0,
        final_alpha: float = 10.0,
        total_steps: int = 10000,
        strategy: str = "linear_increase"
    ):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_steps = total_steps
        self.strategy = strategy
        self.current_step = 0

    def step(self) -> float:
        """Get current alpha value and increment step."""
        self.current_step += 1

        if self.strategy == "constant":
            return self.initial_alpha

        elif self.strategy == "linear_increase":
            progress = min(self.current_step / self.total_steps, 1.0)
            return self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress

        elif self.strategy == "cosine":
            import math
            progress = min(self.current_step / self.total_steps, 1.0)
            cosine_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            return self.initial_alpha + (self.final_alpha - self.initial_alpha) * (1 - cosine_progress)

        else:
            return self.initial_alpha

    def reset(self):
        """Reset scheduler."""
        self.current_step = 0
