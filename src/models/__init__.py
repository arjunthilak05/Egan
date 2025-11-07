"""Models module for Entanglement-Regularized QGAN."""

from .generator import (
    QuantumGenerator,
    HybridQuantumGenerator,
    ClassicalGenerator
)

from .discriminator import (
    Discriminator,
    ConvDiscriminator,
    PatchDiscriminator,
    WassersteinDiscriminator,
    GradientPenaltyMixin
)

from .losses import (
    EntropyRegularizedGANLoss,
    StandardGANLoss,
    AdaptiveEntropyScheduler,
    minibatch_discrimination_loss,
    sample_repulsion_loss
)

from .qgan import (
    EntropyRegularizedQGAN,
    StandardQGAN,
    create_qgan_model
)

__all__ = [
    # Generators
    'QuantumGenerator',
    'HybridQuantumGenerator',
    'ClassicalGenerator',
    # Discriminators
    'Discriminator',
    'ConvDiscriminator',
    'PatchDiscriminator',
    'WassersteinDiscriminator',
    'GradientPenaltyMixin',
    # Losses
    'EntropyRegularizedGANLoss',
    'StandardGANLoss',
    'AdaptiveEntropyScheduler',
    'minibatch_discrimination_loss',
    'sample_repulsion_loss',
    # Models
    'EntropyRegularizedQGAN',
    'StandardQGAN',
    'create_qgan_model',
]
