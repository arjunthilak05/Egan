"""
Configuration management for experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class QuantumConfig:
    """Quantum circuit configuration."""
    n_qubits: int = 8
    n_layers: int = 2
    circuit_type: str = "hardware_efficient"
    entropy_method: str = "pennylane"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    latent_dim: int = 4
    hidden_dim: int = 128
    use_classical_preprocess: bool = True
    use_classical_postprocess: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    n_epochs: int = 100
    batch_size: int = 64
    lr_generator: float = 0.001
    lr_discriminator: float = 0.001
    n_disc_steps: int = 1
    adversarial_loss: str = "bce"


@dataclass
class EntropyRegularizationConfig:
    """Entropy regularization configuration."""
    alpha: float = 1.0  # Entropy weight
    beta: float = 0.1   # Diversity weight
    use_scheduler: bool = False
    scheduler_strategy: str = "linear_increase"
    final_alpha: float = 10.0


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "default_experiment"
    model_type: str = "entropy_qgan"  # 'entropy_qgan', 'standard_qgan', 'classical_gan'
    dataset: str = "toy"  # 'toy', 'mnist', 'fashion_mnist'
    device: str = "cpu"
    seed: int = 42
    save_dir: str = "./results"

    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    entropy: EntropyRegularizationConfig = field(default_factory=EntropyRegularizationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'dataset': self.dataset,
            'device': self.device,
            'seed': self.seed,
            'save_dir': self.save_dir,
            'quantum': self.quantum.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'entropy': self.entropy.__dict__
        }

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        config = cls(
            name=data['name'],
            model_type=data['model_type'],
            dataset=data['dataset'],
            device=data['device'],
            seed=data['seed'],
            save_dir=data['save_dir']
        )

        # Update nested configs
        for key, value in data['quantum'].items():
            setattr(config.quantum, key, value)

        for key, value in data['model'].items():
            setattr(config.model, key, value)

        for key, value in data['training'].items():
            setattr(config.training, key, value)

        for key, value in data['entropy'].items():
            setattr(config.entropy, key, value)

        return config


# Preset configurations

def get_toy_experiment_config(alpha: float = 1.0) -> ExperimentConfig:
    """Get configuration for toy distribution experiment."""
    config = ExperimentConfig(
        name=f"toy_alpha{alpha}",
        model_type="entropy_qgan",
        dataset="toy"
    )

    config.model.latent_dim = 4
    config.quantum.n_qubits = 6
    config.quantum.n_layers = 2
    config.training.n_epochs = 200
    config.training.batch_size = 64
    config.entropy.alpha = alpha

    return config


def get_mnist_experiment_config(alpha: float = 1.0, image_size: int = 8) -> ExperimentConfig:
    """Get configuration for MNIST experiment."""
    config = ExperimentConfig(
        name=f"mnist_alpha{alpha}_size{image_size}",
        model_type="entropy_qgan",
        dataset="mnist"
    )

    output_dim = image_size * image_size

    config.model.latent_dim = 8
    config.model.hidden_dim = 128
    config.quantum.n_qubits = 10
    config.quantum.n_layers = 3
    config.training.n_epochs = 100
    config.training.batch_size = 32
    config.entropy.alpha = alpha

    return config
