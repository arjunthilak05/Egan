"""
Hybrid Quantum-Classical Generator

This is our main contribution: a quantum generator that can be regularized
by entanglement entropy to enforce diversity and prevent mode collapse.
"""

import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple, Optional
import numpy as np

from ..quantum import (
    get_quantum_circuit,
    create_qnode,
    EntanglementEntropyCalculator,
    postprocess_quantum_output
)


class QuantumGenerator(nn.Module):
    """
    Quantum Generator for QGAN with entanglement entropy computation.

    Architecture:
    1. Classical preprocessing (optional): latent → quantum latent
    2. Quantum circuit: quantum latent → entangled quantum state
    3. Measurement: quantum state → classical outputs
    4. Classical postprocessing (optional): quantum outputs → final samples
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        n_qubits: int,
        n_layers: int,
        circuit_type: str = "hardware_efficient",
        use_classical_preprocess: bool = True,
        use_classical_postprocess: bool = True,
        entropy_method: str = "pennylane"
    ):
        """
        Args:
            latent_dim: Dimension of input latent vector
            output_dim: Dimension of generated output
            n_qubits: Number of qubits in quantum circuit
            n_layers: Number of layers in quantum circuit
            circuit_type: Type of quantum circuit architecture
            use_classical_preprocess: Add classical layer before quantum
            use_classical_postprocess: Add classical layer after quantum
            entropy_method: Method for entropy calculation
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type

        # Create quantum circuit
        self.quantum_circuit = get_quantum_circuit(
            circuit_type, n_qubits, n_layers
        )

        # Create QNode
        self.qnode = create_qnode(self.quantum_circuit, interface="torch")

        # Initialize quantum circuit parameters
        n_params = self.quantum_circuit.get_num_params()
        self.quantum_params = nn.Parameter(
            torch.randn(n_params) * 0.1
        )

        # Classical preprocessing (latent → quantum latent)
        if use_classical_preprocess:
            self.preprocess = nn.Sequential(
                nn.Linear(latent_dim, n_qubits),
                nn.Tanh()  # Bound to [-1, 1] for quantum encoding
            )
        else:
            self.preprocess = nn.Identity()
            assert latent_dim == n_qubits, "latent_dim must equal n_qubits without preprocessing"

        # Classical postprocessing (quantum measurements → output)
        if use_classical_postprocess:
            self.postprocess = nn.Sequential(
                nn.Linear(n_qubits, output_dim),
                nn.Tanh()
            )
        else:
            self.postprocess = lambda x: postprocess_quantum_output(x, output_dim)

        # Entanglement entropy calculator
        subsystem_qubits = list(range(n_qubits // 2))
        self.entropy_calculator = EntanglementEntropyCalculator(
            n_qubits, subsystem_qubits, method=entropy_method
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generate samples from latent vectors.

        Args:
            latent: Latent vectors [batch_size, latent_dim]

        Returns:
            Generated samples [batch_size, output_dim]
        """
        batch_size = latent.shape[0]
        outputs = []

        for i in range(batch_size):
            # Preprocess latent
            quantum_latent = self.preprocess(latent[i])

            # Quantum circuit forward pass
            quantum_output = self.qnode(self.quantum_params, quantum_latent)

            # Convert to tensor if needed
            if not isinstance(quantum_output, torch.Tensor):
                quantum_output = torch.tensor(quantum_output, dtype=torch.float32)

            # Postprocess
            output = self.postprocess(quantum_output)

            outputs.append(output)

        return torch.stack(outputs)

    def compute_entanglement_entropy(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute entanglement entropy for a batch of latent vectors.

        This is THE KEY NOVELTY: we compute S(ρ_A) for the quantum state
        generated from each latent vector, which serves as a diversity measure.

        Args:
            latent: Latent vectors [batch_size, latent_dim]

        Returns:
            Entropy values [batch_size]
        """
        batch_size = latent.shape[0]
        entropies = []

        for i in range(batch_size):
            # Preprocess latent
            quantum_latent = self.preprocess(latent[i])

            # Compute entropy
            entropy = self.entropy_calculator.compute_entropy(
                self.qnode,
                self.quantum_params,
                quantum_latent
            )

            entropies.append(entropy)

        return torch.stack(entropies)

    def forward_with_entropy(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples AND compute entanglement entropy in one pass.

        Args:
            latent: Latent vectors [batch_size, latent_dim]

        Returns:
            (generated_samples, entanglement_entropies)
        """
        samples = self.forward(latent)
        entropies = self.compute_entanglement_entropy(latent)

        return samples, entropies


class HybridQuantumGenerator(nn.Module):
    """
    More sophisticated hybrid architecture with deeper classical components.

    This version uses a deeper classical network for both pre and post processing,
    suitable for more complex data like images.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        n_qubits: int,
        n_layers: int,
        hidden_dim: int = 128,
        circuit_type: str = "hardware_efficient"
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits

        # Classical encoder: latent → quantum input
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh()
        )

        # Quantum circuit
        self.quantum_circuit = get_quantum_circuit(circuit_type, n_qubits, n_layers)
        self.qnode = create_qnode(self.quantum_circuit, interface="torch")

        n_params = self.quantum_circuit.get_num_params()
        self.quantum_params = nn.Parameter(torch.randn(n_params) * 0.1)

        # Classical decoder: quantum output → final output
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

        # Entropy calculator
        subsystem_qubits = list(range(n_qubits // 2))
        self.entropy_calculator = EntanglementEntropyCalculator(
            n_qubits, subsystem_qubits, method="pennylane"
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        outputs = []

        for i in range(batch_size):
            # Encode
            quantum_input = self.encoder(latent[i])

            # Quantum processing
            quantum_output = self.qnode(self.quantum_params, quantum_input)

            if not isinstance(quantum_output, torch.Tensor):
                quantum_output = torch.tensor(quantum_output, dtype=torch.float32)

            # Decode
            output = self.decoder(quantum_output)
            outputs.append(output)

        return torch.stack(outputs)

    def compute_entanglement_entropy(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        entropies = []

        for i in range(batch_size):
            quantum_input = self.encoder(latent[i])
            entropy = self.entropy_calculator.compute_entropy(
                self.qnode,
                self.quantum_params,
                quantum_input
            )
            entropies.append(entropy)

        return torch.stack(entropies)

    def forward_with_entropy(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self.forward(latent)
        entropies = self.compute_entanglement_entropy(latent)
        return samples, entropies


class ClassicalGenerator(nn.Module):
    """
    Classical generator baseline for comparison.

    Standard MLP generator without any quantum components.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)

    def compute_entanglement_entropy(self, latent: torch.Tensor) -> torch.Tensor:
        """Return zeros since classical generators have no entanglement."""
        return torch.zeros(latent.shape[0])

    def forward_with_entropy(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self.forward(latent)
        entropies = self.compute_entanglement_entropy(latent)
        return samples, entropies
