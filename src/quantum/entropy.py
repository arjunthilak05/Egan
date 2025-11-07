"""
Entanglement Entropy Calculation and Gradient Computation

Implements von Neumann entropy S(ρ_A) = -Tr(ρ_A log ρ_A) for bipartite systems.
This is the CORE NOVELTY of our paper: using entanglement entropy as a
differentiable regularization term in the GAN loss function.
"""

import pennylane as qml
import numpy as np
import torch
from typing import List, Tuple, Optional


def von_neumann_entropy_numerical(
    state_vector: np.ndarray,
    subsystem_qubits: List[int],
    total_qubits: int,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate von Neumann entropy of a subsystem using numerical methods.

    S(ρ_A) = -Tr(ρ_A log ρ_A) = -∑_i λ_i log λ_i

    Args:
        state_vector: Full quantum state vector (2^n amplitudes)
        subsystem_qubits: List of qubit indices for subsystem A
        total_qubits: Total number of qubits in the system
        epsilon: Small constant to avoid log(0)

    Returns:
        Von Neumann entropy (in nats, use log base 2 for bits)
    """
    # Reshape state vector into tensor
    shape = [2] * total_qubits
    state_tensor = state_vector.reshape(shape)

    # Identify subsystem B (complement of A)
    all_qubits = set(range(total_qubits))
    subsystem_b = list(all_qubits - set(subsystem_qubits))

    # Compute reduced density matrix ρ_A by tracing out subsystem B
    if len(subsystem_b) > 0:
        # Trace over subsystem B indices
        axes_to_trace = tuple(subsystem_b)
        # Create conjugate for density matrix calculation
        rho_a = np.tensordot(
            state_tensor,
            state_tensor.conj(),
            axes=(axes_to_trace, axes_to_trace)
        )
        # Reshape to matrix form
        dim_a = 2 ** len(subsystem_qubits)
        rho_a = rho_a.reshape(dim_a, dim_a)
    else:
        # If subsystem A is the whole system, ρ_A = |ψ⟩⟨ψ|
        rho_a = np.outer(state_vector, state_vector.conj())

    # Compute eigenvalues of reduced density matrix
    eigenvalues = np.linalg.eigvalsh(rho_a)

    # Filter out numerical noise (negative eigenvalues, zeros)
    eigenvalues = eigenvalues[eigenvalues > epsilon]

    # Normalize (should sum to 1, but numerical errors may occur)
    eigenvalues = eigenvalues / np.sum(eigenvalues)

    # Calculate von Neumann entropy
    entropy = -np.sum(eigenvalues * np.log(eigenvalues + epsilon))

    return float(entropy)


def von_neumann_entropy_pennylane(
    qnode,
    params: torch.Tensor,
    latent: torch.Tensor,
    subsystem_wires: List[int]
) -> torch.Tensor:
    """
    Calculate von Neumann entropy using PennyLane's built-in methods.
    This version supports automatic differentiation.

    Args:
        qnode: PennyLane QNode that prepares the quantum state
        params: Circuit parameters (torch tensor)
        latent: Latent vector input (torch tensor)
        subsystem_wires: Wires (qubit indices) for subsystem A

    Returns:
        Entropy as a differentiable torch tensor
    """
    @qml.qnode(qnode.device, interface="torch", diff_method="parameter-shift")
    def entropy_qnode(p, z):
        # Prepare the quantum state
        qnode.func(p, z)
        # Calculate von Neumann entropy of subsystem
        return qml.vn_entropy(wires=subsystem_wires, log_base=np.e)

    return entropy_qnode(params, latent)


class EntanglementEntropyCalculator:
    """
    Differentiable entanglement entropy calculator for QGAN training.

    This class provides multiple methods for computing entropy:
    1. Direct PennyLane computation (fast, autodiff-friendly)
    2. State vector extraction + numerical computation (more flexible)
    3. Sampling-based estimation (NISQ-realistic)
    """

    def __init__(
        self,
        n_qubits: int,
        subsystem_qubits: Optional[List[int]] = None,
        method: str = "pennylane"
    ):
        """
        Args:
            n_qubits: Total number of qubits
            subsystem_qubits: Qubits in subsystem A (defaults to first half)
            method: 'pennylane', 'numerical', or 'sampling'
        """
        self.n_qubits = n_qubits
        self.method = method

        # Default: split system in half
        if subsystem_qubits is None:
            self.subsystem_qubits = list(range(n_qubits // 2))
        else:
            self.subsystem_qubits = subsystem_qubits

    def compute_entropy(
        self,
        qnode,
        params: torch.Tensor,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entanglement entropy with automatic differentiation support.

        Args:
            qnode: Quantum circuit as PennyLane QNode
            params: Circuit parameters
            latent: Latent vector

        Returns:
            Entropy value (differentiable)
        """
        if self.method == "pennylane":
            return self._compute_pennylane(qnode, params, latent)
        elif self.method == "numerical":
            return self._compute_numerical(qnode, params, latent)
        elif self.method == "sampling":
            return self._compute_sampling(qnode, params, latent)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _compute_pennylane(self, qnode, params, latent) -> torch.Tensor:
        """Use PennyLane's built-in entropy calculation."""
        return von_neumann_entropy_pennylane(
            qnode, params, latent, self.subsystem_qubits
        )

    def _compute_numerical(self, qnode, params, latent) -> torch.Tensor:
        """
        Extract state vector and compute entropy numerically.
        Note: This breaks the computational graph for gradients.
        """
        # Get state vector from circuit
        @qml.qnode(qnode.device, interface="torch")
        def state_qnode(p, z):
            qnode.func(p, z)
            return qml.state()

        state_vector = state_qnode(params, latent)

        # Convert to numpy for numerical computation
        state_np = state_vector.detach().cpu().numpy()

        entropy = von_neumann_entropy_numerical(
            state_np,
            self.subsystem_qubits,
            self.n_qubits
        )

        # Convert back to torch tensor
        return torch.tensor(entropy, dtype=torch.float32, requires_grad=False)

    def _compute_sampling(
        self,
        qnode,
        params,
        latent,
        n_samples: int = 1000
    ) -> torch.Tensor:
        """
        Estimate entropy using measurement samples (NISQ-realistic).
        Uses subsystem measurement statistics to approximate entropy.

        This is less accurate but represents what's achievable on real hardware.
        """
        @qml.qnode(qnode.device, interface="torch")
        def sample_qnode(p, z):
            qnode.func(p, z)
            return qml.sample(wires=self.subsystem_qubits)

        # Collect samples
        samples = []
        for _ in range(n_samples):
            sample = sample_qnode(params, latent)
            samples.append(sample)

        samples = torch.stack(samples)

        # Convert bitstrings to integers
        subsystem_size = len(self.subsystem_qubits)
        powers = 2 ** torch.arange(subsystem_size - 1, -1, -1, dtype=torch.float32)
        integers = torch.sum(samples * powers, dim=1).long()

        # Count occurrences
        counts = torch.bincount(integers, minlength=2**subsystem_size)
        probabilities = counts.float() / n_samples

        # Compute Shannon entropy (approximates von Neumann for pure states)
        epsilon = 1e-10
        probabilities = probabilities + epsilon
        entropy = -torch.sum(probabilities * torch.log(probabilities))

        return entropy

    def compute_batch_entropy(
        self,
        qnode,
        params_batch: torch.Tensor,
        latent_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy for a batch of latent vectors.

        Args:
            qnode: Quantum circuit
            params_batch: Parameters (same for all in batch, or per-sample)
            latent_batch: Batch of latent vectors [batch_size, latent_dim]

        Returns:
            Entropy values [batch_size]
        """
        batch_size = latent_batch.shape[0]
        entropies = []

        for i in range(batch_size):
            if params_batch.dim() == 1:
                # Same parameters for all samples
                params = params_batch
            else:
                # Different parameters per sample
                params = params_batch[i]

            entropy = self.compute_entropy(qnode, params, latent_batch[i])
            entropies.append(entropy)

        return torch.stack(entropies)


def maximum_entropy(n_qubits_subsystem: int) -> float:
    """
    Theoretical maximum entropy for a subsystem.

    For a pure state, max entanglement gives S_max = log(d) where d = 2^n
    is the subsystem dimension.

    Args:
        n_qubits_subsystem: Number of qubits in subsystem A

    Returns:
        Maximum possible entropy (in nats)
    """
    return np.log(2 ** n_qubits_subsystem)


def entanglement_entropy_gradient_analytical(
    qnode,
    params: torch.Tensor,
    latent: torch.Tensor,
    subsystem_wires: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both entropy and its gradient with respect to parameters.

    Uses parameter-shift rule or adjoint method for efficient gradient computation.

    Args:
        qnode: PennyLane QNode
        params: Circuit parameters (requires_grad=True)
        latent: Latent vector
        subsystem_wires: Subsystem qubit indices

    Returns:
        (entropy_value, gradient_wrt_params)
    """
    params.requires_grad_(True)

    @qml.qnode(qnode.device, interface="torch", diff_method="parameter-shift")
    def entropy_qnode(p):
        qnode.func(p, latent)
        return qml.vn_entropy(wires=subsystem_wires, log_base=np.e)

    entropy = entropy_qnode(params)
    entropy.backward()
    gradient = params.grad.clone()

    return entropy, gradient
