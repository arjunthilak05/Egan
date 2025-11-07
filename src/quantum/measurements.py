"""
Quantum Measurement Utilities

Helper functions for measuring quantum states and extracting classical data
from quantum circuits.
"""

import pennylane as qml
import torch
import numpy as np
from typing import List, Tuple


def measure_expectation_values(
    qnode,
    params: torch.Tensor,
    latent: torch.Tensor,
    observables: str = "pauli_z"
) -> torch.Tensor:
    """
    Measure expectation values of observables on each qubit.

    Args:
        qnode: PennyLane QNode
        params: Circuit parameters
        latent: Latent vector
        observables: 'pauli_z', 'pauli_x', 'pauli_y', or 'computational_basis'

    Returns:
        Expectation values [n_qubits]
    """
    return qnode(params, latent)


def sample_computational_basis(
    qnode,
    params: torch.Tensor,
    latent: torch.Tensor,
    n_shots: int = 1000
) -> torch.Tensor:
    """
    Sample from computational basis measurement.

    Args:
        qnode: PennyLane QNode
        params: Circuit parameters
        latent: Latent vector
        n_shots: Number of measurement shots

    Returns:
        Samples [n_shots, n_qubits]
    """
    @qml.qnode(qnode.device, interface="torch")
    def sampling_circuit(p, z):
        qnode.func(p, z)
        return qml.sample()

    samples = []
    for _ in range(n_shots):
        sample = sampling_circuit(params, latent)
        samples.append(sample)

    return torch.stack(samples)


def expectation_to_image(expectations: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert quantum expectation values to image pixels.

    Maps [-1, 1] (Pauli-Z expectation) to [0, 1] (pixel intensity).

    Args:
        expectations: Expectation values from quantum circuit
        image_size: Target image dimensions (height, width)

    Returns:
        Image tensor [channels, height, width]
    """
    # Map [-1, 1] to [0, 1]
    pixels = (expectations + 1.0) / 2.0

    # Reshape to image dimensions
    n_pixels = image_size[0] * image_size[1]

    if len(pixels) < n_pixels:
        # Pad if not enough qubits
        padding = torch.zeros(n_pixels - len(pixels))
        pixels = torch.cat([pixels, padding])
    elif len(pixels) > n_pixels:
        # Truncate if too many qubits
        pixels = pixels[:n_pixels]

    # Reshape to image
    image = pixels.reshape(1, image_size[0], image_size[1])

    return image


def quantum_samples_to_distribution(
    samples: torch.Tensor,
    n_bins: int = 256
) -> torch.Tensor:
    """
    Convert quantum samples to probability distribution.

    Useful for visualizing generated distributions from quantum circuits.

    Args:
        samples: Measurement samples [n_shots, n_qubits]
        n_bins: Number of histogram bins

    Returns:
        Probability distribution [n_bins]
    """
    # Convert bitstrings to real values in [0, 1]
    n_qubits = samples.shape[1]
    powers = 2.0 ** torch.arange(n_qubits - 1, -1, -1, dtype=torch.float32)
    values = torch.sum(samples.float() * powers, dim=1) / (2 ** n_qubits)

    # Create histogram
    hist = torch.histc(values, bins=n_bins, min=0.0, max=1.0)

    # Normalize to probability
    prob_dist = hist / torch.sum(hist)

    return prob_dist


def postprocess_quantum_output(
    measurements: torch.Tensor,
    output_dim: int,
    method: str = "linear"
) -> torch.Tensor:
    """
    Post-process quantum measurements into desired classical output format.

    Args:
        measurements: Raw quantum measurements [n_qubits]
        output_dim: Desired output dimension
        method: 'linear', 'repeat', or 'truncate'

    Returns:
        Processed output [output_dim]
    """
    n_measurements = len(measurements)

    if method == "linear":
        # Linear interpolation/extrapolation
        if n_measurements == output_dim:
            return measurements

        # Create interpolation indices
        indices = torch.linspace(0, n_measurements - 1, output_dim)
        indices_floor = indices.long()
        indices_ceil = torch.clamp(indices_floor + 1, max=n_measurements - 1)

        # Interpolation weights
        weights = indices - indices_floor.float()

        # Interpolate
        output = (1 - weights) * measurements[indices_floor] + weights * measurements[indices_ceil]

        return output

    elif method == "repeat":
        # Repeat measurements to match output dimension
        repeats = output_dim // n_measurements + 1
        repeated = measurements.repeat(repeats)
        return repeated[:output_dim]

    elif method == "truncate":
        # Truncate or pad
        if n_measurements >= output_dim:
            return measurements[:output_dim]
        else:
            padding = torch.zeros(output_dim - n_measurements)
            return torch.cat([measurements, padding])

    else:
        raise ValueError(f"Unknown method: {method}")


def batch_quantum_forward(
    qnode,
    params: torch.Tensor,
    latent_batch: torch.Tensor
) -> torch.Tensor:
    """
    Process a batch of latent vectors through quantum circuit.

    Args:
        qnode: PennyLane QNode
        params: Circuit parameters (shared across batch)
        latent_batch: Batch of latent vectors [batch_size, latent_dim]

    Returns:
        Batch of measurements [batch_size, n_qubits]
    """
    batch_size = latent_batch.shape[0]
    outputs = []

    for i in range(batch_size):
        output = qnode(params, latent_batch[i])
        outputs.append(torch.tensor(output))

    return torch.stack(outputs)
