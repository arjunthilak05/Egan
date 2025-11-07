"""
Quantum Circuit Architectures for Entanglement-Regularized QGAN

Implements various parameterized quantum circuits (PQCs) that serve as
the quantum generator component.
"""

import pennylane as qml
import numpy as np
from typing import List, Tuple, Callable


class QuantumCircuit:
    """Base class for parameterized quantum circuits."""

    def __init__(self, n_qubits: int, n_layers: int, dev_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev_name, wires=n_qubits)

    def get_num_params(self) -> int:
        """Return total number of parameters in the circuit."""
        raise NotImplementedError

    def circuit(self, params: np.ndarray, latent: np.ndarray) -> List[float]:
        """Execute the quantum circuit and return measurements."""
        raise NotImplementedError


class HardwareEfficientCircuit(QuantumCircuit):
    """
    Hardware-efficient ansatz with rotation layers and entangling gates.

    Architecture:
    - Input encoding: Amplitude or angle encoding of latent vector
    - Variational layers: RY rotations + circular entangling CNOTs
    - Designed for NISQ devices
    """

    def get_num_params(self) -> int:
        # 3 rotations per qubit per layer
        return 3 * self.n_qubits * self.n_layers

    def circuit(self, params: np.ndarray, latent: np.ndarray):
        """
        Args:
            params: Trainable parameters (shape: [n_layers, n_qubits, 3])
            latent: Classical latent vector (encoded into quantum state)
        """
        params = params.reshape(self.n_layers, self.n_qubits, 3)

        # Encode latent vector using angle encoding
        for i in range(min(self.n_qubits, len(latent))):
            qml.RY(latent[i], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(params[layer, i, 0], wires=i)
                qml.RY(params[layer, i, 1], wires=i)
                qml.RZ(params[layer, i, 2], wires=i)

            # Entangling layer (circular CNOTs)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        # Measure in computational basis
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]


class StronglyEntanglingCircuit(QuantumCircuit):
    """
    Strongly entangling ansatz from PennyLane.

    Features:
    - Random rotation layers
    - All-to-all entangling structure
    - Maximum entanglement capacity
    """

    def get_num_params(self) -> int:
        # StronglyEntanglingLayers uses 3 * n_qubits params per layer
        return 3 * self.n_qubits * self.n_layers

    def circuit(self, params: np.ndarray, latent: np.ndarray):
        params = params.reshape(self.n_layers, self.n_qubits, 3)

        # Encode latent
        for i in range(min(self.n_qubits, len(latent))):
            qml.RY(latent[i], wires=i)

        # Strongly entangling layers
        qml.StronglyEntanglingLayers(params, wires=range(self.n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]


class IQPStyleCircuit(QuantumCircuit):
    """
    IQP (Instantaneous Quantum Polynomial) inspired circuit.

    Features:
    - Hadamard layers for superposition
    - Diagonal ZZ entangling gates
    - Classically hard to simulate
    """

    def get_num_params(self) -> int:
        # 1 param per qubit per layer + entangling params
        n_entangling = self.n_qubits * (self.n_qubits - 1) // 2
        return self.n_layers * (self.n_qubits + n_entangling)

    def circuit(self, params: np.ndarray, latent: np.ndarray):
        param_idx = 0

        # Encode latent with Hadamard + phase
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            if i < len(latent):
                qml.RZ(latent[i], wires=i)

        # IQP layers
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for i in range(self.n_qubits):
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1

            # All-to-all ZZ interactions
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.IsingZZ(params[param_idx], wires=[i, j])
                    param_idx += 1

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]


class SimplifiedTwoLocalCircuit(QuantumCircuit):
    """
    Simplified two-local circuit (common in VQE/QAOA).

    Features:
    - Single-qubit RY rotations
    - Nearest-neighbor CNOTs
    - Balance between expressiveness and efficiency
    """

    def get_num_params(self) -> int:
        return 2 * self.n_qubits * self.n_layers

    def circuit(self, params: np.ndarray, latent: np.ndarray):
        params = params.reshape(self.n_layers, 2 * self.n_qubits)

        # Encode latent
        for i in range(min(self.n_qubits, len(latent))):
            qml.RY(latent[i], wires=i)

        for layer in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(params[layer, i], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Second rotation
            for i in range(self.n_qubits):
                qml.RY(params[layer, self.n_qubits + i], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]


def create_qnode(circuit: QuantumCircuit, interface: str = "torch") -> Callable:
    """
    Create a QNode (quantum function) compatible with autodiff.

    Args:
        circuit: QuantumCircuit instance
        interface: 'torch', 'tensorflow', 'jax', or 'numpy'

    Returns:
        QNode function that can be used in gradient-based optimization
    """
    @qml.qnode(circuit.dev, interface=interface, diff_method="parameter-shift")
    def qnode(params, latent):
        return circuit.circuit(params, latent)

    return qnode


# Factory function
def get_quantum_circuit(
    circuit_type: str,
    n_qubits: int,
    n_layers: int,
    dev_name: str = "default.qubit"
) -> QuantumCircuit:
    """
    Factory function to create quantum circuits.

    Args:
        circuit_type: 'hardware_efficient', 'strongly_entangling',
                     'iqp_style', or 'simplified_two_local'
        n_qubits: Number of qubits
        n_layers: Number of circuit layers
        dev_name: PennyLane device name

    Returns:
        QuantumCircuit instance
    """
    circuits = {
        'hardware_efficient': HardwareEfficientCircuit,
        'strongly_entangling': StronglyEntanglingCircuit,
        'iqp_style': IQPStyleCircuit,
        'simplified_two_local': SimplifiedTwoLocalCircuit,
    }

    if circuit_type not in circuits:
        raise ValueError(f"Unknown circuit type: {circuit_type}. "
                        f"Available: {list(circuits.keys())}")

    return circuits[circuit_type](n_qubits, n_layers, dev_name)
