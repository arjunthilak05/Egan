"""Quantum module for Entanglement-Regularized QGAN."""

from .circuits import (
    QuantumCircuit,
    HardwareEfficientCircuit,
    StronglyEntanglingCircuit,
    IQPStyleCircuit,
    SimplifiedTwoLocalCircuit,
    get_quantum_circuit,
    create_qnode
)

from .entropy import (
    von_neumann_entropy_numerical,
    von_neumann_entropy_pennylane,
    EntanglementEntropyCalculator,
    maximum_entropy,
    entanglement_entropy_gradient_analytical
)

from .measurements import (
    measure_expectation_values,
    sample_computational_basis,
    expectation_to_image,
    quantum_samples_to_distribution,
    postprocess_quantum_output,
    batch_quantum_forward
)

__all__ = [
    # Circuits
    'QuantumCircuit',
    'HardwareEfficientCircuit',
    'StronglyEntanglingCircuit',
    'IQPStyleCircuit',
    'SimplifiedTwoLocalCircuit',
    'get_quantum_circuit',
    'create_qnode',
    # Entropy
    'von_neumann_entropy_numerical',
    'von_neumann_entropy_pennylane',
    'EntanglementEntropyCalculator',
    'maximum_entropy',
    'entanglement_entropy_gradient_analytical',
    # Measurements
    'measure_expectation_values',
    'sample_computational_basis',
    'expectation_to_image',
    'quantum_samples_to_distribution',
    'postprocess_quantum_output',
    'batch_quantum_forward',
]
