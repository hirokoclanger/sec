#!/usr/bin/env python3

import pennylane as qml
from pennylane import numpy as np

##############################################################################
# 1. Define a Toy Finance Problem (4 assets)
##############################################################################
# Expected returns (mu) for 4 assets:
mu = np.array([0.10, 0.20, 0.15, 0.05])  # example returns
# Covariance matrix (sigma):
Sigma = np.array([
    [0.05, 0.01, 0.00, 0.00],
    [0.01, 0.06, 0.02, 0.01],
    [0.00, 0.02, 0.05, 0.02],
    [0.00, 0.01, 0.02, 0.04]
])
# Risk aversion parameter:
lmbda = 1.0

# Construct Q = lambda * Sigma - diag(mu)
Q = lmbda * Sigma - np.diag(mu)

n_assets = len(mu)
n_qubits = n_assets

##############################################################################
# 2. Build the Cost Hamiltonian H_C
##############################################################################
# For a QUBO form x^T Q x, each x_i corresponds to qubit i
# We'll interpret Q_ij as a coefficient for qubits i and j.
# Construct a Pennylane Hamiltonian from Q.

coeffs = []
ops = []

for i in range(n_qubits):
    for j in range(i, n_qubits):
        q_ij = Q[i, j]
        if np.abs(q_ij) > 1e-9:  # non-zero coefficient
            # For i == j, the term is q_ii * x_i^2 => q_ii * (1/2)(I - Z_i)
            # For i != j, the term is q_ij * x_i x_j => q_ij * (1/4)(I - Z_i - Z_j + Z_i Z_j)
            if i == j:
                # Single-qubit Z
                coeffs.append(-0.5 * q_ij)  # -0.5 q_ii for Z_i
                ops.append(qml.PauliZ(i))
                # also add 0.5 q_ii as a constant shift => which doesn't affect QAOA circuit
            else:
                # Two-qubit term
                coeffs.append(0.25 * q_ij)  # 0.25 q_ij for Z_i Z_j
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
                # plus other single-qubit terms from expansion, but those can be consolidated

H_C = qml.Hamiltonian(coeffs, ops)

##############################################################################
# 3. Define the Mixer Hamiltonian H_M
##############################################################################
# Standard mixer: sum of X_i over all qubits

H_M = qml.Hamiltonian(
    [1.0]*n_qubits, 
    [qml.PauliX(i) for i in range(n_qubits)]
)

##############################################################################
# 4. Set up QAOA Circuit in PennyLane
##############################################################################
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="numpy")
def qaoa_circuit(params):
    """
    params is shaped (p, 2) => each layer has [gamma, beta]
    We'll do p layers of the standard QAOA evolution:
      - e^{-i gamma H_C} e^{-i beta H_M}
    """
    p = len(params)

    # Start in an equal superposition
    for i in range(n_qubits):
        qml.Hadamard(i)

    # QAOA alternation
    for layer in range(p):
        gamma, beta = params[layer]

        # Cost Hamiltonian
        qml.ApproxTimeEvolution(H_C, gamma, 1)

        # Mixer Hamiltonian
        qml.ApproxTimeEvolution(H_M, beta, 1)

    # Return the expectation value of the cost Hamiltonian
    return qml.expval(H_C)


##############################################################################
# 5. Classical Optimization of QAOA Parameters
##############################################################################
def cost_function(params):
    # We want to MINIMIZE the expectation value of H_C
    return qaoa_circuit(params)

# Number of QAOA layers
p = 2
# Initialize parameters randomly
np.random.seed(42)
init_params = 0.01 * np.random.randn(p, 2)

opt = qml.GradientDescentOptimizer(stepsize=0.05)
max_iterations = 100

params = init_params
for it in range(max_iterations):
    params = opt.step(cost_function, params)
    if (it+1) % 10 == 0:
        current_cost = cost_function(params)
        print(f"Iteration {it+1:3d} | Cost = {current_cost:.5f}")

print(f"Optimal parameters:\n{params}")
print(f"Final cost value: {cost_function(params):.5f}")

##############################################################################
# 6. Interpreting the Result
##############################################################################
# We measure the circuit at the final parameters many times to get probable bitstrings
@qml.qnode(dev, interface="numpy")
def qaoa_sampling_circuit(params):
    # same structure
    p = len(params)
    for i in range(n_qubits):
        qml.Hadamard(i)
    for layer in range(p):
        gamma, beta = params[layer]
        qml.ApproxTimeEvolution(H_C, gamma, 1)
        qml.ApproxTimeEvolution(H_M, beta, 1)
    return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits)]

n_samples = 1000
samples = np.array([qaoa_sampling_circuit(params) for _ in range(n_samples)])
# samples.shape => (n_samples, n_qubits), each entry is +1 or -1 from PauliZ measurement

# Convert Z=+1 => x_i=0, Z=-1 => x_i=1
bitstrings = 0.5*(1 - samples)
# Count frequencies
unique, counts = np.unique(bitstrings, axis=0, return_counts=True)

print("\nSampled bitstrings and frequencies:")
for bits, cnt in zip(unique, counts):
    print(f"{bits.astype(int)} : {cnt} occurrences")

# The bitstring(s) with highest frequency might be the best solution (lowest cost).