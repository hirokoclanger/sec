#!/usr/bin/env python3

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

##########################
# 1. Generate Synthetic Data
##########################
# For demonstration, we create a small binary classification dataset
X, y = make_classification(
    n_samples=200,      # number of data points
    n_features=4,       # total features
    n_informative=4,    # all features are "informative" in this synthetic example
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalize features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to Torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

##########################
# 2. PennyLane Device Setup
##########################
# We'll use the default qubit simulator for a small circuit
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

##########################
# 3. Define the QNode (Quantum Circuit)
##########################
# We embed classical features into the circuit, then apply trainable gates.
# The circuit output (expectation value) will be interpreted as a 'logit' for binary classification.

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    inputs: [features] – 1D tensor
    weights: Trainable parameters that define the variational circuit
    """
    # Angle embedding of features
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)

    # Trainable layers (example: a single layer of entangling + rotations)
    for i in range(n_qubits):
        qml.RZ(weights[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    for i in range(n_qubits, 2*n_qubits):
        qml.RX(weights[i], wires=i - n_qubits)

    # Measure the expectation value on the first qubit
    return qml.expval(qml.PauliZ(0))

##########################
# 4. PyTorch Module Wrapping the QNode
##########################
# We'll create a simple PyTorch "layer" that calls the above quantum circuit.
# This layer outputs a single logit. You can combine it with classical layers if desired.

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        # We'll have 2*n_qubits trainable parameters in our small circuit
        self.weights = nn.Parameter(0.01 * torch.randn(2*n_qubits))

    def forward(self, x):
        # x shape: (batch_size, n_features)
        logits = []
        for x_single in x:
            # Evaluate the circuit for each sample
            circ_out = quantum_circuit(x_single, self.weights)
            logits.append(circ_out)
        return torch.stack(logits)

##########################
# 5. Define a Simple Model (Quantum-only or Hybrid)
##########################
# Here we use only the quantum layer for demonstration.
# For a hybrid model, add classical layers before/after the quantum layer.

class HybridModel(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.q_layer = QuantumLayer(n_qubits)

    def forward(self, x):
        # Output is a single logit per sample
        return self.q_layer(x)

##########################
# 6. Training Setup
##########################
model = HybridModel(n_qubits)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)
n_epochs = 10
batch_size = 16

# Simple mini-batch generator
def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

##########################
# 7. Training Loop
##########################
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in get_batches(X_train_torch, y_train_torch, batch_size):
        optimizer.zero_grad()
        # Forward pass
        logits = model(X_batch)
        # Compute loss
        loss = criterion(logits.view(-1), y_batch)
        # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(X_train) // batch_size)
    print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {avg_loss:.4f}")

##########################
# 8. Evaluation
##########################
model.eval()
with torch.no_grad():
    logits_test = model(X_test_torch).view(-1)
    # Convert logits to predicted labels (0 or 1)
    preds_test = (torch.sigmoid(logits_test) >= 0.5).float()

    # Calculate accuracy
    accuracy = (preds_test == y_test_torch).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")