"""
Trainable layers for the MLP. Each layer implements forward() for the forward pass
and backward() to compute gradients and pass them to the previous layer.
"""
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        # initialization, same as PyTorch's default for ReLU networks
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x  # save input for backprop
        return x @ self.W + self.b  # same as nn.Linear

    def backward(self, grad):
        # grad is dLoss/d(output of this layer)
        self.grad_W = self.x.T @ grad   # dLoss/dW
        self.grad_b = grad.sum(axis=0)  # dLoss/db
        return grad @ self.W.T          # dLoss/d(input), passed to previous layer