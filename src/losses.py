"""
Loss functions. Measures how wrong the model's predictions are.
forward() computes the scalar loss value
backward() returns the gradient of the loss with respect to the model's output logits, which kicks off backpropagation.
"""
import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, y_true):
        # softmax
        shifted = logits - logits.max(axis=1, keepdims=True)  # numerical stability
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        
        # cross entropy loss
        n = y_true.shape[0]
        self.y_true = y_true
        log_probs = -np.log(self.probs[np.arange(n), y_true] + 1e-9)
        return log_probs.mean()

    def backward(self):
        n = self.y_true.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.y_true] -= 1
        return grad / n