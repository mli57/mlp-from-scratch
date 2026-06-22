"""
Optimizers, uses each layer's gradients filled in by the backprop process to update the weights.
step() applies one update across all trainable layers.
"""

class SGD:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'grad_W'):  # only Linear layers have weights
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b