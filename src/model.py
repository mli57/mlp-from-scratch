"""
The MLP (Multi Layer Perceptron) model. Takes a list of layers and activation functions,
chains them together, and handles the full forward and backward pass over all of them.
"""

class MLP:
    def __init__(self, layers):
        self.layers = layers  # pass in [Linear, ReLU, Linear, ReLU, Linear] etc.

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)