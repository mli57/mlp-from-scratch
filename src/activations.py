"""
ReLU Activation functions, used right after linear layers to allow the network to learn complex patterns. 
Each one has forward() and backward() so they plug into the same layer pipeline as Linear.
"""

class ReLU:
    def forward(self, x):
        self.mask = (x>0) # saves positive values
        return x *self.mask
    
    def backward(self, grad):
        return grad*self.mask # zero out gradients where input was negative
    
    