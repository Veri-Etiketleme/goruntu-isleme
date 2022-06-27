import numpy as np
import math
from ann.layer import Layer


class Linear(Layer):
    """
    Base class for Linear Layers of the network
    """
    def __init__(self, in_dim: int, out_dim: int, initialization='xavier'):
        """
        Initialize a hidden layer for the ann
        """
        self.in_dim = in_dim #bias
        self.out_dim = out_dim
        self.initialization = initialization
        
        # reset history
        self.reset()
        self.initialize()
#         print(f"Layer: {self},  initialization: {self.W}")
        
    def reset(self):        
        self.W = np.zeros((self.in_dim, self.out_dim))
        self.b = np.zeros(self.out_dim)
        
        # last forwarded input
        self._input = None                
        # last calculated gradient for updating wts
        self.grad = np.zeros_like(self.W)

    def initialize(self):
        """
        Initialize weight vectors as per the given initialization strategy
        :return:
        """
        if self.initialization == 'xavier': # sigmoid and tanh
            self.W = np.random.uniform(-1, 1, (self.W.shape)) * math.sqrt(6/(self.in_dim + self.out_dim))
        elif self.initialization == 'kaiming': # ReLU
            self.W = np.random.standard_normal(self.W.shape) * math.sqrt(2./self.in_dim)
        else:
            raise Exception("Unsupported Initialization strategy")

    def forward(self, _input: np.array):
        """
        _input (1000, 3)
        """
        if len(_input.shape) == 1: # 1D array
            _input = _input.reshape(1, -1) # make it a matrix
        self._input = _input
        
#         print(self._input.shape, self.W.shape, self.b.shape)
        return (np.matmul(self._input, self.W) + self.b) # (100,3) X (3,2) = (100, 2) + (2,)

    def back_prop(self, dl_dy, lr): # (2, )
        """
        :param dl_dy: (size, out_dim)
        calculates dL/dw and back return dL/dx for back prop
        Also capable of calculating loss' for the single given unit index
        """
        size = self._input.shape[0] # size, in_dim
        self.grad = self._input.T @ dl_dy # (in_dim, out_dim)
        # matrix from array
        self.grad = self.grad.reshape(self.W.shape)
        
#         print(f"Grad: {self.grad}, shape: {self.grad.shape}")
        
        # update wts
        self.grad = lr * self.grad
#         print(f"{self.grad}, {self.grad.shape}")
#         print(f"W: {self.W}, {self.W.shape}")        
        self.W -= self.grad
        
        # update bias
        self.b -= lr * np.mean(dl_dy, axis=0)
        
#         print(f"dL_dy: {dl_dy}, {dl_dy.shape}")
        dy_dx = dl_dy.reshape(size, self.out_dim) @ self.W.T
        # (size, in_dim)
        return dy_dx 

    def update(self, lr):
        """
        :param lr: Learning Rate
        """
        self.W -= lr * self.grad

    def __str__(self):
        return f"{self.__class__}: {(self.in_dim, self.out_dim)}"


if __name__ == '__main__':
    inp = np.random.randint(0, 255, 6)
    l1 = Linear(6, 3)
    l1.initialize()
    print(l1.forward(inp))
    print(l1.back_prop(np.array([0.1, 0.2, 0.3])))
