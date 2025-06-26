import numpy as np
from Optimization import Optimizers
from Layers import Base
import copy

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(size=(input_size + 1, output_size))
        self.bias = np.random.uniform(size = (1, output_size))
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None
        self.temp = []

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        self.lastIn = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        self.lastOut = np.dot(self.lastIn, self.weights)
        return self.lastOut
     
    def backward(self, error_tensor):
        dW = np.dot(self.lastIn.T, error_tensor)
        dx = np.dot(error_tensor, self.weights[:-1].T)
        db = np.sum(error_tensor, axis = 0)
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, dW)

            self.bias = self._optimizer.bias.calculate_update(self.bias, db)
       
        self.gradient_bias = error_tensor
        self.gradient_weights = dW
       
        return dx


    def initialize(self, weights_initializer, bias_initializer):
        self.weights[:-1] = weights_initializer.initialize(self.weights[:-1].shape, self.input_size, self.output_size)
        self.weights[-1:] = bias_initializer.initialize((1, self.output_size), 1, self.output_size)

    
    
