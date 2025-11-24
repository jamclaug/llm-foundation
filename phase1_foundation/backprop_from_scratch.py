import math
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__ (self, data, children=(), _op = '', label = ''):
        self.data = data
        self.grad = 0.0 # initialize gradient to zero
        self._prev = set(children) # set of child nodes
        self._op = _op # operation that produced this node
        self.label = label # optional label for visualization

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    def tanh(self):
        n = self.data
        tanh = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(tanh, (self,), 'tanh')
        return out
    
