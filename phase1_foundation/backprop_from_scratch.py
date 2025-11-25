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
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            local_deriv = 1.0 if out.data == 0 else out.data
            self.grad += local_deriv * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        n = self.data
        tanh = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(tanh, (self,), 'tanh')
        def _backward():
            local_deriv = (1-tanh**2)
            self.grad = local_deriv * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # Build the list of nodes in forward order
        nodes = []
        visited = set()
        def build_nodes(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_nodes(child)
                nodes.append(v)

        build_nodes(self)        
        # We need to traverse backwards through all the nodes of the expression, so reverse the list
        nodes = list(reversed(nodes))
        
        # Initialize the gradient
        self.grad = 1.0
        for node in nodes:
            node._backward()