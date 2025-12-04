# I am going to do this in two ways:
#
# 1.) Do backpropagation using NumPy on Matrices - This is more of an applied NN case
# 2.) Generalize
#
# Maybe some other things?

import numpy as np

# tanh activation function
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return (1-np.tanh(x)**2)

# mean square error loss function
def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

# Implement backprop with numpy
class TinyNN:
    def __init__(self, layers):
        self.params = {}
        self.cache = {}
        self.grads = {}
        self.layers = layers # [in, hidden, out]
        self.init_parameters()

    def init_parameters(self):
        for i in range(1, len(self.layers)):
            n_in = self.layers[i-1]
            n_out = self.layers[i]
            # Init weights
            self.params[f'W{i}'] = np.random.randn(n_in, n_out) * 0.1 
            self.params[f'b{i}'] = np.random.randn(1, n_out)
    
    def forward(self, X):        
        A = X
        for i in range(1, len(self.layers)):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            Z = np.dot(A, W) + b
            self.cache[f'Z{i}'] = Z
            
            if i == len(self.layers) - 1:
                A = Z
            else:
                A = tanh(Z)

            self.cache[f'A{i}'] = A # cache activated output A for next layer
        return A
    
    def backward(self, X, y_true, y_pred):
        
        m = X.shape[0] # number of examples
        
        # Start with the gradient of L wrt output A
        dA = mse_loss_derivative(y_true, y_pred)

        for i in range(len(self.layers) - 1, 0, -1):
            # i is the current layer index (e.g., 2, then 1)
            Z = self.cache[f'Z{i}']
            A_prev = self.cache[f'A{i - 1}'] if i > 1 else X

            # If not output, apply activation backward.
            if i != len(self.layers) - 1:
                dA = dA * tanh_derivative(Z)

            #Calc gradient for W and b using dot product
            dW = np.dot(A_prev.T, dA) / m
            db = np.sum(dA, axis=0, keepdims=True)

            # Calc gradient backward to previous layer's A
            dA = np.dot(dA, self.params[f'W{i}'].T)

            # store gradients
            self.grads[f'dW{i}'] = dW
            self.grads[f'db{i}'] = db

    def update_parameters(self, h):
        # h is the learning rate
        for i in range(1, len(self.layers)):
            self.params[f'W{i}'] -= h * self.grads[f'dW{i}']
            self.params[f'b{i}'] -= h * self.grads[f'db{i}']


nn = TinyNN(layers = [2, 4, 1])

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]]) # XOR

h = 0.1
epochs = 10000

# TRaining loop
for epoch in range(epochs):
    y_pred = nn.forward(X)
    loss = mse_loss(y, y_pred)
    nn.backward(X, y, y_pred)

    nn.update_parameters(h)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Predictions:")        
print(nn.forward(X))






# 3 inputs, 2 hidden layers and 2 outputs 

input_size = 2
hidden_size = 3
output_size = 1
h = 0.1
epochs = 10000

X = np.array([[0,0], [0,1],[1,0],[1,1]]) # 4 x input_size
y = np.array([[0], [1], [1], [1]]) # OR

# Set everything up
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1 
b2 = np.zeros((1, output_size))

for epoch in range(epochs):
    # Hidden layer
    Z1 = (np.dot(X, W1) + b1)
    A1 = tanh(Z1)

    # Output layer
    Z2 = (np.dot(A1, W2) + b2)
    A2 = Z2

    # Output layer and delta
    d_loss_d_A2 = mse_loss_derivative(y, A2)
    d_loss_d_Z2 = d_loss_d_A2

    dW2 = np.dot(A1.T, d_loss_d_Z2)
    db2 = np.sum(d_loss_d_Z2, axis = 0, keepdims=True)

    d_loss_d_A1 = np.dot(d_loss_d_Z2, W2.T)
    d_loss_d_Z1 = d_loss_d_A1 * tanh_derivative(Z1)

    dW1 = np.dot(X.T, d_loss_d_Z1)
    db1 = np.sum(d_loss_d_Z1, axis = 0, keepdims = True)

    W1 -= h * dW1
    b1 -= h * db1
    W2 -= h * dW2
    b2 -= h * db2

    # print loss
    if epoch % 1000 == 0:
        loss = mse_loss(y, A2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

Z1 = np.dot(X,W1) + b1
A1 = tanh(Z1)
Z2 = np.dot(A1, W2) + b2
preds = Z2

print("Predictions:")
print(preds)

