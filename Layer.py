import numpy as np

class Layer:
    def __init__(self, n, activation_function, activation_function_deriv, n_prev, learning_rate):
        self.n = n
        self.n_prev = n_prev
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation_function_deriv = activation_function_deriv
        self.initWeights(n, n_prev)
        self.bias = np.zeros((n, 1))
        self.z = None
        self.output = None
        self.input = None


    def initWeights(self, n, n_prev):
        self.weights = np.random.randn(n, n_prev) * 0.01

    # ForwardProp
    # Z[l] = W[l]A[l - 1] + B[l]
    # A[l] = G[1](Z[l])

    # W[l] - weights for the layer l
    # G[l] - activation function for layer L

    def forwardPropogation(self, input):
        self.input = input
        self.z = np.matmul(self.weights, self.input) + self.bias
        self.output = self.activation_function(self.z)
        return self.output


    # Backward Propogation
    # dZ[l] = dA[l] * G[l]'(Z[l])
    # dW[l] = (1 / m) *(dZ[l] . (A[l - 1])T)
    # dB[l] = (1 / m) * (np.sum(dZ[l], axis=1, keepDims=true))
    # dA[l - 1] = (W[l])T . dZ[l]

    # * - element wise multiplication
    # . - matrix multiplication
    # (A)T - transpose of matrix A
    # m - no of training examples

    # here input is dA[l] of dimension (n, m) where n is no of neurons in the layer and m is the no of training examples
    def backwardPropogation(self, input):
        m = input.shape[1]
        dZ = input * self.activation_function_deriv(self.z)
        dW = (1 / m) * (np.matmul(dZ, np.transpose(self.input)))
        dB = (1 / m) * (np.sum(dZ, axis=1, keepdims=True))
        dA_prev = np.matmul(np.transpose(self.weights), dZ)
        self.weights = self.weights - self.learning_rate * dW
        self.bias = self.bias - self.learning_rate * dB
        return dA_prev