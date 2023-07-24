import numpy as np

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(x * -1))

def relu():
    pass

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0)

def noActivation():
    pass

def tanhDeriv(x):
    return 1 - (np.tanh(x) * np.tanh(x))

def sigmoidDeriv(x):
    return sigmoid(x) * (1 - sigmoid(x))
    
def reluDeriv():
    pass

def softmaxDeriv(x):
    pass

def noActivationDeriv():
    pass

activation_function_map = {"tanh": tanh, "sigmoid": sigmoid, "relu": relu, "None": noActivation, "softmax": softmax}
activation_function_deriv_map = {"tanh" : tanhDeriv, "sigmoid": sigmoidDeriv, "relu": reluDeriv, "None": noActivationDeriv, 'softmax': softmaxDeriv}