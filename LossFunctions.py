import numpy as np

def meanSquaredError(y_pred, y_act):
    return np.sum(np.mean(np.power(y_pred - y_act, 2), axis=0))

def meanSquaredErrorDeriv(y_pred, y_act):
    return 2 * (y_pred - y_act) / y_pred.shape[0]

def categoricalCrossEntropy(y_pred, y_act):
    return np.sum(-1 * np.sum(y_act * np.log(y_pred), axis=0))
    
# considering categorical cross entropy is used for softmax function - then returning
def categoricalCrossEntropyDeriv(y_pred, y_act):
    return y_pred - y_act

loss_functions_map = {"mean_squared_error": meanSquaredError, "categorical_cross_entropy": categoricalCrossEntropy}
loss_functions_deriv_map = {"mean_squared_error": meanSquaredErrorDeriv, "categorical_cross_entropy" : categoricalCrossEntropyDeriv}