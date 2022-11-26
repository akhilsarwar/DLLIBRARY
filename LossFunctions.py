import numpy as np

def meanSquaredError(y_pred, y_act):
    return np.mean(np.power(y_pred - y_act, 2))

def meanSquaredErrorDeriv(y_pred, y_act):
    return 2 * (y_act - y_pred) / y_pred.shape[0]

loss_functions_map = {"mean_squared_error": meanSquaredError}
loss_functions_deriv_map = {"mean_squared_error": meanSquaredErrorDeriv}