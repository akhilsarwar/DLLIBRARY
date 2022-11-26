from ActivationFunctions import activation_function_map, activation_function_deriv_map
from LossFunctions import loss_functions_map, loss_functions_deriv_map
from Layer import Layer 


class NeuralNetwork:
    def __init__(self, learning_rate=0.003):
        self.learning_rate=learning_rate
        self.layers=[]
        self.loss_function=loss_functions_map['mean-squared-error']
        self.loss_function_deriv=loss_functions_deriv_map['mean-squared-error']

    def addLayer(self, n=20, activation_function=None):
        if activation_function is not None:
            n_prev = n
            if len(self.layers) != 0:
                n_prev = self.layers[-1].n
            self.layers.append(Layer(n=n, activation_function=activation_function_map[activation_function], activation_function_deriv=activation_function_deriv_map[activation_function], n_prev=n_prev, learning_rate=self.learning_rate))
        else:
            self.layers.append(Layer(n=n, activation_function=activation_function_map['None'], activation_function_deriv=activation_function_deriv_map['None'], n_prev=n_prev, learning_rate=self.learning_rate))

    def use(self, loss_function):
        self.loss_function = loss_functions_map[loss_function]
        self.loss_function_deriv=loss_functions_deriv_map[loss_function]


    def fit(self, x, y, epoch=100):
        # modifying the shape of W for first layer since the previous layer (input) size can only be determined after fitting data
        n_prev = x.shape[0]
        self.layers[0].n_prev = n_prev
        self.layers[0].initWeights(self.layers[0].n, n_prev)

        y_pred=x
        for i in range(epoch):
            for j in range(len(self.layers)):
                y_pred = self.layers[j].forwardPropogation(y_pred)
            error = self.loss_function(y_pred=y_pred, y_act=y)
            
            y_rev = self.loss_function_deriv(y_pred=y_pred, y_act=y)
            for j in range(len(self.layers) - 1, -1, -1):
                y_rev = self.layers[j].backwardPropogation(y_rev)
            
                
    def predict(self, x):
        y_pred = x
        for layer in self.layers:
            y_pred = layer.forwardPropogation(y_pred)
        return y_pred     

