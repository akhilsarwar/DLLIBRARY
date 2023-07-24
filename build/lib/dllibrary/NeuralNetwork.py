from dllibrary.ActivationFunctions import activation_function_map, activation_function_deriv_map
from dllibrary.LossFunctions import loss_functions_map, loss_functions_deriv_map
from dllibrary.Layer import Layer 


class NeuralNetwork:
    def __init__(self):
        self.layers=[]
        self.loss_function=loss_functions_map['mean_squared_error']
        self.loss_function_deriv=loss_functions_deriv_map['mean_squared_error']

    def addLayer(self, n=20, activation_function=None):
        n_prev = n
        if len(self.layers) != 0:
            n_prev = self.layers[-1].n
        if activation_function is not None:
            self.layers.append(Layer(n=n, activation_function=activation_function_map[activation_function], activation_function_deriv=activation_function_deriv_map[activation_function], n_prev=n_prev))
        else:
            self.layers.append(Layer(n=n, activation_function=activation_function_map['None'], activation_function_deriv=activation_function_deriv_map['None'], n_prev=n_prev))



    def fit(self, x, y, epoch=10, learning_rate=0.03):
        # modifying the shape of W for first layer since the previous layer (input) size can only be determined after fitting data
        n_prev = x.shape[0]
        self.layers[0].n_prev = n_prev
        self.layers[0].initWeights(self.layers[0].n, n_prev) 

        # assign the loss function based on the activation function
        if self.layers[-1].activation_function is activation_function_map['softmax']:
            self.loss_function = loss_functions_map['categorical_cross_entropy']
            self.loss_function_deriv = loss_functions_deriv_map['categorical_cross_entropy']


        for i in range(epoch):
            y_pred=x
            for j in range(len(self.layers)):
                y_pred = self.layers[j].forwardPropogation(y_pred)
            error = self.loss_function(y_pred=y_pred, y_act=y)

            print(y_pred[:, 0: 3], y[:, 0: 3])

            y_rev = self.loss_function_deriv(y_pred=y_pred, y_act=y)

            for j in range(len(self.layers) - 1, -1, -1):
                y_rev = self.layers[j].backwardPropogation(y_rev, learning_rate=learning_rate)

            print(f'Epoch {i} -----> error: {error}')

                
    def predict(self, x):
        y_pred = x
        for layer in self.layers:
            y_pred = layer.forwardPropogation(y_pred)
        return y_pred     



