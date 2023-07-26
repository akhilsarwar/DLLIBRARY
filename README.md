# Deep Learning Library

## Setup
- Download the wheel file
```
wget https://github.com/akhilsarwar/DLLibrary/blob/main/dist/dllibrary-0.1.0-py3-none-any.whl
```

- Create a virtual environment if required and install the python wheel file.

```
pip3 install dllibrary-0.1.0-py3-none-any.whl
```

## Library Docs
### Classes 
#### Neural Network


| Variables   |      Type      |  Description |
|----------|:-------------:|:------|
| layers | list | Contains Layer objects |
| loss_function |    function   |   Loss function for NN for calculation of backpropogation |


#### Functions
##### <i>addLayer(n=20, activation_function=None)</i> - adds a layer to the neural network. A new layer object is added to the layers list.
- n - (<i>Integer</i>) Number of neurons in that layer
- activation_function - (<i>Function</i>) Activation Function for that layer


##### <i>fit(x, y, epoch=10, learning_rate=0.03)</i> - trains the neural network
 - x - (np array)independent data
 - y - (np array)target data
 - epoch - (Integer) number of iterations for training
 - learning_rate - (Real)a hyper-parameter used to govern the pace at which an algorithm updates.

 ##### <i>predict(x)</i> - predicts the target variable after training the model


<hr>


#### Layer

| Variables   |      Type      |  Description |
|----------|:-------------:|:------|
| n | Integer | Number of Neurons in the layer |
| n_prev |    Integer   |   Number of neurons in the previous layer |
| activation_function |    function   |   Activation Funciton applied for the layer |
| activation_function_deriv |    function   |   Derivative of the Activation Funciton applied for the layer |
| bias |    np Array   |   Bias |
| z |  np Array  |  Output of the linear formula <i>WX + B</i> where W is the weights, X - output of the previous layer, B is bias |
| output |  np Array  |  Output of the <i>activation_function(WX + B)</i> |
| input |  np Array  |  Output of the <i>activation_function(WX + B)</i> of previous Layer which is stored later for backpropogation|

#### Functions

##### <i>initWeights(n, n_prev)</i> -  initializes the weights before training starts.
- n - (integer) Number of Neurons in the layer
- n_prev - (integer) Number of Neurons in the previous layer


##### <i>forwardPropogation(input)</i> computes the linear function combined with activation {activation_function(WX + B)} and stores in output variable
 - input - (np Array) output of previous layer

 ##### <i>backwardPropogation(input, learning_rate)</i> - computes the derivatives dW and dB and updates the weights {W} and bias {B}. Returns the result of backprop to be used by the previous layer.
 - input - (np Array) Result Matrix returned by the latter layer during backprop 
 - learning_rate - (Real) Alpha.


 ### Other Utility Functions
#### ActivationFunctions.py

##### <i>tanh(x), sigmoid(x), softmax(x)</i>
##### Differentiation Functions - <i>tanhDeriv(x), sigmoidDeriv(x), softmaxDeriv(x)</i>

## Usage
Main File