from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from dllibrary.NeuralNetwork import NeuralNetwork


def accuracy(y_pred, y_act):
    pred_max = np.argmax(y_pred, axis=0)
    act_max = np.argmax(y_act, axis=0)
    # print(pred_max, act_max)
    val = (pred_max == act_max).astype(int)
    # print(val)
    return 100 * np.sum(val) / val.shape[0] 


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# reshaping the data and converting the type into float
x_train = np.transpose(x_train.reshape(x_train.shape[0], 28 * 28))
x_train = x_train.astype('float64')
x_train /= 255

# converting to categorical data - y_train
y_train = np.transpose(np_utils.to_categorical(y_train))


x_test = np.transpose(x_test.reshape(x_test.shape[0], 28 * 28))
x_test = x_test.astype('float64')
x_test /= 255


y_test = np.transpose(np_utils.to_categorical(y_test))


nn = NeuralNetwork()
nn.addLayer(n=10, activation_function='tanh')
nn.addLayer(n=10, activation_function='softmax')

nn.fit(x=x_train[:, 0:1000], y=y_train[:, 0:1000], epoch=1000, learning_rate=0.03)

y_pred = nn.predict(x=x_train)


print('Accuracy: ', accuracy(y_pred, y_train))

