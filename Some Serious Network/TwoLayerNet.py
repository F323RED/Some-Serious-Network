import numpy as np
import F323REDsMLKit as ml

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.parmas = {}

        self.parmas['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.parmas['b1'] = np.zeros(hidden_size)

        self.parmas['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.parmas['b2'] = np.zeros(output_size)

    def Predict(self, x):
        W1, W2 = self.parmas['W1'], self.parmas['W2']
        b1, b2 = self.parmas['b1'], self.parmas['b2']

        a1 = np.dot(x, W1) + b1
        z1 = ml.Sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = ml.Softmax(a2)

        return y

    def Loss(self, x, t):
        y = self.Predict(x)

        return ml.CrossEntropyError(y, t)

    def Accuracy(self, x, t):
        y = self.Predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

    def Gradient(self, x, t):
        loss_w = lambda W : self.Loss(x, t)

        grads = {}
        
        grads['W1'] = ml.NumeriaclGradient(loss_w, self.parmas['W1'])
        grads['b1'] = ml.NumeriaclGradient(loss_w, self.parmas['b1'])
        grads['W2'] = ml.NumeriaclGradient(loss_w, self.parmas['W2'])
        grads['b2'] = ml.NumeriaclGradient(loss_w, self.parmas['b2'])

        return grads
