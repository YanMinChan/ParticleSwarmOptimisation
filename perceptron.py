import numpy as np

class Perceptron:

    activationFunc = 0
    output = 0

    def __init__(self,activationFunc):
        self.activationFunc = activationFunc

    def CalculateOutput(self, weights, input, bias):
        z = np.dot(np.array(input).T,np.array(weights)) + bias
        return self.activationFunc(z)
