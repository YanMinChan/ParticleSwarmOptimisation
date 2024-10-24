import numpy as np

class Perceptron:

    activationFunc = 0
    output = 0

    def __init__(self,activationFunc):
        self.activationFunc = activationFunc

# YM: I think we haven't implement the activation func right?
    def CalculateOutput(self, weights, input):
        return np.dot(np.array(input).T,np.array(weights))
