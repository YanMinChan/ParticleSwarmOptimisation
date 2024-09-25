import numpy as np

class Perceptron:

    activationFunc = 0
    output = 0

    def __init__(self,activationFunc):
        self.activationFunc = activationFunc

    def CalculateOutput(weights, input):
        np.dot(input.T,weights)