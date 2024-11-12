import perceptron
import numpy as np

class layer:

    activationFunc = 0
    numOfNodes = 0 # The number of perceptron (or nodes?) in that layer

    def __init__(self, activationFunc, numOfNodes):
        self.activationFunc = activationFunc
        self.numOfNodes = numOfNodes
        self.perceptronArr = [perceptron.Perceptron(activationFunc) for _ in range(self.numOfNodes)] # An array of perceptron object
        self.output = None # An array containing the output of the layer (same size as the numOfNodes)
        self.inputs = None # set up some variables for storing inputs and weights (might not be necessary but could be useful to save them to memory)
        self.weigths = None
        self.bias = None

    def layerCalculation(self, inputs, weights, bias):

        self.weights = weights
        self.inputs = inputs
        self.bias = bias

        self.output = [0] * len(self.perceptronArr)
        # initialize all weights as 1 in case we do not have any specified
        # it is now an 2D array, lists containing weights for a single neuron (one weight for each input) grouped into a list (one list of weights for every neuron)
        # if weights == None:
        #     self.weights = [[0.2] * len(inputs) for _ in range(self.numOfNodes)]

        # if bias == None:
        #     self.bias = [1] * len(self.perceptronArr)
        for i, perc in enumerate(self.perceptronArr):
            self.output[i] = perc.CalculateOutput(weights=self.weights[i], input=inputs, bias=self.bias[i])
        return self.output
        
