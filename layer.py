import perceptron
perc = perceptron.Perceptron()

class layer:

    activationFunc = 0
    numOfNodes = 0 # The number of perceptron (or nodes?) in that layer

    def __init__(self, activationFunc, numOfNodes, input, weight):
        self.activationFunc = activationFunc
        self.numOfNodes = numOfNodes
        self.input = input
        self.perceptronArr = [perceptron.Perceptron(activationFunc) for _ in range(self.numOfNodes)] # An array of perceptron object
        self.output = [0] * self.numOfNodes # An array containing the output of the layer (same size as the numOfNodes)
        self.weight = weight # Assume we take weight from the function that call this class

    def layerCalculation(self):
        for i, perc in enumerate(self.perceptronArr):
            self.output[i] = perc.CalculateOutput(self.weight, self.input)
        return self.output
        
