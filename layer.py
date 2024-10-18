import perceptron

class layer:

    activationFunc = 0
    numOfNodes = 0 # The number of perceptron (or nodes?) in that layer

    def __init__(self, activationFunc, numOfNodes):
        self.activationFunc = activationFunc
        self.numOfNodes = numOfNodes
        self.perceptronArr = [perceptron.Perceptron(activationFunc) for _ in range(self.numOfNodes)] # An array of perceptron object
        self.output = [0] * self.numOfNodes # An array containing the output of the layer (same size as the numOfNodes)
        self.inputs = None # set up some variables for storing inputs and weights (might not be necessary but could be useful to save them to memory)
        self.weigths = None

        # Now that I think of it the weights might be more complicated than I thought, now we are treating it as one array for a layer, but
        # every perceptron actually has its own array of weights. Maybe here the input needs to be a 2D array, like an array of arrays?

        # Changed the inputs and the weights to be passed to the calculation function instead of the constructor, probably makes more sense like this?
        # I imagine we might want to change those without remaking a layer

    def layerCalculation(self, inputs, weights = None):

        self.weights = weights
        self.inputs = inputs

        # initialize all weights as 1 in case we do not have any specified
        # it is now an 2D array, lists containing weights for a single neuron (one weight for each input) grouped into a list (one list of weights for every neuron)
        if weights == None:
            weights = [[1] * len(inputs)] * self.numOfNodes

        for i, perc in enumerate(self.perceptronArr):
            self.output[i] = perc.CalculateOutput(weights=weights[i], input=inputs)
        return self.output
        
