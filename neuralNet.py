class neuralNet:

    layers = []

    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

    def forwardCalculation(self,input):
        for layer in self.layers:
            output = layer.layerCalculation(input)
            input = output
        
        return output
