class neuralNet:

    layers = []

    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

    def forwardCalculation(self,input,weight=None):
        for layer in self.layers:
            output = layer.layerCalculation(input, weight)
            input = output
        
        return output
    
    def sseCalculation(self, yhat, y):
        yhat = yhat.apply(lambda x:x[0]) # flatten from list to float
        sse = sum((yhat - y)**2)
        return sse
