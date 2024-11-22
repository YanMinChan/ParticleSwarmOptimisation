class neuralNet:

    #layers = []

    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)

    def forwardCalculation(self,input,weight=None,bias=None):
        i = 0
        for layer in self.layers:
            output = layer.layerCalculation(input, weight[i], bias[i])
            #print("input", input)
            input = output
            i = i + 1
        #print("output", output)
        return output
    
    def errorCalculation(self, yhat, y):
        yhat = yhat.apply(lambda x:x[0]) # flatten from list to float
        sse = sum((abs(yhat - y)))/len(yhat)
        return sse
