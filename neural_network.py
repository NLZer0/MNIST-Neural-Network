import numpy as np
from numpy.core.numeric import outer 

class NeuralNetwork:

    class ReluLay:

        def __init__(self, size:tuple):
            self.W = 2*np.random.rand(size[0], size[1]) - 1

        def forward(self, X):
            self.output = X.dot(self.W)
            self.output = self.output * (self.output > 0)
            return self.output

    class SoftmaxLay:

        def __init__(self, size:int):
            self.W = 2*np.random.rand(size, 1)

        def softmax(self, y):
            y = np.exp(y)
            sum = np.sum(y)
            y /= sum
            return y

        def forward(self, X):
            self.output = X.dot(self.W)
            self.output = self.softmax(self.output)
            return self.output



    def __init__(self):
        self.count_of_lays = 0
        self.layers = np.array([])   
    
    def CreateLayerRelu(self, size:tuple):
        new_lay = self.ReluLay(size)
        self.layers = np.append(self.layers, new_lay)

    def CreateLayerSoftmax(self, size:int):
        new_lay = self.SoftmaxLay(size)
        self.layers = np.append(self.layers, new_lay)

    
    def forward(self, X_train):
        output = self.layers[0].forward(X_train)
        for it in self.layers[1:]:
            output = it.forward(output)
        
        print(output)

            





X = np.random.rand(4,5)
y = np.random.rand(3,1)


nn = NeuralNetwork()

nn.CreateLayerRelu(size=(5,5))
nn.CreateLayerRelu(size=(5,3))
nn.CreateLayerSoftmax(size=3)

nn.forward(X)


