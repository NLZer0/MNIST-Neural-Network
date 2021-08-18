import numpy as np
from numpy.core.numeric import outer 

class NeuralNetwork:
    
    def __init__(self):
        self.count_of_lays = 0
        self.layers = np.array([])   

    class ReluLayer:

        def __init__(self, size:tuple):
            self.W = 2*np.random.rand(size[0], size[1]) - 1

        def forward(self, X):
            self.X = X
            output = X.dot(self.W)
            output = output * (output > 0)
            self.output = output
            return output

        def HeavySide(self, x):
            return (x>0).astype('int32')

        def backward(self, output_grad, learning_rate = 0.01):
            
            grad_X = (output_grad*(self.HeavySide(self.output))).dot(self.W.T)
            grad_W = self.X.T.dot(output_grad * self.HeavySide(self.output)) 
            self.W -= learning_rate * grad_W
            return grad_X

    class LayerLinear:

        def __init__(self, size:tuple):
            self.W = 2*np.random.rand(size[0], size[1]) - 1

        def forward(self, X):
            self.X = X
            output = X.dot(self.W)
            return output
        
        def backward(self, output_grad, learning_rate = 0.01):
            grad_W = self.X.T.dot(output_grad)
            grad_X = output_grad.dot(self.W.T)
            self.W -= learning_rate * grad_W
            return grad_X

    def softmax(self, y):
        for it in range(y.shape[0]):
            new_row = np.exp(y[it,:] - np.max(y[it,:]))
            sum = np.sum(new_row)
            y[it,:] = new_row/sum
        return y

    
    def CreateLayerRelu(self, size:tuple):
        new_lay = self.ReluLayer(size)
        self.layers = np.append(self.layers, new_lay)

    def CreateLayerLinear(self, size:tuple):
        new_lay = self.LayerLinear(size)
        self.layers = np.append(self.layers, new_lay)
    
    def forward(self, X_train):
        output = self.layers[0].forward(X_train)
        for it in self.layers[1:]:
            output = it.forward(output)
        return output
    
    def CrossEntropyLoss(self, predictions, true_labels):

        # Альтеранативный способ поиска кросс энтропии
        # cr_en_loss = 0
        # for it in range(len(true_labels)):
        #     cr_en_loss += -np.log(predictions[it, true_labels[it]])
        
        self.predictions = predictions
        self.true_labels = true_labels
        arr = predictions[range(len(true_labels)), true_labels.reshape(1,-1)]
        
        # Значение не должно быть равно 0, для того, чтобы не было ошибки при взятии логарифма, потому передадим близкое к 0 значение
        arr[arr == 0] += 1e-6
        cr_en_loss = np.sum(-np.log(arr))

        return cr_en_loss/len(true_labels)

    def backward(self, learning_rate = 0.01):
        """Работает только при использовании softmax и кроссэнтропии"""
        
        output_delta = np.zeros_like(self.predictions)
        for it in range(output_delta.shape[0]):
            output_delta[it, self.true_labels[it]] = 1
        output_grad = self.predictions - output_delta

        for it in self.layers[::-1]:
            output_grad = it.backward(output_grad,learning_rate)



# batch_size = 4
# count_of_classes = 10
# learning_rate = 0.1

# X = np.random.rand(batch_size,5)
# y = np.random.randint(0, count_of_classes, batch_size).reshape(batch_size,1)


# nn = NeuralNetwork()

# nn.CreateLayerRelu(size=(5,7))
# nn.CreateLayerRelu(size=(7,5))
# nn.CreateLayerLinear(size=(5,count_of_classes))

# epochs_count = 10
# for it in range(epochs_count):
#     output = nn.forward(X)
#     prediction = nn.softmax(output)
#     loss = nn.CrossEntropyLoss(prediction, y)
#     nn.backward(learning_rate)
#     print(loss)


