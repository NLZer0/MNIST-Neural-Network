import numpy as np
import pandas as pd 

train_data = pd.read_csv('data/diabetes.csv')

features = list(train_data)

mode_values = train_data.mean()[1:-1]
for it in features[1:-1]:
    train_data[it] = train_data[it].replace(0, mode_values[it])


X = train_data.drop(columns=['Outcome', 'Insulin', 'SkinThickness'], axis=1)

y = train_data.Outcome

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

X_train_means = X_train.mean()
X_test_means = X_test.mean()
for it in list(X_train):
    X_train[it] -= X_train_means[it]
    X_test[it] -= X_test_means[it]

X_train = np.asarray(X_train) 
X_test = np.asarray(X_test)
y_train = np.asarray(y_train).reshape(-1,1)
y_test = np.asarray(y_test).reshape(-1,1)

print(X_train.shape, y_train.shape)

from neural_network import NeuralNetwork

count_of_classes = 2
count_of_features = X_train.shape[1]
learning_rate = 1e-04
epochs_count = 50

nn = NeuralNetwork()

nn.CreateLayerRelu(size=(count_of_features,9))
nn.CreateLayerRelu(size=(9,11))
nn.CreateLayerLinear(size=(11,count_of_classes))


for it in range(epochs_count):
    output = nn.forward(X_train)
    prediction = nn.softmax(output)
    loss = nn.CrossEntropyLoss(prediction, y_train)
    nn.backward(learning_rate)
    print(loss)

