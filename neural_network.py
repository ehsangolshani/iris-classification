from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np

"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris
optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
epoch = 500
"""

torch.manual_seed(1234)

# hyperparameters
hl = 10
lr = 0.01
num_epoch = 10000


# build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


"""
SECTION 1 : Load and setup data for training
the datasets separated in two files from originai datasets:
iris_train.csv = datasets for training purpose, 80% from the original data
iris_test.csv  = datasets for testing purpose, 20% from the original data
"""

# load
dataset = pd.read_csv("iris.data", names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# change string value to numeric
dataset.loc[dataset['species'] == 'Iris-setosa', 'species'] = 0
dataset.loc[dataset['species'] == 'Iris-versicolor', 'species'] = 1
dataset.loc[dataset['species'] == 'Iris-virginica', 'species'] = 2

dataset = dataset.apply(pd.to_numeric)

# change dataframe to numpy array
data_set_array = dataset.as_matrix()

kf = KFold(n_splits=10)

for train, test in kf.split(data_set_array):

    # split x and y (feature and target)
    xtrain = np.array(data_set_array)[train][:, :4]
    ytrain = np.array(data_set_array)[train][:, 4]

    net = Net()

    # choose optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # train
    for epoch in range(num_epoch):
        X = Variable(torch.Tensor(xtrain).float())
        Y = Variable(torch.Tensor(ytrain).long())

        # feedforward - backprop
        optimizer.zero_grad()
        out = net(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

        if (epoch) % 50 == 0:
            print('Epoch [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epoch, loss.data[0]))

    """
    # SECTION 3 : Testing model
    # """

    # split x and y (feature and target)
    xtest = np.array(data_set_array)[test][:, :4]
    ytest = np.array(data_set_array)[test][:, 4]

    # get prediction
    X = Variable(torch.Tensor(xtest).float())
    Y = torch.Tensor(ytest).long()
    out = net(X)
    _, predicted = torch.max(out.data, 1)

    # get accuration
    print('Accuracy of the network %d %%' % (100 * torch.sum(Y == predicted) / 30))
