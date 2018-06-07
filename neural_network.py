import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

"""
Build Neural Network Model
It's a Multilayer Perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 4 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris
optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.02
epoch = 2000
"""

torch.manual_seed(1234)

# parameters
hl = 4
lr = 0.02
num_epoch = 2000
k_fold_split_num = 10


# build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


"""
Load and data pre-processing for training
first load csv dataset and define headers for it, then shuffle it, 
then change data species header into numeric values, 
then transform loaded data to numpy array format 
"""

# load dataset
dataset = pd.read_csv("iris.data", names=[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], )

# shuffle data rows
dataset = dataset.reindex(np.random.permutation(dataset.index))

# change string value to numeric
dataset.loc[dataset['species'] == 'Iris-setosa', 'species'] = 0
dataset.loc[dataset['species'] == 'Iris-versicolor', 'species'] = 1
dataset.loc[dataset['species'] == 'Iris-virginica', 'species'] = 2

dataset = dataset.apply(pd.to_numeric)

# change dataframe to numpy array
data_set_array = dataset.as_matrix()

kf = KFold(n_splits=k_fold_split_num)

average_accuracy = 0.0

average_precision_setosa = 0.0
average_precision_versicolor = 0.0
average_precision_virginica = 0.0

average_recall_setosa = 0.0
average_recall_versicolor = 0.0
average_recall_virginica = 0.0

average_f_measure_setosa = 0.0
average_f_measure_versicolor = 0.0
average_f_measure_virginica = 0.0

"""
using K-Fold Cross Validation Technique (k=10)
"""

for train, test in kf.split(data_set_array):

    """
    training model
    """

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
    Testing model and calculating metrics including:
        accuracy
        precision
        recall
        f-measure
    """

    # split x and y (feature and target)
    xtest = np.array(data_set_array)[test][:, :4]
    ytest = np.array(data_set_array)[test][:, 4]

    # get prediction
    X = Variable(torch.Tensor(xtest).float())
    Y = Variable(torch.Tensor(ytest).long())
    out = net(X)

    _, predicted = torch.max(out.data, 1)

    precision, recall, f_measure, support = precision_recall_fscore_support(
        Y, predicted, average=None, labels=[0, 1, 2])

    accuracy = 100 * torch.sum(Y == predicted) / len(test)

    average_accuracy += accuracy

    average_precision_setosa += precision[0]
    average_precision_versicolor += precision[1]
    average_precision_virginica += precision[2]

    average_recall_setosa += recall[0]
    average_recall_versicolor += recall[1]
    average_recall_virginica += recall[2]

    average_f_measure_setosa += f_measure[0]
    average_f_measure_versicolor += f_measure[1]
    average_f_measure_virginica += f_measure[2]

"""
calculate and print final average result    :)
"""

print("\n\nAccuracy of the network: {}%".format(average_accuracy / k_fold_split_num))

print("\nprecision of the network precision for setosa: {}%".format(
    100 * (average_precision_setosa / k_fold_split_num)))
print("precision of the network precision for versicolor: {}%".format(
    100 * (average_precision_versicolor / k_fold_split_num)))
print("precision of the network precision for virginica: {}%".format(
    100 * (average_precision_virginica / k_fold_split_num)))

print("\nAccuracy of the network recall for setosa: {}%".format(
    100 * (average_recall_setosa / k_fold_split_num)))
print("Accuracy of the network recall for versicolor: {}%".format(
    100 * (average_recall_versicolor / k_fold_split_num)))
print("Accuracy of the network recall for virginica: {}%".format(
    100 * (average_recall_virginica / k_fold_split_num)))

print("\nAccuracy of the network f_measure for setosa: {}%".format(
    100 * (average_f_measure_setosa / k_fold_split_num)))
print("Accuracy of the network f_measure for versicolor: {}%".format(
    100 * (average_f_measure_versicolor / k_fold_split_num)))
print("Accuracy of the network f_measure for virginica: {}%\n".format(
    100 * (average_f_measure_virginica / k_fold_split_num)))
