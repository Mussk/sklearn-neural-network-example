import itertools

from sklearn.datasets import load_iris
import random
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
'''
IRIS DATASET
features:
1. sepal length in cm
2. sepal width in cm
3. petal length in cm
4. petal width in cm
5. class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica
'''


class MyPerceptron(object):

    def __init__(self, inputs_count, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.ones(inputs_count + 1)

    def classify(self, x):
        net = np.dot(x, self.weights[1:]) + self.weights[0]
        if net > 0:
            y = 1
        else:
            y = 0
        return y

    def training(self, x, target):
        for epoch in range(self.epochs):
            for xi, yi in zip(x, target):
                classification = self.classify(xi)
                if classification == 0 and yi != 0 \
                        or classification == 1 and yi != 1:
                    self.weights[1:] += self.learning_rate * (yi - classification) * xi
                    self.weights[0] += self.learning_rate * (yi - classification)

    def testing(self, testData_x, testData_y):
        sucsseded_obs = 0
        test_results = []
        for x, y in zip(testData_x, testData_y):
            test_results.append(self.classify(x))
            if self.classify(x) == 1 and y == 1 or self.classify(x) == 0 and y == 0:
                sucsseded_obs += 1
        print('Accuracy: ' + str(sucsseded_obs/len(testData_x)*100) + ' %')



        return test_results



def splitTrainTest (data, target, testPercent):
    trainData_x = []
    trainData_y = []
    testData_x  = []
    testData_y = []
    for d, t in zip(data, target):
        if random.random() < testPercent:
            testData_x.append(d)
            testData_y.append(t)
        else:
            trainData_x.append(d)
            trainData_y.append(t)
    return trainData_x, trainData_y, testData_x, testData_y

def show_iris(testData_x, test_results, title):
    plt.title(title)
    testData_x = np.array(testData_x)

    setosa_res = [el for el in test_results if el == 1]
    setosa_data = np.array(testData_x[:len(setosa_res)])
    non_setosa_data = np.array(testData_x[len(setosa_res):])


    setosa = plt.scatter(setosa_data[:, 0], setosa_data[:, 1], c='red')
    non_setosa = plt.scatter(non_setosa_data[:, 0],non_setosa_data[:, 1], c='blue')


    plt.xlabel('petal length in cm')
    plt.ylabel('petal width in cm')
    plt.legend((setosa, non_setosa),('Setosa','Non Setosa'),loc='lower right')
    plt.show()


iris = load_iris()
x = iris.data[:, [2, 3]]
y = (iris.target == 0).astype(np.int)
x_names = [iris.feature_names[2], iris.feature_names[3]]
y_names = iris.target_names

testPercent = 0.2

trainData_x, trainData_y, testData_x, testData_y = splitTrainTest(x, y, testPercent)

perceptron = MyPerceptron(2, 100, 0.1)

perceptron.training(trainData_x, trainData_y)

test_results = perceptron.testing(testData_x, testData_y)

show_iris(testData_x, test_results, 'MyPerceptron Tests')

#sklearn perceptron

print("Feature size:", len(x))
print("Classes size:", len(y))
print("Classes:\n", y)

classifier = Perceptron(random_state=42)
classifier.fit(x, y)


def test_sklearn(testData_x, testData_y):
    sucsseded_obs = 0
    test_results = []
    for x, y in zip(testData_x,testData_y):
        test_results.append(classifier.predict([x]))
        if classifier.predict([x]) == 1 and y == 1 or classifier.predict([x]) == 0 and y == 0:
            sucsseded_obs += 1
    print('Accuracy (sklearn): ' + str(sucsseded_obs/len(testData_x)*100) + ' %')

    return test_results

test_res_sklern = test_sklearn(testData_x,testData_y)

test_res_sklern = np.array(test_res_sklern).flatten()

show_iris(testData_x, test_res_sklern,'Sklearn Perceptron Tests')

