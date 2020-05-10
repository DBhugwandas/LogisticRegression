# Implementing logistic regression from Scratch

import numpy as np
import math
import copy
from numpy import genfromtxt

def sigmoid(Input):

    Out = 1 / (1 + np.exp(-Input))
    return Out

class LogisticModel():

    def __init__(self, XVal, yVal, alpha, lamb):

        self.XVal = XVal
        self.yVal = yVal
        self.alpha = alpha
        self.lamb = lamb

        # Number of Training Examples
        self.m = XVal.shape[0]
        # Number of features
        self.n = XVal.shape[1]

        # Intializing the theta matrix
        self.Theta = np.zeros((self.n + 1, 1))

        # Inserting the bias term into the XVal
        self.XVal = np.insert(self.XVal, 0, 1, axis=1)

        # Resize
        self.yVal = self.yVal.reshape((self.m, 1))

    def Accuracy(self, X, Y, Theta):

        NbrElements = X.shape[0]

        # FeedFoward
        Z = np.dot(Theta.transpose(), X.transpose())
        Output = (sigmoid(Z)).transpose()  # m x 1 vector

        # Accuracy Calcs
        Outnew = Output > 0.5
        match = Outnew == Y
        Acc = sum(match) / NbrElements

        True_pos = 0
        True_neg = 0
        False_pos = 0
        False_neg = 0

        #Precision and Recall
        for i in range(Outnew.shape[0]):

            if Outnew[i] == 1 and Y[i] == 1:
                True_pos += 1

            elif Outnew[i] == 0 and Y[i] == 0:
                True_neg += 1

            elif Outnew[i] == 1 and Y[i] == 0:
                False_pos += 1

            elif Outnew[i] == 0 and Y[i] == 1:
                False_neg += 1
        try:
            Precision = True_pos / (True_pos + False_pos)
        except:
            Precision = 0

        Recall = True_pos / (True_pos + False_neg)

        return Acc, Precision, Recall

    def Train(self, Xtest, Ytest):

        self.Xtest = Xtest
        self.Xtest = np.insert(self.Xtest, 0, 1, axis=1)
        self.Ytest = Ytest.reshape((Ytest.shape[0], 1))

        for i in range(2000):

            # FeedForward
            Z = np.dot(self.Theta.transpose(), self.XVal.transpose())
            Output = (sigmoid(Z)).transpose()  # m x 1 vector

            # Computing error
            Theta_adj = copy.deepcopy(self.Theta)
            Theta_adj[0] = 0

            Error = 1 / self.m * (np.sum(-self.yVal * np.log(Output) - (1 - self.yVal) * np.log(1 - Output))) + (self.lamb / (2 * self.m)) * np.sum(Theta_adj ** 2)

            # Gradient Descent
            inner = (Output - self.yVal).transpose()
            PartDer = (1 / self.m * np.dot(inner, self.XVal)).transpose() + (self.lamb / self.m) * Theta_adj
            self.Theta = self.Theta - self.alpha * (PartDer)

            # Train Accuracy
            Acc, Prec, Rec = self.Accuracy(self.XVal, self.yVal, self.Theta)

            # Test Accuracy
            TestAcc, Pre1, Rec1 = self.Accuracy(self.Xtest, self.Ytest, self.Theta)

            print("Test Accuracy: ", TestAcc)
            print("Test Precision: ", Pre1)
            print("Test Recall: ", Rec1)

        return self.Theta


my_data = genfromtxt('diabetes.csv', delimiter=',')[1:, :]

# Suffle data
np.random.seed(10)
np.random.shuffle(my_data)

# Selecting Train and Validate
Nbr = my_data.shape[0]
TrainTest = 0.3
Selection = round(Nbr * 0.7)

Ytrain = my_data[0:Selection, -1]
Xtrain = my_data[0:Selection, :-1]

Ytest = my_data[Selection + 1:, -1]
Xtest = my_data[Selection + 1:, :-1]


Model = LogisticModel(Xtrain, Ytrain, 0.0001, 0.01)
Model.Train(Xtest, Ytest)
