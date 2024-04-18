from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch
import numpy as np

# Define model classes
class DecisionTreeModel:
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class NaiveBayesModel:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        if len(proba.shape) == 1:
            # If proba is a 1D array, reshape it to a 2D array with one column
            proba = proba.reshape(-1, 1)
        return proba

class LogisticRegressionModel:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class SVMModel:
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class MLP:
    def __init__(self, **kwargs):
        self.model = MLPClassifier(**kwargs)
         
    def train(self, X, y):
        self.model.fit(X, y)
        self.loss_ = self.calculate_loss(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def calculate_loss(self, X, y_true):
        y_pred = self.predict(X)
        return ((y_true - y_pred) ** 2).mean()

class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class DecisionTreeClassifierModel:
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
