from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.base import BaseEstimator

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

class SVMModel(BaseEstimator):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
                 tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False, random_state=None):
        # Pass explicit arguments to the SVC constructor
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

        self.model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                         probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                         verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                         break_ties=break_ties, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class MLP:
    def __init__(self, **kwargs):
        self.model = MLPClassifier(**kwargs)
        self.losses_ = []
        self.accuracies_ = []

    def train(self, train_x, train_y, test_x, test_y):
        from sklearn.metrics import accuracy_score
        for _ in range(self.model.max_iter):
            self.model.partial_fit(train_x, train_y, classes=np.unique(train_y))
            self.losses_.append(self.model.loss_)
            train_pred = self.model.predict(train_x)
            train_accuracy = accuracy_score(train_y, train_pred)
            self.accuracies_.append(train_accuracy)
            test_pred = self.model.predict(test_x)
            test_accuracy = accuracy_score(test_y, test_pred)
            print(f'Iteration {_+1}/{self.model.max_iter}, Training Loss: {self.model.loss_:.4f}, Training Accuracy: {self.accuracies_[-1]:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    def predict(self, X):
        return self.model.predict(X)
    
    def plot_loss_and_accuracy(self):
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(self.losses_, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(self.accuracies_, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()
        
    
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
    
