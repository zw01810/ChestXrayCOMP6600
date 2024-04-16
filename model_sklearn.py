from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class NaiveBayes:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class LogisticRegressionClassifiers:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class SVM:
    def __init__(self):
        self.model = SVC()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class MLPClassifier:
    def __init__(self, max_iter=200):
        self.model = SklearnMLPClassifier(max_iter=max_iter)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

def evaluate_model(model_name, y_true, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'{model_name} Accuracy:', accuracy)

    # Classification report
    print(f'Classification Report for {model_name}:')
    print(classification_report(y_true, y_pred))

    # Confusion matrix
    print(f'Confusion Matrix for {model_name}:')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # F1 score
    f1 = f1_score(y_true, y_pred, average='micro')
    print(f'{model_name} F1 Score:', f1)

    # Precision score
    precision = precision_score(y_true, y_pred, average='weighted')
    print(f'{model_name} Precision Score:', precision)

    # Recall score
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f'{model_name} Recall Score:', recall)

    '''# ROC AUC score and curve
    if len(np.unique(y_true)) > 2:
        print(f'ROC AUC Score is only supported for binary classification. Skipping for {model_name}.')
    else:
        roc_auc = roc_auc_score(y_true, y_pred)
        print(f'{model_name} ROC AUC Score:', roc_auc)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')
        plt.show()

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}: AP={average_precision:0.2f}')
    plt.show()
    '''

def main():
    # Load dataset
    iris = load_iris() # use the dataloader to load the dataset we are working on
    X = iris.data # replace this with the dataset we are working on
    y = iris.target # replace this with the target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate Decision Tree model
    dt = DecisionTree()
    dt.train(X_train, y_train)
    y_pred = dt.predict(X_test)
    evaluate_model('Decision Tree', y_test, y_pred)

    # Train and evaluate Naive Bayes model
    nb = NaiveBayes()
    nb.train(X_train, y_train)
    y_pred = nb.predict(X_test)
    evaluate_model('Naive Bayes', y_test, y_pred)

    # Train and evaluate Logistic Regression model
    lr = LogisticRegressionClassifiers()
    lr.train(X_train, y_train)
    y_pred = lr.predict(X_test)
    evaluate_model('Logistic Regression', y_test, y_pred)

    # Train and evaluate SVM model
    svm = SVM()
    svm.train(X_train, y_train)
    y_pred = svm.predict(X_test)
    evaluate_model('SVM', y_test, y_pred)

    # Train and evaluate MLP Classifier model
    #mlp = MLPClassifier()
    mlp = MLPClassifier(max_iter=1000)
    mlp.train(X_train, y_train)
    y_pred = mlp.predict(X_test)
    evaluate_model('MLP Classifier', y_test, y_pred)
    
if __name__ == '__main__':
    main()