from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    return accuracy, conf_matrix, report


def evaluate_naive_bayes(model, X_test, y_test):
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test) 
    print(f"proba shape: {proba.shape}, unique classes in y_test: {np.unique(y_test)}")
    accuracy = accuracy_score(y_test, predictions)
    unique_classes = np.unique(y_test)
    # Map the unique classes to a range starting from 0
    class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
    y_test_mapped = np.array([class_mapping[cls] for cls in y_test])
    # Transform y_test into a binary matrix
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test_mapped)
    # Only keep the columns in proba that correspond to the classes in y_test
    proba = proba[:, unique_classes]
    # Compute the ROC AUC score for each class separately
    roc_auc = roc_auc_score(y_test_bin, proba, multi_class='ovr', average='macro')
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    return accuracy, roc_auc, conf_matrix, report


def evaluate_logistic_regression(model, X_test, y_test):    
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    if len(np.unique(y_test)) == 2:  # Binary classification
        roc_auc = roc_auc_score(y_test, proba[:, 1])
    else:
        roc_auc = "Not applicable for multiclass"
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    return accuracy, roc_auc, conf_matrix, report


def evaluate_svm(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    return accuracy, conf_matrix, report


def evaluate_mlp(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    
    return accuracy, conf_matrix, report
