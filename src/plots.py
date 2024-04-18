import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.base import clone

def plot_graphs(model_name,y_test, predictions, acc,conf_mat, clf_report):
 
    
    # Calculate metrics
    #acc = accuracy_score(y_test, predictions)
    #conf_mat = confusion_matrix(y_test, predictions)
    #clf_report = classification_report(y_test, predictions, output_dict=True)

    
    # Plotting confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    # Plotting accuracy score
    plt.figure(figsize=(5, 3))
    sns.barplot(x=[model_name], y=[acc])
    plt.title(f'Accuracy Score for {model_name}: {acc:.2f}')
    plt.ylabel('Accuracy Score')
    plt.show()
    
    # Plotting precision, recall, f1-score from the classification report
    
    plt.figure(figsize=(10, 5))
    #clf_report = report
    categories = list(clf_report.keys())[:-2]  # exclude 'accuracy' and 'macro avg'
    scores = [clf_report[cat]['f1-score'] for cat in categories]
    sns.barplot(x=categories, y=scores)
    plt.title(f'F1-Scores for each class in {model_name}')
    plt.ylabel('F1 Score')
    plt.show()


def train_and_plot_mlp_metrics(model, X_train, y_train, X_test, y_test, iterations):
    """
    Trains the MLP model and plots loss and accuracy curves.
    
    Args:
    model (MLPClassifier): The scikit-learn MLP model.
    X_train (array): Training features.
    y_train (array): Training labels.
    X_test (array): Test features.
    y_test (array): Test labels.
    iterations (int): Number of training iterations.
    """
    # Clone and configure the model for incremental learning
    local_model = clone(model)
    local_model.set_params(max_iter=1, warm_start=True)
    
    losses = []
    accuracies = []

    # Train the model incrementally and record performance metrics
    for i in range(iterations):
        local_model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        losses.append(local_model.loss_)
        y_pred = local_model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    # Plotting the loss curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss', color='red')
    plt.title('Loss Curve during Training')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plotting the accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='blue')
    plt.title('Accuracy Curve during Training')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
