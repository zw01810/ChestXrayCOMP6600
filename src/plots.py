import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

def plot_graphs(model_name,y_test, predictions, acc,conf_mat, report):
    
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
    
    # F1-Score
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    if not isinstance(report, dict):
        print(f'Error: report should be a dictionary, but got {type(report)}')
        return
    categories = [cat for cat in report.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']]
    scores = [report[cat]['f1-score'] for cat in categories]
    sns.barplot(x=categories, y=scores)
    plt.title(f'F1-Scores for each class in {model_name}')
    plt.ylabel('F1 Score')
    
    # Recall
    plt.subplot(2, 2, 2)
    if not isinstance(report, dict):
        print(f'Error: report should be a dictionary, but got {type(report)}')
        return
    categories = [cat for cat in report.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']]
    scores = [report[cat]['recall'] for cat in categories]
    sns.barplot(x=categories, y=scores)
    plt.title(f'Recall for each class in {model_name}')
    plt.ylabel('Recall')
    
    # Precision
    plt.subplot(2, 2, 3)
    if not isinstance(report, dict):
        print(f'Error: report should be a dictionary, but got {type(report)}')
        return
    categories = [cat for cat in report.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']]
    scores = [report[cat]['precision'] for cat in categories]
    sns.barplot(x=categories, y=scores)
    plt.title(f'Precision for each class in {model_name}')
    plt.ylabel('Precision')

    # Support
    plt.subplot(2, 2, 4)
    if not isinstance(report, dict):
        print(f'Error: report should be a dictionary, but got {type(report)}')
        return
    scores = [report[cat]['support'] for cat in categories]
    sns.barplot(x=categories, y=scores)
    plt.title(f'Support for each class in {model_name}')
    plt.ylabel('Support')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


def plot_learning_curve_svm(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples)
        Target relative to X for classification or regression;

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum y-values plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validator.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        -1 means using all processors.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to generate the learning curve.
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def plot_accuracy_curve_svm(model, X, y, cv=5):
    
    from sklearn import svm
    cv_scores = cross_val_score(model, X, y, cv=cv)

    # Plot accuracy curve
    plt.figure()
    plt.plot(range(1, cv+1), cv_scores)
    plt.title('Accuracy Curve')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.show()