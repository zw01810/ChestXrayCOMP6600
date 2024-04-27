import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_fscore_support
from torch import load
import os
from joblib import load
from os import path
import pandas as pd

def load_model(model_name, directory='save_trained_models'):
    model_path = path.join(directory, f'{model_name}.joblib')
    if path.exists(model_path):
        model = load(model_path)
        return model
    else:
        print(f"No model found with name {model_name} in directory {directory}")
        return None

def test_model_evaluation(model, dataloader, label_encoder: LabelEncoder, classification_type: str, runtime, device):
    """
    Function to evaluate a model using PyTorch.
    :param model: PyTorch model to be evaluated.
    :param dataloader: DataLoader containing the test dataset.
    :param label_encoder: The label encoder for y value (label).
    :param classification_type: The classification type. Ex: N-B-M: normal, benign and malignant; B-M: benign and malignant.
    :param runtime: Runtime in seconds.
    :param device: The device to run the model on (e.g., 'cuda' or 'cpu').
    :return: None.
    """
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Inverse transform to original labels if needed
            all_predictions.extend(preds.gpu().numpy())
            all_true_labels.extend(labels.gpu().numpy())

    # Handle the label transformation for different classification types
    if len(label_encoder.classes_) == 2:
        y_true_inv = np.array(all_true_labels)
        y_pred_inv = np.array(all_predictions)
    else:
        y_true_inv = label_encoder.inverse_transform(all_true_labels)
        y_pred_inv = label_encoder.inverse_transform(all_predictions)

    # Calculate accuracy
    accuracy = accuracy_score(y_true_inv, y_pred_inv)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Save the model evaluation results
    model_name = model.__class__.__name__
    save_path = path.join('model_evaluation_results', f'{model_name}_evaluation.txt')
    with open(save_path, 'w') as f:
        f.write(f'Test Accuracy: {accuracy:.4f}\n')
        f.write(f'Runtime: {runtime:.2f} seconds\n')
        f.write(f'Classification Type: {classification_type}\n')
        f.write(f'Confusion Matrix:\n')
        f.write(f'{confusion_matrix(y_true_inv, y_pred_inv)}\n')
        f.write(f'Classification Report:\n')
        f.write(f'{classification_report(y_true_inv, y_pred_inv)}\n')
    print(f'Model evaluation results saved at {save_path}')
    
    
def test_model(model_name, test_data, label_encoder, classification_type, runtime, device):
    """
    Function to test a model using PyTorch.
    :param model_name: The name of the model to be tested.
    :param test_data: The test data.
    :param label_encoder: The label encoder for y value (label).
    :param classification_type: The classification type. Ex: N-B-M: normal, benign and malignant; B-M: benign and malignant.
    :param runtime: Runtime in seconds.
    :param device: The device to run the model on (e.g., 'cuda' or 'cpu').
    :return: None.
    """
    
    model = load_model(model_name)
    if model is not None:
        test_model_evaluation(model, test_data, label_encoder, classification_type, runtime, device)
    else:
        print(f"Model {model_name} not found.") 
        
def main():
    # Set the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the label encoder
    label_encoder = load(r'saved_labels/label_encoder.joblib')

    # Load the test data
    test_data = pd.read_csv(r'Data_csv/data_test_list.txt', sep='\t')

    # Test the model
    test_model('MLP Classifier', test_data, label_encoder, 'B-M', 0.5, device)
    
if __name__ == '__main__':
    main()
    