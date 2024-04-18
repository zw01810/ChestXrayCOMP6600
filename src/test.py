import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import load
from os import path
import os


def load_model(model_name, model_class, folder_path = 'save_trained_models'):

    try:
        # Instantiate the model class
        model = model_class()
        # Define the path to the model file
        model_path = os.path.join(folder_path, f'{model_name}.th')
        # Load the model state dictionary
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: No model file found at {model_path}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

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
            all_predictions.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

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