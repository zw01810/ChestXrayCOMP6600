import os
import pandas as pd
from PIL import Image
from Models.models import *
from sklearn.datasets import make_classification
from torchvision.transforms import ToTensor
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from models_evaluation import *
from plots import *
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import joblib

'''

def load_images_and_labels(csv_file, root_dir):
    data = pd.read_csv(csv_file)
    images = []
    labels = data['labels'].values  # Adjust the column name if different

    for img_name in data['Image Index']:  # Adjust the column name if different
        img_path = os.path.join(root_dir, img_name)
        with Image.open(img_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((64, 64))  # Resize to manage size
            img_array = np.array(img).flatten()  # Flatten the 2D image into 1D
            images.append(img_array)

    return np.array(images), labels

root_dir = 'Chest X-ray Images_extracted'
csv_file = 'Data_csv\exploded_data_with_labels.csv'

X, y = load_images_and_labels(csv_file, root_dir)
print(X, y)

'''

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.to_tensor = ToTensor()  # Transform to convert images to tensor

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        label = self.image_frame.iloc[idx, 3]  # Adjust the column index if different
        image = self.to_tensor(image)

        return image, label

# Load the data
root_dir = 'Chest X-ray Images_extracted'
dataset = ImageDataset(csv_file='Data_csv/exploded_data_with_labels.csv', root_dir=root_dir)
dataloader = DataLoader(dataset = dataset, batch_size=32, shuffle=True)
images, labels = next(iter(dataloader))
#print(len(labels))
#print(labels)
label_encoder = LabelEncoder()
#images, labels = load_images(data, image_folder)
labels = label_encoder.fit_transform(labels)
# Save the encoder
joblib.dump(labels, 'saved_labels/label_encoder.joblib', compress=True)
#print(len(labels))
#print(images.shape, labels.shape)
# Split data into training and testing sets
images, labels = make_classification(n_samples=112120,n_informative=5, n_features=20, n_classes=15, random_state=1)
#print(images, labels)
train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=42)
#print(images, labels)
#print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
#print(len(images), len(labels))
#print(np.unique(labels))

# Reshape test_x into a 2D array
test_x_2d = test_x.reshape(test_x.shape[0], -1)
# Reshape train_x into a 2D array
train_x_2d = train_x.reshape(train_x.shape[0], -1)

# Define the training function  
def train_model(model, X_train, y_train):
    #print(model)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_x_2d)
    X_test_scaled = scaler.transform(test_x_2d)

    if model == 'MLP':
        print("Training MLP...")

        # Initialize and train MLP
        mlp = MLP(hidden_layer_sizes=(100, 50), max_iter=500, warm_start=True, activation='relu',
                            solver='adam', verbose=False, random_state=42, learning_rate_init=.001)
        
        mlp.train(X_train_scaled, train_y, X_test_scaled, test_y )
        test_preds = mlp.predict(X_test_scaled)  
        
        # Evaluate the mlp model
        accuracy, conf_matrix, report, precision, recall, f1, support = evaluate_mlp(mlp, X_test_scaled, test_y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
        print(f'Support: {support}')
        df = pd.DataFrame(report).transpose()
        print(df)
        print(conf_matrix)
        plot_graphs('MLP', test_y, test_preds, accuracy, conf_matrix, report)
        mlp.plot_loss_and_accuracy()
        save_model({'MLP Classifier': mlp})
        
    elif model == 'Decision_Tree':
        print("Training Decision Tree...")
        # Create and train Decision Tree
        tree = DecisionTreeModel(criterion='gini', max_depth=3, random_state=42)
        tree.train(X_train_scaled, train_y)
        test_preds = tree.predict(X_test_scaled)  

        # Evaluate the model
        accuracy, conf_matrix, report = evaluate_model(tree, X_test_scaled, test_y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        df = pd.DataFrame(report).transpose()
        print(df)
        print(conf_matrix)
        plot_graphs('Decision Tree', test_y, test_preds, accuracy, conf_matrix, report)
        save_model({'Decision Tree Classifier': tree})
        
    elif model == 'Naive_Bayes_classifier':
        print("Training Naive Bayes...")
        naive_bayes = NaiveBayesModel()
        naive_bayes.train(X_train_scaled, train_y)
        test_preds = naive_bayes.predict(X_test_scaled)
        # Evaluate the model
        accuracy, roc_auc, conf_matrix, report = evaluate_naive_bayes(naive_bayes, X_test_scaled, test_y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"ROC AUC: {float(roc_auc):.2f}")
        print(report)
        print(conf_matrix)
        plot_graphs('Naive Bayes', test_y, test_preds, accuracy, conf_matrix, report)
        save_model({'Naive Bayes Classifier': naive_bayes})
        
    elif model == 'Logistic_Regression':
        print("Training Logistic Regression...")
        # Create and train Logistic Regression
        logistic_model = LogisticRegressionModel(max_iter=200)
        logistic_model.train(X_train_scaled, train_y)
        test_preds = logistic_model.predict(X_test_scaled)

        # Evaluate the model
        accuracy, roc_auc, conf_matrix, report = evaluate_logistic_regression(logistic_model, X_test_scaled, test_y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        if isinstance(roc_auc, str):
            print(f"ROC AUC: {roc_auc}")
        else:
            print(f"ROC AUC: {float(roc_auc):.2f}")
        print(report)
        print(conf_matrix)
        plot_graphs('Logistic Regression', test_y, test_preds, accuracy, conf_matrix, report)
        # Save the model
        save_model({'Logistic Regression Classifier': logistic_model})
    
    elif model == 'SVM':
        print("Training SVM...")
        # Initialize and train SVM
        svm = SVMModel(kernel='linear', C=1.0, random_state=42)
        svm.fit(X_train_scaled, y_train)
        test_preds = svm.predict(X_test_scaled) 
        # Evaluate the model
        accuracy, conf_matrix, report = evaluate_svm(svm, X_test_scaled, test_y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        df = pd.DataFrame(report).transpose()
        print(df)
        print(conf_matrix)
        plot_graphs('SVM', test_y, test_preds, accuracy, conf_matrix, report)
        
        plot_learning_curve_svm(svm, "Learning Curve for SVM Model", X_train_scaled, y_train, cv=5, n_jobs=-1)
        plot_accuracy_curve_svm(svm, X_test_scaled, test_y, cv=5)
        #cv_scores = svm.cross_val_score(report, X_test_scaled, test_y, cv=5)
        # Save the model
        save_model({'SVM Classifier': svm})
    
    elif model == 'RandomForestModel':
        print("Decision Tree with Random Tree Classifier...")
        # Create and train Random Forest
        forest_model = RandomForestModel(n_estimators=100, max_depth=3, random_state=42)
        forest_model.train(X_train_scaled, train_y)
        test_preds = forest_model.predict(X_test_scaled)  
        # Evaluate the model
        accuracy, conf_matrix, report = evaluate_model(forest_model, X_test_scaled, test_y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(report)
        print(conf_matrix)
        plot_graphs('Random Forest', test_y, test_preds, accuracy, conf_matrix, report)
        save_model({'Random Forest': forest_model})
        
    elif model == 'DecisionTreeClassifier':
        print("Training Decision Tree with GridSearch...")
        dt = DecisionTreeClassifierModel()
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=dt.model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(X_train_scaled, train_y)

        # Best model evaluation
        best_dt = grid_search.best_estimator_
        y_pred = best_dt.predict(X_test_scaled)
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
        print("Test set score: {:.2f}".format(accuracy_score(test_y, y_pred)))
        print(classification_report(test_y, y_pred))
        plot_graphs('Decision Tree Classifier', test_y, y_pred, accuracy_score(test_y, y_pred), confusion_matrix(test_y, y_pred), classification_report(test_y, y_pred, output_dict=True))
        save_model({'Decision Tree Classifier': best_dt})
        
def save_model(model, directory='save_trained_models'):
    from joblib import dump
    from os import path
    import os
    if not path.exists(directory):
        os.makedirs(directory)
    for n, m in model.items():
        save_path = path.join(directory, f'{n}.joblib')
        dump(m, save_path)


