from datasets import load_dataset

# Load the dataset with trust_remote_code to avoid FutureWarning
dataset = load_dataset(
    "alkzar90/NIH-Chest-X-ray-dataset",
    name="image-classification",
    split="train",
    trust_remote_code=True
)

# Select a smaller subset
subset = dataset.select(range(100))  # First 100 samples

# Check available keys
print("Available keys:", subset[0].keys())

# Since 'image' is available, try to access the image data
# This accesses the actual image data, which may be a Pillow Image object
images = [item['image'] for item in subset if 'image' in item]

# Check if images were successfully retrieved
if not images:
    print("Error: No valid images found")
else:
    print("Successfully retrieved images.")

# Flatten and map labels to their corresponding class names
labels = [item.get('labels', []) for item in subset]
flattened_labels = [item for sublist in labels for item in sublist]

class_labels = {
    0: "No Finding",
    1: "Atelectasis",
    2: "Cardiomegaly",
    3: "Effusion",
    4: "Infiltration",
    5: "Mass",
    6: "Nodule",
    7: "Pneumonia",
    8: "Pneumothorax",
    9: "Consolidation",
    10: "Edema",
    11: "Emphysema",
    12: "Fibrosis",
    13: "Pleural Thickening",
    14: "Hernia",
}

label_names = [class_labels.get(label, "Unknown") for label in flattened_labels]

# Additional code to work with the retrieved images and labels
from PIL import Image

# Example: Convert images to grayscale and resize
processed_images = [img.convert("L").resize((256, 256)) for img in images]

import matplotlib.pyplot as plt

# Display the first 5 images
for i, img in enumerate(processed_images[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
#plt.show()

# Display the first few label names
#print("First 5 label names:", label_names[:50])

def categorize_label(label_name):
    if label_name == "No Finding":
        return "No Finding"
    elif label_name == "Pneumonia":
        return "Pneumonia"
    else:
        return "other"

new_label_names = [categorize_label(label) for label in label_names]

#print("First 5 label names:", new_label_names[:50])

import numpy as np

# Convert images to NumPy arrays for ML
image_arrays = [np.array(img) for img in processed_images]

from sklearn.preprocessing import LabelBinarizer

# Convert class names to one-hot encoding
lb = LabelBinarizer()
one_hot_labels = lb.fit_transform(label_names)



#=======================================================================================================\
#this is where im testing linear regression
#=======================================================================================================\

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data with a clear linear relationship
np.random.seed(0)
X = np.linspace(0, 10, 100)  # Independent variable
y = 2 * X + 3 + np.random.normal(0, 1, 100)  # Dependent variable with added noise

# Convert to a DataFrame
data = pd.DataFrame({'X': X, 'y': y})

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['y'], test_size=0.2, random_state=0)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot the data and the linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Regression Line')
plt.legend()
plt.show()

# Plot residuals
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(X_test, residuals, color='purple', label='Residuals')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.legend()
plt.show()
