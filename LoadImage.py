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
    elif label_name == "Infiltration":
        return "Infiltration"
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

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the linear regression workflow
def perform_linear_regression(images, labels):
    # Ensure consistent feature and target lengths
    feature_length = len(images)
    target_length = len(labels)

    if feature_length != target_length:
        min_length = min(feature_length, target_length)
        images = images[:min_length]
        labels = labels[:min_length]
        print("Adjusted lengths for consistency.")

    # Flatten the image arrays for use in linear regression
    image_arrays = [np.array(img).flatten() for img in images]

    # Convert the target labels to numeric (example: using LabelEncoder)
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        image_arrays, numeric_labels, test_size=0.2, random_state=42
    )

    # Create a linear regression model and fit it
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = lin_reg.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)

    return lin_reg

# Example usage
# Assuming 'images' contains a list of images and 'new_label_names' contains the mapped label names
# Perform linear regression on the dataset
linear_regression_model = perform_linear_regression(images, new_label_names)

