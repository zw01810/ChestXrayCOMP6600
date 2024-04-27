from datasets import load_dataset

# Load the dataset with the correct config and split
dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", name="image-classification", split="train")

# Select a smaller subset
subset = dataset.select(range(100))  # First 100 samples

# Get the labels, handling missing values
labels = [item.get('labels', []) for item in subset]

# Flatten the list of labels to avoid unhashable type errors
flattened_labels = [item for sublist in labels for item in sublist]

# Mapping integer labels to class names
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

# Convert flattened labels to their class names
label_names = [class_labels.get(label, "Unknown") for label in flattened_labels]

# Check available keys to identify any missing keys
print("Available keys:", subset[0].keys())

# Get image paths without triggering KeyErrors
image_paths = [item.get('image_file_path', None) for item in subset]

# Identify any missing image paths
missing_paths = [path for path in image_paths if path is None]
if missing_paths:
    print("Warning: Some image paths are missing")

# Further code to process or analyze the subset
