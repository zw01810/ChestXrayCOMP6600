import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np

# Define a transform to preprocess the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a manageable size
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images
])

# Load the dataset from Hugging Face
dataset = load_dataset('alkzar90/NIH-Chest-X-ray-dataset')

# New labels mapping
new_labels_map = {'No Finding': 0, 'Infiltration': 1, 'Other': 2}

# Function to map the original labels to the new categories
def map_labels_to_new_categories(label):
    if label == 'No Finding':
        return new_labels_map['No Finding']
    elif label == 'Infiltration':
        return new_labels_map['Infiltration']
    else:
        return new_labels_map['Other']

# Apply the new label mapping to the dataset
train_labels = torch.tensor([map_labels_to_new_categories(label) for label in dataset['train']['label']])

# Assuming the dataset has a split 'train'
train_data = dataset['train'].map(lambda x: transform(x['image']), batched=True)

# DataLoader
train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=64, shuffle=True)

# Logistic Regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# Model initialization
input_size = 128 * 128
num_classes = 3  # 'No Finding', 'Infiltration', 'Other'
model = LogisticRegression(input_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Save the model
torch.save(model.state_dict(), 'logistic_regression_model.pth')

# Load the model for evaluation
model.load_state_dict(torch.load('logistic_regression_model.pth'))
model.eval()  # Set the model to evaluation mode

# Assume you have a test dataset with the same transformation applied
# Create test_loader similar to train_loader
# ...

# Calculate accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test dataset: {accuracy:.2f}%')
