import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np

# Define a transform to resize and flatten the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a manageable size
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images
])

# Load the dataset from Hugging Face
dataset = load_dataset('alkzar90/NIH-Chest-X-ray-dataset')

# Assuming the dataset has a split 'train'
train_data = dataset['train'].map(lambda x: transform(x['image']), batched=True)
train_labels = torch.tensor(np.array(dataset['train']['label']))

# DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(128*128, 512)  # Adjust input size according to your resize operation
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, len(np.unique(train_labels)))  # Output layer size = number of classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
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

# Assuming you have the same transform defined for preprocessing
test_dataset = ImageFolder(root='path_to_test_dataset', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No need to track gradients
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test dataset: {accuracy:.2f}%')



