import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
# Define the training and validation datasets
train_dataset = torchvision.datasets.ImageFolder(
    "/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/train",
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
)
val_dataset = torchvision.datasets.ImageFolder(
    "/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/val",
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the ResNet-50 model
model = torchvision.models.resnet50(pretrained=True)
num_classes = 2
model.fc = nn.Linear(2048, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    losses = []
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss
    print(f"Epoch {epoch + 1}/{epochs}: Loss {np.mean(losses):.4f}")

# Evaluate the model on the validation set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {correct / total:.4f}")

# Plot the loss and accuracy
plt.plot(losses)
plt.title("Loss")
plt.show()

# Save the model
torch.save(model.state_dict(), "resnet50.pth")
