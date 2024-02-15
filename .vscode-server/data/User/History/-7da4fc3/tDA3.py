import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from matplotlib import pyplot as plt

# Define the data transformations
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets
train_dataset = torchvision.datasets.ImageFolder(
    os.path.join("path/to/train/dataset"), transform=train_transforms
)
val_dataset = torchvision.datasets.ImageFolder(
    os.path.join("path/to/val/dataset"), transform=val_transforms
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model
model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, 2)

# Define the loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    losses = []
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss
    print(f"Epoch {epoch + 1}: loss {sum(losses) / len(losses)}")

# Evaluate the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy: {correct / total}")

# Save the model
torch.save(model.state_dict(), "resnet50.pth")

# Plot the loss and accuracy
losses = [loss.item() for loss in losses]
plt.plot(losses)
plt.title("Loss")
plt.savefig("loss.png")

# Plot the accuracy
accuracy = 100 * correct / total
plt.plot([accuracy])
plt.title("Accuracy")
plt.savefig("accuracy.png")
