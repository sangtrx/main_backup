
# Import the necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

# Define the transform to resize and pad the images
transform = transforms.Compose([
    transforms.Resize((256, -1)), # Resize the shorter edge to 256 and keep the aspect ratio
    transforms.Pad((0, 0, 0, 0), padding_mode='edge'), # Pad the image with the edge values to make it square
    transforms.ToTensor(), # Convert the image to a tensor
])

# Load the datasets for train, val and test
trainset = torchvision.datasets.ImageFolder(root='/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/train', transform=transform)
valset = torchvision.datasets.ImageFolder(root='/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/val', transform=transform)
testset = torchvision.datasets.ImageFolder(root='/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/test', transform=transform)

# Define the batch size and the number of workers
batch_size = 32
num_workers = 4

# Create the data loaders for train, val and test
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Define the device to use (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the resnet 50 model and modify the last layer to match the number of classes (2)
model = torchvision.models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

# Move the model to the device
model.to(device)

# Define the loss function (cross entropy) and the optimizer (SGD with momentum)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Define the number of epochs to train
num_epochs = 10

# Initialize lists to store the losses and accuracies for each epoch
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Loop over the epochs
for epoch in range(num_epochs):

    # Set the model to training mode
    model.train()

    # Initialize variables to store the running loss and correct predictions for train
    running_loss = 0.0
    running_corrects = 0

    # Loop over the batches in trainloader
    for i, data in enumerate(trainloader):

        # Get the inputs and labels
        inputs, labels = data

        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the running loss and corrects
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)

    # Compute the epoch loss and accuracy for train
    epoch_loss = running_loss / len(trainset)
    epoch_accuracy = running_corrects.double() / len(trainset)

    # Append them to the lists
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Print them
    print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to store the running loss and correct predictions for val
    running_loss = 0.0
    running_corrects = 0

    # Loop over the batches in valloader
    for i, data in enumerate(valloader):

        # Get the inputs and labels
        inputs, labels = data

        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass with no gradient computation
        with torch.no_grad():
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Update the running loss and corrects
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)

    # Compute the epoch loss and accuracy for val
    epoch_loss = running_loss / len(valset)
    epoch_accuracy = running_corrects.double() / len(valset)

    # Append them to the lists
    val_losses.append(epoch_loss)
    val_accuracies.append(epoch_accuracy)

    # Print them
    print(f'Epoch {epoch+1}, Val Loss: {epoch_loss:.4f}, Val Accuracy: {epoch_accuracy:.4f}')

# Plot the losses and accuracies for train and val
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epoch')
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Epoch')
plt.tight_layout()

# Save the plot as an image
plt.savefig('loss_accuracy_plot.png')

# Evaluate the model on the test set
model.eval()

# Initialize variables to store the predictions and labels for test
test_preds = []
test_labels = []

# Loop over the batches in testloader
for i, data in enumerate(testloader):

    # Get the inputs and labels
    inputs, labels = data

    # Move the inputs and labels to the device
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Forward pass with no gradient computation
    with torch.no_grad():
        outputs = model(inputs)

        # Get the predictions
        _, preds = torch.max(outputs, 1)

        # Append them to the lists
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Compute the metrics for test
test_precision = metrics.precision_score(test_labels, test_preds)
test_recall = metrics.recall_score(test_labels, test_preds)
test_f1 = metrics.f1_score(test_labels, test_preds)
test_confusion_matrix = metrics.confusion_matrix(test_labels, test_preds)

# Print them
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
print(f'Test Confusion Matrix:\n{test_confusion_matrix}')

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Test Set')

# Save the plot as an image
plt.savefig('confusion_matrix_plot.png')