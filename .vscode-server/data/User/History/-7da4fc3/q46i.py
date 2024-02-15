import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Set the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformations to be applied to the images
transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the ResNet50 model
model = torchvision.models.resnet50(pretrained=True)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load the train, val, and test datasets
train_dataset = torchvision.datasets.ImageFolder("/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/train", transform_loader=transforms)
val_dataset = torchvision.datasets.ImageFolder("/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/val", transform_loader=transforms)
test_dataset = torchvision.datasets.ImageFolder("/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification/test", transform_loader=transforms)

# Create dataloaders for the train, val, and test datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

# Train the model
for epoch in range(10):
    losses = []
    accuracies = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the losses and accuracies
        losses.append(loss.item())
        accuracies.append(torch.mean((outputs.argmax(1) == labels).float()).item())

    # Plot the losses and accuracies
    plt.plot(losses)
    plt.plot(accuracies)
    plt.savefig("losses_and_accuracies.png")

    # Evaluate the model on the val and test sets
    val_loss, val_accuracy = evaluate(model, val_loader)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("Val loss: {:.4f}, val accuracy: {:.4f}".format(val_loss, val_accuracy))
    print("Test loss: {:.4f}, test accuracy: {:.4f}".format(test_loss, test_accuracy))

def evaluate(model, dataloader):
    losses = []
    accuracies = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track the losses and accuracies
            losses.append(loss.item())
            accuracies.append(torch.mean((outputs.argmax(1) == labels).float()).item())

    return np.mean(losses), np.mean(accuracies)