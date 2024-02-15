import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from sklearn.model_selection import KFold

# Data transformation
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define number of folds
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle = False)

# Dataset loading
data_dir = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification_sync_5fold'
image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform)
dataloaders = {"test": torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transform), batch_size=128, shuffle=False, num_workers=4)}

class_names = image_datasets.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Initialize the model
# model_ft = models.resnet50(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
# model_ft = model_ft.to(device)
# criterion = nn.CrossEntropyLoss()
def plot_training(train_loss, val_loss, train_acc, val_acc):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Losses')

    plt.subplot(1, 2, 2)
    plt.plot([acc.cpu().numpy() for acc in train_acc], label='Training Accuracy')
    plt.plot([acc.cpu().numpy() for acc in val_acc], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracies')

    plt.tight_layout()
    plt.savefig('training_plot_sync.png')




def evaluate_model(model, phase):
    model.eval()

    running_corrects = 0
    preds_list = []
    labels_list = []

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    print('{} Acc: {:.4f}'.format(phase, epoch_acc))

    precision = precision_score(labels_list, preds_list, average='weighted')
    recall = recall_score(labels_list, preds_list, average='weighted')
    f1 = f1_score(labels_list, preds_list, average='weighted')
    cm = confusion_matrix(labels_list, preds_list)

    print('{} Precision: {:.4f}'.format(phase, precision))
    print('{} Recall: {:.4f}'.format(phase, recall))
    print('{} F1 score: {:.4f}'.format(phase, f1))
    print('{} Confusion Matrix:\n {}'.format(phase, cm))
# K-Fold Cross Validation
results = []
for fold, (train_ids, val_ids) in enumerate(kf.split(image_datasets)):
    print(f"FOLD {fold}")
    print("--------------------------------")

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    dataloaders["train"] = torch.utils.data.DataLoader(
                          image_datasets, 
                          batch_size=128, sampler=train_subsampler)
    dataloaders["val"] = torch.utils.data.DataLoader(
                          image_datasets,
                          batch_size=128, sampler=val_subsampler)

    # Initialize the model for this fold
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train and evaluate for this fold
    _, train_loss, val_loss, train_acc, val_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
    plot_training(train_loss, val_loss, train_acc, val_acc)
    evaluate_model(model_ft, 'val')
    results.append(evaluate_model(model_ft, 'test'))

    # Save the model for this fold
    PATH = f"resnet50_cls_sync_fold_{fold}.pt"
    torch.save(model_ft.state_dict(), PATH)

# Print fold results
print(f"K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS")
print("----------------------------------------------------------------")
sum = 0.0
for i, result in enumerate(results):
    print(f"Fold {i}: {result} %")
    sum += result
print(f"Average: {sum/len(results)} %")