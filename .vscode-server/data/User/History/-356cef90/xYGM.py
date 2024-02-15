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

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification_pad'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)






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

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)


evaluate_model(model_ft, 'val')
evaluate_model(model_ft, 'test')


# Specify a path
PATH = "resnet50_cls_pad.pt"

# Save
torch.save(model_ft.state_dict(), PATH)