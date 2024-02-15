import shutil
import matplotlib.image as mpimg

# Load the saved model
PATH = "resnet50_cls.pt"
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(PATH))
model = model.to(device)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def eval_and_save_incorrect_images(model, phase, root_save_dir):
    model.eval()

    running_corrects = 0
    incorrect_image_data = []

    # Iterate over the test data
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

        # Save the inputs and predictions where the model got wrong
        for i in range(inputs.size()[0]):
            if preds[i] != labels[i]:
                incorrect_image_data.append((inputs[i].cpu(), preds[i], labels[i]))

    epoch_acc = running_corrects.double() / dataset_sizes[phase]
    print('{} Acc: {:.4f}'.format(phase, epoch_acc))

    # Now save the incorrect images
    for i, (img, pred, true) in enumerate(incorrect_image_data):
        # Convert image from tensor to numpy array
        img = img.numpy().transpose((1, 2, 0))
        img = np.clip(img, 0, 1)

        # Prepare saving path
        save_dir = os.path.join(root_save_dir, 'predicted_{}_true_{}'.format(class_names[pred], class_names[true]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'incorrect_image_{}.png'.format(i))

        # Save the image
        mpimg.imsave(save_path, img)

# Call the evaluation function
root_save_dir = '/home/tqsang/incorrect_images'
eval_and_save_incorrect_images(model, 'test', root_save_dir)
