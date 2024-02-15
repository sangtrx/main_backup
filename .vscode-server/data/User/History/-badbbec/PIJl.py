from torchvision import models
from PIL import Image
from torchvision.utils import save_image

# ... (other import statements) ...

def save_incorrect_images(phase='test'):
    # Create a directory to save images if not exist
    if not os.path.exists("incorrect_images"):
        os.makedirs("incorrect_images")

    incorrect_predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            incorrect = (preds != labels).nonzero(as_tuple=True)[0]
            incorrect_predictions.extend([(preds[i].item(), labels[i].item()) for i in incorrect.cpu().numpy()])
            for i in incorrect:
                image = inputs[i].cpu().clone()
                image = image.squeeze(0)
                image = transforms.ToPILImage()(image)
                
                incorrect_folder_path = f"incorrect_images/{class_names[preds[i]]}"
                if not os.path.exists(incorrect_folder_path):
                    os.makedirs(incorrect_folder_path)

                image.save(f"{incorrect_folder_path}/{i}.png")
        
    return incorrect_predictions


# Load the saved model
model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

# Here you should replace 'resnet50_cls.pt' with your model's location
model_ft.load_state_dict(torch.load('resnet50_cls.pt'))

incorrect_predictions = save_incorrect_images('test')
