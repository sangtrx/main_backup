import cv2
import numpy as np
import glob
from ultralytics import YOLO
import os 


###################### resnet
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

from torchvision.transforms.functional import to_tensor, normalize

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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from PIL import Image

# def preprocess_image(image: np.array, shift_pixels: int = 0) -> np.array:
#     # Convert the image to PIL Image
#     image = Image.fromarray(image)

#     # Resize the image
#     aspect_ratio = image.size[0] / image.size[1]
#     if aspect_ratio > 1:
#         # width is greater than height
#         new_image = image.resize((256, int(256 / aspect_ratio)))
#     else:
#         # height is greater than width
#         new_image = image.resize((int(256 * aspect_ratio), 256))

#     # Create a black 256x256 image
#     black_image = Image.new('RGB', (256, 256))

#     # Compute the position where the image should be pasted
#     paste_position = ((black_image.size[0] - new_image.size[0]) // 2 + shift_pixels,
#                       (black_image.size[1] - new_image.size[1]) // 2)

#     # Paste the image
#     black_image.paste(new_image, paste_position)

#     return np.array(black_image)

def preprocess_image(image: np.array, shift_pixels: int = 0) -> np.array:
    # No need to convert to PIL Image here
    # Resize the image
    aspect_ratio = image.shape[1] / image.shape[0] # Note the change in ordering because we are using numpy array now
    if aspect_ratio > 1:
        # width is greater than height
        new_image = cv2.resize(image, (256, int(256 / aspect_ratio)))
    else:
        # height is greater than width
        new_image = cv2.resize(image, (int(256 * aspect_ratio), 256))

    # Create a black 256x256 image
    black_image = np.zeros((256, 256, 3), dtype=np.uint8)

    # Compute the position where the image should be pasted
    paste_position = ((black_image.shape[1] - new_image.shape[1]) // 2 + shift_pixels,
                      (black_image.shape[0] - new_image.shape[0]) // 2)

    # Paste the image
    black_image[paste_position[1]:paste_position[1] + new_image.shape[0],
                paste_position[0]:paste_position[0] + new_image.shape[1]] = new_image

    return black_image

######################
def process_video(model, video_path, output_path, area):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1

        if not success:
            break

        output_frame = frame.copy()

        # Define the bounding rectangle area
        x, y, x2, y2 = area

        # Crop the frame to area
        cropped_frame = frame[y:y2, x:x2]

        # Perform inference on the area
        results = model.predict(cropped_frame, conf=0.8)

        # Iterate through the detections
        for i, (box, score, class_) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
            box_x1, box_y1, box_x2, box_y2 = box.tolist()
            # Check if the box is completely inside the area
            if (box_x1 >= 5 and box_x2 <= x2-x-5): # Only need to check 2left n right vertical line | and (box_y1 >= 5 and box_y2 <= y2-y-5): # Adjust area coordinates back to original frame
                # crop the box then feed to resnet model to do classification
                # Crop the bounding box from the frame
                cropped_box = cropped_frame[int(box_y1):int(box_y2), int(box_x1):int(box_x2)]
                
                # Prepare the image for input to the ResNet model
                # img = cv2.cvtColor(cropped_box, cv2.COLOR_BGR2RGB)
                # img = preprocess_image(cropped_box)
                # cv2.imwrite('debug.png', img)

                # img = transforms.ToPILImage()(img)
                # img = data_transforms['val'](img)
                # img = img.unsqueeze(0)  # add batch dimension
                # img = img.to(device)

                # img = cv2.cvtColor(cropped_box, cv2.COLOR_BGR2RGB)
                img1 = preprocess_image(cropped_box)
                cv2.imwrite('img1.png',img1)
                img2 = cv2.imread('img1.png')
                
                img = to_tensor(img2)  # This does the same thing as `transforms.ToPILImage()` and `transforms.ToTensor()`
                img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensor
                img = img.unsqueeze(0)  # Add batch dimension
                img = img.to(device)

                

                # Pass the cropped box to the classification model for prediction
                classification_model.eval()  # Set to evaluation mode
                with torch.no_grad():
                    output = classification_model(img)
                    _, preds = torch.max(output, 1)

                
                # Get class prediction
                predicted_class = class_names[preds]
                print(predicted_class)
                import pdb;pdb.set_trace()

                color = (255, 0, 0) if predicted_class == 'defect' else (0, 0, 255)
                cv2.rectangle(output_frame, (int(box_x1+x), int(box_y1+y)), (int(box_x2+x), int(box_y2+y)), color, 2)
                # Display text on top left of the box
                cv2.putText(output_frame, predicted_class, (int(box_x1+x), int(box_y1+y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Draw the bounding rectangle area
        color = (0, 255, 0) 
        cv2.rectangle(output_frame, (x, y), (x2, y2), color, 2)

        # write frame
        out.write(output_frame)
    cap.release()
    out.release()

model_path = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/runs/detect/yolov8x_brio_det_only/weights/best.pt'
model = YOLO(model_path)

# Define the areas for each group
areas = {
    '1': [711, 391, 711+420, 391+420],
    '2-12': [510,389,510+600,389+600], # replace with actual area  656,420,541,615
    '13-29': [438,228,438+750,228+750], # replace with actual area
    '30': [589,451,589+550,451+550], # replace with actual area
    '31-40': [909,927,909+1200,927+1200] # replace with actual area
}

# Get all video files
video_files = glob.glob('/mnt/tqsang/dot1_vid/dot1_*.mp4')

# Group video files
video_groups = {
    '1': [],
    '2-12': [],
    '13-29': [],
    '30': [],
    '31-40': []
}

# Create output directories if they don't exist
output_dirs = ['/mnt/tqsang/test_vid']
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

for video_file in video_files:
    
    # if video_file != '/mnt/tqsang/dot1_vid/dot1_1.mp4':
    #     continue
    # print(video_file.split('_')[2].split('.')[0])
    video_number = int(video_file.split('_')[2].split('.')[0])
    if video_number == 1:
        video_groups['1'].append(video_file)
    elif 2 <= video_number <= 12:
        video_groups['2-12'].append(video_file)
    elif 13 <= video_number <= 29:
        video_groups['13-29'].append(video_file)
    elif video_number == 30:
        video_groups['30'].append(video_file)
    elif 31 <= video_number <= 40:
        video_groups['31-40'].append(video_file)




# Load the trained ResNet model
classification_model = models.resnet50(pretrained=False)
num_ftrs = classification_model.fc.in_features
classification_model.fc = nn.Linear(num_ftrs, 2)
classification_model = classification_model.to(device)
classification_model.load_state_dict(torch.load("resnet50_cls_sync.pt"))


# Process each video
for group, videos in video_groups.items():
    for video in videos:
        if int(video.split('_')[2].split('.')[0]) in [3,4,7,8,9,40]: #test videos
            print(video)
            process_video(model, video, f'/mnt/tqsang/test_vid/{video.split("/")[-1]}', areas[group])
