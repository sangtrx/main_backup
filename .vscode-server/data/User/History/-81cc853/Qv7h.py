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

from collections import Counter



# Global counter for normal and defect chickens
defect_count = 0
normal_count = 0

# Dictionary to keep track of each chicken
chicken_dict = {}

data_dir = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification_pad'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from PIL import Image

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
    global defect_count, normal_count, chicken_dict
    # Define a counter for unique chicken identifiers
    chicken_id_counter = 0
    hit_left = False
    hit_right = False
    chicken_dict = {}
    defect_count = 0
    normal_count = 0
    featheravg_count = 0

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

                # Draw counters on the top of area box



        # Define the bounding rectangle area
        x, y, x2, y2 = area

        # Crop the frame to area
        cropped_frame = frame[y:y2, x:x2]

        # Perform inference on the area
        results = model.predict(cropped_frame, conf=0.8)


        cv2.putText(output_frame, f"Avg feathers count: {round(featheravg_count, 2)}", (x, y-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(output_frame, f"Defect count: {defect_count}", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(output_frame, f"Normal count: {normal_count}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        # Iterate through the detections
        for i, (box, score, class_) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
            box_x1, box_y1, box_x2, box_y2 = box.tolist()
            # Check if the box is completely inside the area
            if (box_x1 >= 5 and box_x2 <= x2-x-5): # Only need to check 2left n right vertical line 

                ## feather detection
                # Calculate the smallest square area that contains the bounding box
                min_side_length = max(box_x2 - box_x1, box_y2 - box_y1)
                center_x = (box_x1 + box_x2) / 2
                center_y = (box_y1 + box_y2) / 2
                square_x1 = int(center_x - min_side_length / 2)
                square_y1 = int(center_y - min_side_length / 2)
                square_x2 = int(center_x + min_side_length / 2)
                square_y2 = int(center_y + min_side_length / 2)

                # Adjust coordinates to stay within the valid range of the cropped_frame
                if square_x1 < 0:
                    square_x2 -= square_x1
                    square_x1 = 0
                if square_y1 < 0:
                    square_y2 -= square_y1
                    square_y1 = 0
                if square_x2 > cropped_frame.shape[1]:
                    square_x1 -= (square_x2 - cropped_frame.shape[1])
                    square_x2 = cropped_frame.shape[1]
                if square_y2 > cropped_frame.shape[0]:
                    square_y1 -= (square_y2 - cropped_frame.shape[0])
                    square_y2 = cropped_frame.shape[0]


                # Crop the smallest square area from the frame
                cropped_square = cropped_frame[square_y1:square_y2, square_x1:square_x2]

                
                # Save the cropped square as an image
                cv2.imwrite('cropped_square.png', cropped_square)

                # Perform inference on the smallest square area for feather detection
                results_feather = model_feather.predict(cropped_square, conf=0.8)

                # Count all the feathers that in the box_x1, box_y1, box_x2, box_y2 area
                feather_count = sum(1 for i, (feather_box, feather_score, feather_class) in 
                                        enumerate(zip(results_feather[0].boxes.xyxy, 
                                                    results_feather[0].boxes.conf, 
                                                    results_feather[0].boxes.cls)) 
                                        if feather_box[0] + square_x1  >= box_x1 and feather_box[1] + square_y1 >= box_y1 and 
                                        feather_box[2] + square_x1 <= box_x2 and feather_box[3] + square_y1 <= box_y2)
                
                # Draw bounding box for each feather in results_feather inside the area
                for i, (feather_box, feather_score, feather_class) in enumerate(zip(results_feather[0].boxes.xyxy, 
                                                                                    results_feather[0].boxes.conf, 
                                                                                    results_feather[0].boxes.cls)):
                    if feather_box[0] >= box_x1 and feather_box[1] >= box_y1 and feather_box[2] <= box_x2 and feather_box[3] <= box_y2:
                        # Draw the bounding box for each feather
                        feather_box_x1, feather_box_y1, feather_box_x2, feather_box_y2 = feather_box.tolist()
                        cv2.rectangle(output_frame, 
                                    (int(feather_box_x1+x + square_x1), int(feather_box_y1+y + square_y1 )), 
                                    (int(feather_box_x2+x + square_x1), int(feather_box_y2+y + square_y1)), 
                                    (0, 255, 100), 2)  

                # crop the box then feed to resnet model to do classification
                # # Crop the bounding box from the frame
                # cropped_box = cropped_frame[int(box_y1):int(box_y2), int(box_x1):int(box_x2)]
                
                # # Prepare the image for input to the ResNet model
                # img = preprocess_image(cropped_box)
                # img = to_tensor(img)  # This does the same thing as `transforms.ToPILImage()` and `transforms.ToTensor()`
                # img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensor
                # img = img.unsqueeze(0)  # Add batch dimension
                # img = img.to(device)

                # Pass the cropped box to the classification model for prediction
                # classification_model.eval()  # Set to evaluation mode
                # with torch.no_grad():
                #     output = classification_model(img)
                #     _, preds = torch.max(output, 1)

                
                # Get class prediction
                if feather_count >= 1:
                    predicted_class = 'defect'
                else:
                    predicted_class = 'normal'
                # predicted_class = class_names[preds]

                # Check if chicken is crossing the area of left line
                if    20 >= box_x1 >= 5:
                    if hit_left == False:
                        hit_left = True  
                        hit_right = False                  
                        # Create a unique chicken id
                        chicken_id = f"{chicken_id_counter}"
                        chicken_id_counter += 1

                        chicken_dict[chicken_id] = {"classifications": [predicted_class], "feather_count":[feather_count]}

                # If chicken already crossed the left line, continue to update its classifications
                if  hit_left == True and hit_right == False:
                        chicken_dict[chicken_id]["classifications"].append(predicted_class)
                        chicken_dict[chicken_id]["feather_count"].append(feather_count)

                # If chicken crossed the area of right line
                if    x2-x-20<= box_x2 <= x2-x-5:
                    if hit_right == False and hit_left == True:
                        hit_right = True
                        hit_left  = False
                        # Compute average classification result
                        print(Counter(chicken_dict[chicken_id]["classifications"]).most_common(1))
                        avg_result = Counter(chicken_dict[chicken_id]["classifications"]).most_common(1)[0][0]

                        # Calculate average feather count
                        feather_sum = sum(chicken_dict[chicken_id]["feather_count"])
                        feather_total = len(chicken_dict[chicken_id]["feather_count"])
                        featheravg_count = feather_sum / feather_total if feather_total != 0 else 0

                        # Update the counters and remove the chicken from the dictionary
                        if avg_result == 'defect':
                            defect_count += 1
                        else:
                            normal_count += 1

                        del chicken_dict[chicken_id]


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


# model_path = '/home/tqsang/JSON2YOLO/feather_YOLO_det/runs/detect/yolov8x_brio_feather_det_512_NoRotate/weights/best.pt'
model_path = 'runs/detect/yolov8x_brio_feather_det_512_NoRotate_dot2/weights/best.pt'
# model_path = '/home/tqsang/JSON2YOLO/feather_YOLO_seg/runs/segment/yolov8x_brio_feather_seg/weights/best.pt'

model_feather = YOLO(model_path)
# Define the areas for each group
areas = {
    '1-13': [1319, 598, 1319+1000, 598+1000],
}

# Get all video files
video_files = glob.glob('/mnt/tqsang/dot2_vid/dot2_*.mp4')

# Group video files
video_groups = {
    '1-13': [],
}



for video_file in video_files:
    video_number = int(video_file.split('_')[2].split('.')[0])
    if 1 <= video_number <= 10:
        video_groups['1-13'].append(video_file)
    # elif 2 <= video_number <= 12:
    #     video_groups['2-12'].append(video_file)
    # elif 13 <= video_number <= 29:
    #     video_groups['13-29'].append(video_file)
    # elif video_number == 30:
    #     video_groups['30'].append(video_file)
    # elif 31 <= video_number <= 40:
    #     video_groups['31-40'].append(video_file)




# Load the trained ResNet model
classification_model = models.resnet50(pretrained=False)
num_ftrs = classification_model.fc.in_features
classification_model.fc = nn.Linear(num_ftrs, 2)
classification_model = classification_model.to(device)
classification_model.load_state_dict(torch.load("resnet50_cls_pad.pt"))

# Create output directories if they don't exist
output_dirs = ['/mnt/tqsang/test_vid_noCls_wCrop_512']
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
# Process each video
for group, videos in video_groups.items():
    for video in videos:
        # if int(video.split('_')[2].split('.')[0]) in [3,4,7,8,9,40]: #test videos
            print(video)
            process_video(model, video, f'/mnt/tqsang/test_vid_noCls_wCrop_512/{video.split("/")[-1]}', areas[group])
