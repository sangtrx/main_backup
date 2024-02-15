import cv2
import numpy as np
import glob
from ultralytics import YOLO
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time
import copy
from torchvision.transforms.functional import to_tensor, normalize

# Function to perform inference and draw bounding boxes
def perform_inference_and_draw_boxes(model, img, output_frame, color, part_name):
    results = model.predict(img, conf=0.8)
    part_count = 0
    
    # Draw bounding boxes for the detected objects
    for i, (box, score, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        cv2.rectangle(output_frame, 
                      (int(box_x1), int(box_y1)), 
                      (int(box_x2), int(box_y2)), 
                      color, 2)
        part_count = part_count+1

    return part_count

# Define the paths to the YOLO models
model_paths = {
    'feather': '/home/tqsang/JSON2YOLO/feather_YOLO_det/runs/detect/yolov8x_brio_feather_det_512_NoRotate_dot2_2/weights/best.pt',
    'wing': '/home/tqsang/JSON2YOLO/wing_YOLO_det/runs/detect/yolov8x_wing2/weights/best.pt',
    'skin': '/home/tqsang/JSON2YOLO/skin_YOLO_det/runs/detect/yolov8x_skin/weights/best.pt',
    'flesh': '/home/tqsang/JSON2YOLO/flesh_YOLO_det/runs/detect/yolov8x_flesh/weights/best.pt'
}

# Load the YOLO models
models = {part: YOLO(model_path) for part, model_path in model_paths.items()}

# Define a fixed color list and text color list
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
text_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

# Create the output folder if it doesn't exist
output_directory = '/mnt/tqsang/dot2_test/test_output'
os.makedirs(output_directory, exist_ok=True)

# Loop through images in the specified directory
image_directory = '/mnt/tqsang/dot2_test/test_imgs'

for image_path in glob.glob(os.path.join(image_directory, '*.png')):
    img = cv2.imread(image_path)
    output_frame = img.copy()

    part_counts = {}

    # Perform inference and draw boxes for each part using the fixed colors
    for i, (part, model) in enumerate(models.items()):
        color = colors[i % len(colors)]  # Cycle through the fixed color list
        text_color = text_colors[i % len(text_colors)]  # Get the corresponding text color
        part_count = perform_inference_and_draw_boxes(model, img, output_frame, color, part)
        part_counts[part] = part_count

        # Put text next to the image with the part counts using the text color
        text_position = (10, 30)
        for part, count in part_counts.items():
            text = f'{part}: {count}'
            cv2.putText(output_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            text_position = (text_position[0], text_position[1] + 30)

    # Save the output image to the output directory
    output_image_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(output_image_path, output_frame)

print("Processing complete.")
