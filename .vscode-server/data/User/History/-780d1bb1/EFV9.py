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



        
                # Perform inference on the smallest square area for feather detection
                results_feather = model_feather.predict(cropped_square, conf=0.8)


                
                # Draw bounding box for each feather in results_feather 
                for i, (feather_box, feather_score, feather_class) in enumerate(zip(results_feather[0].boxes.xyxy, 
                                                                                    results_feather[0].boxes.conf, 
                                                                                    results_feather[0].boxes.cls)):
                        # Draw the bounding box for each feather
                        feather_box_x1, feather_box_y1, feather_box_x2, feather_box_y2 = feather_box.tolist()
                        cv2.rectangle(output_frame, 
                                    (int(feather_box_x1+x + square_x1), int(feather_box_y1+y + square_y1 )), 
                                    (int(feather_box_x2+x + square_x1), int(feather_box_y2+y + square_y1)), 
                                    (0, 255, 100), 2)  



                
 



model_path = '/home/tqsang/JSON2YOLO/feather_YOLO_det/runs/detect/yolov8x_brio_feather_det_512_NoRotate_dot2_2/weights/best.pt'

model_feather = YOLO(model_path)

model_path = '/home/tqsang/JSON2YOLO/wing_YOLO_det/runs/detect/yolov8x_wing2/weights/best.pt'
model_wing = YOLO(model_path)
model_path = '/home/tqsang/JSON2YOLO/skin_YOLO_det/runs/detect/yolov8x_skin/weights/best.pt'
model_skin = YOLO(model_path)

model_path = '/home/tqsang/JSON2YOLO/flesh_YOLO_det/runs/detect/yolov8x_flesh/weights/best.pt'
model_flesh = YOLO(model_path)


