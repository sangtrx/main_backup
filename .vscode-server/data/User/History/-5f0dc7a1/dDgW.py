import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from PIL import ImageOps
import numpy as np
import cv2


# Load the YOLOv8 model
model = YOLO('/mnt/tqsang/hand_dataset_YOLO/runs/detect/yolov8x_hand_full_aug/weights/best.pt')

model.predict(source = '/mnt/tqsang/hand_dataset_YOLO/datasets/val/images', save=True, imgsz=256, conf=0.5)

