import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from PIL import ImageOps
import numpy as np
import cv2


# Load the YOLOv8 model
model = YOLO('/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det/runs/detect/yolov8x_brio_det_v3/weights/best.pt')

model.predict(source = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det/datasets/test/images', save=True, imgsz=256, conf=0.5)

