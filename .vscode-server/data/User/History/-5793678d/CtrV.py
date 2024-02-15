import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from PIL import ImageOps
import numpy as np
import cv2


def print_hsv_values(image_path, x, y):
    # Load the image
    img = Image.open(image_path)
    img_np = np.array(img)

    # Convert the image to HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Print the HSV values at the specified location
    print("HSV values at", x, y, ":", hsv[int(y), int(x)])

# Call the function for the specified image
image_path = "/mnt/tqsang/chicken_part1/part1_cropped_frames/00044998_R.png"
x, y = 300.8313293457031, 333.047119140625
print_hsv_values(image_path, x, y)
