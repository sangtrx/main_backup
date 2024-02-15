import os
import json
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps
from pathlib import Path
import math
import random

def rotate_image_and_bbox(img, bbox, angle):
    img_center = (img.width // 2, img.height // 2)
    img_rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True, center=img_center)
    
    xmin, ymin, xmax, ymax = bbox
    box_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
    box_center_rotated = rotate_point(img_center, box_center, math.radians(angle))
    
    width = xmax - xmin
    height = ymax - ymin
    new_bbox = (box_center_rotated[0] - width // 2, box_center_rotated[1] - height // 2,
                box_center_rotated[0] + width // 2, box_center_rotated[1] + height // 2)
    
    return img_rotated, new_bbox

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

def is_center_in_area(bbox, area):
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    area_center = ((area[0] + area[2]) / 2, (area[1] + area[3]) / 2)
    area_half_size = (area[2] - area[0]) / 2
    
    rotated_center = rotate_point(area_center, (center_x, center_y), math.radians(-area[4]))
    dx = abs(rotated_center[0] - area_center[0])
    dy = abs(rotated_center[1] - area_center[1])

    return dx <= area_half_size and dy <= area_half_size

def find_best_area(areas, bbox):
    for area in areas:
        if is_center_in_area(bbox, area):
            return area
    return None

def process_xml_file(xml_path, frame_number, areas, img_folder, json_data, img_id, ann_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img = Image.open(os.path.join(img_folder, f"{frame_number:08d}.png"))

    for obj in root.findall("object"):
        class_id = int(obj.find("name").text)
        bbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = [int(float(bbox.find(coord).text)) for coord in ["xmin", "ymin", "xmax", "ymax"]]

        crop_area = find_best_area(areas, [xmin, ymin, xmax, ymax])
        if crop_area is None:
            continue

        rotation_angle = random.randint(0, 360)
        img_rot
