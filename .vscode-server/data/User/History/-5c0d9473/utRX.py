import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
import math
import random


def is_center_in_area(bbox, area):
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return area[0] <= center_x <= area[2] and area[1] <= center_y <= area[3]


def find_best_area(areas, bbox):
    for area in areas:
        if is_center_in_area(bbox, area):
            return area
    return None


def rotate_bbox(bbox, angle, img_width, img_height):
    img_center = (img_width / 2, img_height / 2)
    box_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    angle_rad = math.radians(angle)
    new_box_center = (
        math.cos(angle_rad) * (box_center[0] - img_center[0]) - math.sin(angle_rad) * (box_center[1] - img_center[1]) + img_center[0],
        math.sin(angle_rad) * (box_center[0] - img_center[0]) + math.cos(angle_rad) * (box_center[1] - img_center[1]) + img_center[1]
    )

    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]

    new_bbox = (
        new_box_center[0] - box_width / 2,
        new_box_center[1] - box_height / 2,
        new_box_center[0] + box_width / 2,
        new_box_center[1] + box_height / 2,
    )

    return tuple(map(int, new_bbox))


def rotate_area(area, angle, img_width, img_height):
    img_center = (img_width / 2, img_height / 2)
    area_center = ((area[0] + area[2]) / 2, (area[1] + area[3]) / 2)

    angle_rad = math.radians(angle)
    new_area_center = (
        math.cos(angle_rad) * (area_center[0] - img_center[0]) - math.sin(angle_rad) * (area_center[1] - img_center[1]) + img_center[0],
        math.sin(angle_rad) * (area_center[0] - img_center[0]) + math.cos(angle_rad) * (area_center[1] - img_center[1]) + img_center[1]
    )

    area_width = area[2] - area[0]
    area_height = area[3] - area[1]

    new_area = (
        new_area_center[0] - area_width / 2,
        new_area_center[1] - area_height / 2,
        new_area_center[0] + area_width / 2,
        new_area_center[1] + area_height / 2,
    )

    return tuple(map(int, new_area))


def process_xml_file(xml_path, frame_number, areas, img_folder, json_data, img_id, ann_id, angle):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img = Image.open(os.path.join(img_folder, f"{frame_number:08d}.png"))

    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    img_width, img_height = img.size

    for obj in root.findall("object"):
        class_id = int(obj.find("name").text)
        bbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = [int(float(bbox.find(coord).text)) for coord in ["xmin", "ymin", "xmax", "ymax"]]

        rotated_bbox = rotate_bbox([xmin, ymin, xmax, ymax], angle, img_width, img_height)

        crop_area = find_best_area(areas, rotated_bbox)
        if crop_area is None:
            continue

        rotated_area = rotate_area(crop_area, angle, img_width, img_height)
        cropped = img.crop(rotated_area)
        img_name = f"part{part}_{frame_number:08d}_{class_id}.png"
        cropped.save(os.path.join(output_folder, img_name))

        xmin, ymin, xmax, ymax = rotated_bbox
        xmin -= rotated_area[0]
        xmax -= rotated_area[0]
        ymin -= rotated_area[1]
        ymax -= rotated_area[1]

        json_data["images"].append({
            "id": img_id,
            "license": 1,
            "file_name": img_name,
            "height": rotated_area[3] - rotated_area[1],
            "width": rotated_area[2] - rotated_area[0]
        })

        json_data["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": class_id,
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "area": (xmax - xmin) * (ymax - ymin),
            "iscrowd": 0
        })

        ann_id += 1
    return ann_id
