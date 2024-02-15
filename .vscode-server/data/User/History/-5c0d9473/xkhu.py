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
        img_rotated, new_bbox = rotate_image_and_bbox(img, [xmin, ymin, xmax, ymax], rotation_angle)
        cropped = img_rotated.crop(crop_area)
        img_name = f"part{part}_{frame_number:08d}_{class_id}_rot{rotation_angle}.png"
        cropped.save(os.path.join(output_folder, img_name))

        xmin, ymin, xmax, ymax = new_bbox
        xmin -= crop_area[0]
        xmax -= crop_area[0]
        ymin -= crop_area[1]
        ymax -= crop_area[1]

        json_data["images"].append({
            "id": img_id,
            "license": 1,
            "file_name": img_name,
            "height": crop_area[3] - crop_area[1],
            "width": crop_area[2] - crop_area[0]
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

def create_json_file(output_path, json_data):
    with open(output_path, "w") as f:
        json.dump(json_data, f)

root_folder = "/mnt/tqsang"
part1_folder = os.path.join(root_folder, "chicken_part1")
part2_folder = os.path.join(root_folder, "chicken_part2")

max_frame1 = 44606
max_frame2 = 4893

part1_xml_folder = os.path.join(part1_folder, "xml_44606")
part2_xml_folder = os.path.join(part2_folder, "xml1_4893")

part1_img_folder = os.path.join(part1_folder, "frames")
part2_img_folder = os.path.join(part2_folder, "frames")

areas_part1 = [(240, 100, 1008, 868), (878, 139, 1646, 907)]
areas_part2 = [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]

rotated_areas_part1 = [(x1, y1, x2, y2, random.randint(0, 360)) for x1, y1, x2, y2 in areas_part1]
rotated_areas_part2 = [(x1, y1, x2, y2, random.randint(0, 360)) for x1, y1, x2, y2 in areas_part2]

part1_xml_files = [f for f in sorted(os.listdir(part1_xml_folder)) if int(f[:-4]) <= max_frame1]
part2_xml_files = [f for f in sorted(os.listdir(part2_xml_folder)) if int(f[:-4]) <= max_frame2]

num_xml_files1 = len(part1_xml_files)
num_xml_files2 = len(part2_xml_files)
train_split1 = math.ceil(num_xml_files1 * 0.8)
train_split2 = math.ceil(num_xml_files2 * 0.8)

train_folder = "train"
val_folder = "val"

Path(train_folder).mkdir(exist_ok=True)
Path(val_folder).mkdir(exist_ok=True)

train_json = {
    "categories": [{"id": i, "name": str(i)} for i in range(9)],
    "images": [],
    "annotations": []
}

val_json = {
    "categories": [{"id": i, "name": str(i)} for i in range(9)],
    "images": [],
    "annotations": []
}

img_id = 1
ann_id = 1

for part, (xml_folder, img_folder, xml_files, max_frame, train_split, areas) in enumerate([
    (part1_xml_folder, part1_img_folder, part1_xml_files, max_frame1, train_split1, rotated_areas_part1),
    (part2_xml_folder, part2_img_folder, part2_xml_files, max_frame2, train_split2, rotated_areas_part2),
], 1):
    for i, xml_file in enumerate(xml_files):
        
        xml_path = os.path.join(xml_folder, xml_file)
        frame_number = int(xml_file[:-4])

        print(f"Processing part {part}, frame {frame_number} ({i+1}/{len(xml_files)})")
        
        if i < train_split:
            output_folder = train_folder
            json_data = train_json
        else:
            output_folder = val_folder
            json_data = val_json

        ann_id = process_xml_file(xml_path, frame_number, areas, img_folder, json_data, img_id, ann_id)
        img_id += 1

create_json_file("train.json", train_json)
create_json_file("val.json", val_json)


