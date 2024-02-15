import os
import json
from pathlib import Path
from PIL import Image

def coco_to_yolo_bbox(bbox, img_width, img_height):
    x_center = (bbox[0] + bbox[2] / 2) / img_width
    y_center = (bbox[1] + bbox[3] / 2) / img_height
    width = bbox[2] / img_width
    height = bbox[3] / img_height
    return [x_center, y_center, width, height]

def convert_coco_json_to_yolo_txt(coco_json_path, img_dir, output_dir):
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}
    annotations = coco_data["annotations"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        img_data = images[ann["image_id"]]
        img_path = os.path.join(img_dir, img_data["file_name"])
        img_width, img_height = img_data["width"], img_data["height"]

        yolo_bbox = coco_to_yolo_bbox(ann["bbox"], img_width, img_height)
        category_id = ann["category_id"]

        txt_path = os.path.join(output_dir, os.path.splitext(img_data["file_name"])[0] + ".txt")

        with open(txt_path, "a") as txt_file:
            txt_file.write(f"{category_id} {' '.join(map(str, yolo_bbox))}\n")

coco_json_train_path = "train.json"
coco_json_val_path = "val.json"
train_img_dir = "train"
val_img_dir = "val"

output_train_txt_dir = "train_txt"
output_val_txt_dir = "val_txt"

convert_coco_json_to_yolo_txt(coco_json_train_path, train_img_dir, output_train_txt_dir)
convert_coco_json_to_yolo_txt(coco_json_val_path, val_img_dir, output_val_txt_dir)
