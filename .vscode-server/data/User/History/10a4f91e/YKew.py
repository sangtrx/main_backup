import os
import xml.etree.ElementTree as ET
from shutil import copyfile
from pathlib import Path

def convert_voc_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []

    for obj in root.findall("object"):
        cls = 0  # Set the class to 0 (hand)

        # Extract the bounding box
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / (2 * img_width)
        y_center = (ymin + ymax) / (2 * img_height)
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        boxes.append((cls, x_center, y_center, width, height))

    return boxes

def create_yolo_dataset(base_dir, output_dir, img_dir, xml_dir):
    xml_files = sorted(os.listdir(xml_dir))
    n_train = int(len(xml_files) * 0.8)

    for i, xml_file in enumerate(xml_files):
        img_name = xml_file[:-4] + ".png"  # Assuming images are in .png format
        img_path = os.path.join(img_dir, img_name)
        xml_path = os.path.join(xml_dir, xml_file)

        img = Image.open(img_path)
        img_width, img_height = img.size

        boxes = convert_voc_to_yolo(xml_path, img_width, img_height)

        if i < n_train:
            subset = "train"
        else:
            subset = "val"

        output_img_dir = os.path.join(output_dir, subset, "images")
        output_label_dir = os.path.join(output_dir, subset, "labels")

        Path(output_img_dir).mkdir(parents=True, exist_ok=True)
        Path(output_label_dir).mkdir(parents=True, exist_ok=True)

        # Copy the image
        copyfile(img_path, os.path.join(output_img_dir, img_name))

        # Save the labels
        with open(os.path.join(output_label_dir, img_name[:-4] + ".txt"), "w") as f:
            for box in boxes:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")

base_dir = "/mnt/tqsang/"
output_dir = os.path.join(base_dir, "hand_dataset_YOLO")
img_dir = os.path.join(base_dir, "chicken_part1/frames")
xml_dir = os.path.join(base_dir, "hand")

create_yolo_dataset(base_dir, output_dir, img_dir, xml_dir)
