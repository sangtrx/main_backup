import os
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

def create_yolo_dataset(src_images_folder, src_annotations_folder, dst_images_folder, dst_labels_folder):
    Path(dst_images_folder).mkdir(parents=True, exist_ok=True)
    Path(dst_labels_folder).mkdir(parents=True, exist_ok=True)

    for xml_file in os.listdir(src_annotations_folder):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(src_annotations_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get the size of the image
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # Create a .txt file with the same name as the image
            txt_filename = os.path.splitext(xml_file)[0] + '.txt'
            txt_path = os.path.join(dst_labels_folder, txt_filename)

            with open(txt_path, 'w') as f:
                for obj in root.findall('object'):
                    # Change the class of all xml to 0 in YOLO
                    cls = 0

                    # Get bounding box coordinates in Pascal VOC format
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    # Convert Pascal VOC format to YOLO format
                    x_center = ((xmin + xmax) / 2) / width
                    y_center = ((ymin + ymax) / 2) / height
                    box_width = (xmax - xmin) / width
                    box_height = (ymax - ymin) / height

                    # Write YOLO format annotation to the .txt file
                    f.write(f"{cls} {x_center} {y_center} {box_width} {box_height}\n")

            # Copy the corresponding image to the images folder
            image_filename = os.path.splitext(xml_file)[0] + '.png'
            src_image_path = os.path.join(src_images_folder, image_filename)
            dst_image_path = os.path.join(dst_images_folder, image_filename)
            shutil.copyfile(src_image_path, dst_image_path)

src_images_folder = "/mnt/tqsang/chicken_part1/frames"
src_annotations_folder = "/mnt/tqsang/hand"
dst_images_folder = "/mnt/tqsang/hand_dataset_YOLO/images"
dst_labels_folder = "/mnt/tqsang/hand_dataset_YOLO/labels"

create_yolo_dataset(src_images_folder, src_annotations_folder, dst_images_folder, dst_labels_folder)
