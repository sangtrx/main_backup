import os
import xml.etree.ElementTree as ET
import random
import shutil

# Paths to the folders
images_folder = '/mnt/tqsang/scale_sample'
labels_folder = '/mnt/tqsang/label_scale_sample'
output_folder = '/mnt/tqsang/yolo_scale'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Create train and validation subdirectories
train_images_dir = os.path.join(output_folder, 'datasets', 'train', 'images')
train_labels_dir = os.path.join(output_folder, 'datasets', 'train', 'labels')
val_images_dir = os.path.join(output_folder, 'datasets', 'val', 'images')
val_labels_dir = os.path.join(output_folder, 'datasets', 'val', 'labels')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get list of XML files
xml_files = [xml_file for xml_file in os.listdir(labels_folder) if xml_file.endswith('.xml')]
random.shuffle(xml_files)  # Shuffle the list

# Calculate train-validation split
num_train = int(0.8 * len(xml_files))
train_xml_files = xml_files[:num_train]
val_xml_files = xml_files[num_train:]

# Function to copy files
def copy_files(xml_files, src_folder, dest_images_folder, dest_labels_folder):
    for xml_file in xml_files:
        xml_path = os.path.join(labels_folder, xml_file)
        image_filename = xml_file.replace('.xml', '.png')
        image_path = os.path.join(images_folder, image_filename)
        
        shutil.copy(image_path, os.path.join(dest_images_folder, image_filename))
        shutil.copy(xml_path, os.path.join(dest_labels_folder, xml_file))

# Copy files for train and validation sets
copy_files(train_xml_files, images_folder, train_images_dir, train_labels_dir)
copy_files(val_xml_files, images_folder, val_images_dir, val_labels_dir)

print("Files copied to train and validation directories.")

# Conversion to YOLOv5 format
for xml_file in train_xml_files + val_xml_files:
    xml_path = os.path.join(labels_folder, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image information
    image_filename = root.find('filename').text
    image_path = os.path.join(images_folder, image_filename)
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    # Create or open the corresponding YOLO label file
    yolo_label_path = os.path.join(output_folder, image_filename.replace('.png', '.txt'))
    
    # Convert and write bounding box coordinates
    with open(yolo_label_path, 'w') as yolo_label_file:
        for obj in root.findall('object'):
            class_id = int(obj.find('name').text)
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)

            # Calculate YOLO coordinates
            x_center = (xmin + xmax) / (2 * image_width)
            y_center = (ymin + ymax) / (2 * image_height)
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Write YOLO label to the file
            yolo_label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Processed: {xml_file}")

print("Conversion to YOLOv5 format completed.")
