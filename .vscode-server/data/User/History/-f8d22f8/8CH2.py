import os
import xml.etree.ElementTree as ET
import random
import shutil

# Paths to the folders
images_folder = '/mnt/tqsang/scale_sample'
labels_folder = '/mnt/tqsang/label_scale_sample'
output_folder = '/mnt/tqsang/yolo_scale'
train_ratio = 0.8  # 80% for training, 20% for validation

# Create the output folders if they don't exist
train_images_folder = os.path.join(output_folder, 'datasets', 'train', 'images')
train_labels_folder = os.path.join(output_folder, 'datasets', 'train', 'labels')
val_images_folder = os.path.join(output_folder, 'datasets', 'val', 'images')
val_labels_folder = os.path.join(output_folder, 'datasets', 'val', 'labels')

for folder in [train_images_folder, train_labels_folder, val_images_folder, val_labels_folder]:
    os.makedirs(folder, exist_ok=True)

# Create a list of XML files
xml_files = [xml_file for xml_file in os.listdir(labels_folder) if xml_file.endswith('.xml')]
random.shuffle(xml_files)  # Shuffle the list for randomness

# Split the data based on the train-validation ratio
num_train = int(len(xml_files) * train_ratio)
train_xml_files = xml_files[:num_train]
val_xml_files = xml_files[num_train:]

# Move/Copy files to respective train and val folders
def process_files(xml_files, images_folder, labels_folder, images_output, labels_output):
    for xml_file in xml_files:
        xml_path = os.path.join(labels_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_filename = root.find('filename').text
        image_path = os.path.join(images_folder, image_filename)
        yolo_label_path = os.path.join(labels_folder, xml_file.replace('.xml', '.txt'))

        if os.path.exists(image_path) and os.path.exists(yolo_label_path):
            shutil.copy(image_path, images_output)
            shutil.copy(yolo_label_path, labels_output)

# Process train data
process_files(train_xml_files, images_folder, labels_folder, train_images_folder, train_labels_folder)

# Process validation data
process_files(val_xml_files, images_folder, labels_folder, val_images_folder, val_labels_folder)

print("Data distribution and conversion to YOLOv5 format completed.")
