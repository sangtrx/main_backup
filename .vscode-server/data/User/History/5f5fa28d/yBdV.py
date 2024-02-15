import os
import shutil

# Set the paths
source_dir = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_1chick1img/img'
destination_dir = '/home/tqsang/JSON2YOLO/feather_YOLO_det/datasets/val/images'
labels_dir = '/home/tqsang/JSON2YOLO/feather_YOLO_det/datasets/val/labels'

# Get the list of .txt files in the labels directory
txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

# Copy .jpg files that have the same name as .txt files
for txt_file in txt_files:
    base_name = os.path.splitext(txt_file)[0]
    jpg_file = f"{base_name}.png"

    source_file_path = os.path.join(source_dir, jpg_file)
    destination_file_path = os.path.join(destination_dir, jpg_file)

    if os.path.exists(source_file_path):
        shutil.copy(source_file_path, destination_file_path)
    else:
        print(f"WARNING: Corresponding .png file not found for {txt_file}")

print("Copying completed.")
