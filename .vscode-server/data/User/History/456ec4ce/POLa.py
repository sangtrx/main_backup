import os
import shutil
import random

from PIL import Image

def copy_files(src_img_folder, src_txt_folder, dst_folder, filenames, class_label):
    for filename in filenames:
        # Strip the extension from filename
        filename_without_ext = os.path.splitext(filename)[0]
        
        img_src = os.path.join(src_img_folder, f"{filename}")
        txt_src = os.path.join(src_txt_folder, f"{filename_without_ext}.txt")

        # Open the image to get its size
        with Image.open(img_src) as img:
            img_width, img_height = img.size

        # Create destination directories if they don't exist
        img_dst_dir = os.path.join(dst_folder, 'images')
        os.makedirs(img_dst_dir, exist_ok=True)
        txt_dst_dir = os.path.join(dst_folder, 'labels')
        os.makedirs(txt_dst_dir, exist_ok=True)

        img_dst = os.path.join(img_dst_dir, f"{filename}")
        txt_dst = os.path.join(txt_dst_dir, f"{filename_without_ext}.txt")

        # Copy the image file
        shutil.copy(img_src, img_dst)

        # Copy the .txt file, change the first int to class_label, and normalize the bounding box coordinates
        with open(txt_src, 'r') as f_src, open(txt_dst, 'w') as f_dst:
            for line in f_src:
                parts = line.split(' ')
                parts[0] = str(class_label)
                
                # Normalize the bounding box coordinates
                parts[1] = str(float(parts[1]) / img_width)
                parts[2] = str(float(parts[2]) / img_height)
                parts[3] = str(float(parts[3]) / img_width)
                parts[4] = str(float(parts[4]) / img_height)

                f_dst.write(' '.join(parts))





def split_data(filenames, train=0.6, val=0.2, test=0.2):
    # random.shuffle(filenames)
    total = len(filenames)
    train_count = int(total * train)
    val_count = int(total * val)

    train_files = filenames[:train_count]
    val_files = filenames[train_count:train_count + val_count]
    test_files = filenames[train_count + val_count:]

    return train_files, val_files, test_files

def process_dataset(src_img_folder, src_txt_folder, dst_base, list_file, class_label):
    with open(list_file, 'r', encoding='ISO-8859-1') as f:
        filenames = [line.strip() for line in f]

    train_files, val_files, test_files = split_data(filenames)

    copy_files(src_img_folder, src_txt_folder, os.path.join(dst_base, 'train'), train_files, class_label)
    copy_files(src_img_folder, src_txt_folder, os.path.join(dst_base, 'val'), val_files, class_label)
    copy_files(src_img_folder, src_txt_folder, os.path.join(dst_base, 'test'), test_files, class_label)

# Define paths and class labels
src_img_folder = '/mnt/tqsang/dot1_data/original'
src_txt_folder = '/mnt/tqsang/dot1_data/label'
dst_base = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det/datasets'
defect_list_file = '/mnt/tqsang/dot1_data/defect.txt'
normal_list_file = '/mnt/tqsang/dot1_data/normal.txt'

# Process datasets
process_dataset(src_img_folder, src_txt_folder, dst_base, defect_list_file, 1)
process_dataset(src_img_folder, src_txt_folder, dst_base, normal_list_file, 0)
