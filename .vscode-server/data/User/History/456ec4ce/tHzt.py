import os
import shutil
import random

from PIL import Image

# Copy the .txt file, change the first int to class_label, convert to xywh, and normalize the bounding box coordinates
with open(txt_src, 'r') as f_src, open(txt_dst, 'w') as f_dst:
    for line in f_src:
        parts = line.split(' ')
        parts[0] = str(class_label)
        parts[1] = ''

        # Convert the bounding box coordinates from xyxy to xywh
        x_min = float(parts[2])
        y_min = float(parts[3])
        x_max = float(parts[4])
        y_max = float(parts[5])

        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        # Normalize the bounding box coordinates
        parts[2] = str(x_center / img_width)
        parts[3] = str(y_center / img_height)
        parts[4] = str(width / img_width)
        parts[5] = str(height / img_height)

        del parts[1]
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
