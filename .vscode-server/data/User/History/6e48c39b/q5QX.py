import os
import shutil

from PIL import Image

def copy_and_crop_files(src_img_folder, src_txt_folder, dst_folder, filenames, class_label):
    class_label_dir = 'defect' if class_label == 1 else 'normal'
    for filename in filenames:
        # Strip the extension from filename
        filename_without_ext = os.path.splitext(filename)[0]
        
        img_src = os.path.join(src_img_folder, f"{filename}")
        txt_src = os.path.join(src_txt_folder, f"{filename_without_ext}.txt")

        # Open the image to get its size and to crop it later
        with Image.open(img_src) as img:
            img_width, img_height = img.size

            # Open the corresponding label file and crop the image based on the bounding box
            with open(txt_src, 'r') as f:
                for line in f:
                    parts = line.split(' ')
                    x_min = float(parts[1])
                    y_min = float(parts[2])
                    x_max = float(parts[3])
                    y_max = float(parts[4])

                    cropped_img = img.crop((x_min, y_min, x_max, y_max))

                    # Save the cropped image to the destination directory
                    dst_dir = os.path.join(dst_folder, class_label_dir)
                    os.makedirs(dst_dir, exist_ok=True)

                    cropped_img.save(os.path.join(dst_dir, f"{filename_without_ext}.jpg"))

def process_dataset(src_img_folder, src_txt_folder, dst_base, list_file, class_label):
    with open(list_file, 'r', encoding='ISO-8859-1') as f:
        filenames = [line.strip() for line in f]

    copy_and_crop_files(src_img_folder, src_txt_folder, os.path.join(dst_base, 'train'), filenames, class_label)

# Define paths and class labels
src_img_folder = '/mnt/tqsang/dot1_data/original'
src_txt_folder = '/mnt/tqsang/dot1_data/label'
dst_base = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det/datasets'
defect_list_file = '/mnt/tqsang/dot1_data/defect.txt'
normal_list_file = '/mnt/tqsang/dot1_data/normal.txt'

# Process datasets
process_dataset(src_img_folder, src_txt_folder, dst_base, defect_list_file, 1)
process_dataset(src_img_folder, src_txt_folder, dst_base, normal_list_file, 0)
