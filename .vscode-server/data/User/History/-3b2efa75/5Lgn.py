import os
from PIL import Image
import numpy as np

def process_images(dataset_path, new_dataset_path, sync_normal=False, shift_pixels=0):
    for filename in os.listdir(dataset_path):
        # Load the image
        image = Image.open(os.path.join(dataset_path, filename))

        # Resize image
        aspect_ratio = image.size[0] / image.size[1]
        if aspect_ratio > 1:
            # width is greater than height
            new_image = image.resize((256, int(256 / aspect_ratio)))
        else:
            # height is greater than width
            new_image = image.resize((int(256 * aspect_ratio), 256))

        # Create a black 256x256 image
        black_image = Image.new('RGB', (256, 256))

        # Compute the position where the image should be pasted
        paste_position = ((black_image.size[0] - new_image.size[0]) // 2 + shift_pixels,
                          (black_image.size[1] - new_image.size[1]) // 2)
        
        # Paste the image
        black_image.paste(new_image, paste_position)

        # Save the new image
        save_name = filename if not sync_normal else f"{filename.split('.')[0]}_{shift_pixels}.png"
        black_image.save(os.path.join(new_dataset_path, save_name))

base_path = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification'
new_base_path = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/datasets_classification_sync'
splits = ['test', 'train', 'val']
classes = ['normal', 'defect']

for split in splits:
    for cls in classes:
        new_path = os.path.join(new_base_path, split, cls)
        
        # Create directory if it does not exist
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # Compute sync_normal factor
        num_defects = len(os.listdir(os.path.join(base_path, split, 'defect')))
        num_normals = len(os.listdir(os.path.join(base_path, split, 'normal')))
        sync_normal = num_defects / num_normals

        if cls == 'normal':
            for i in range(int(sync_normal)):
                process_images(os.path.join(base_path, split, cls), 
                               new_path, 
                               sync_normal=True, 
                               shift_pixels=2*i)
        else: 
            process_images(os.path.join(base_path, split, cls), 
                           new_path)
