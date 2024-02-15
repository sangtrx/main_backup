#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = '/home/tqsang/V100/tqsang/crop_obj/chick_fewshot' ## change this 
IMAGE_DIR = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_new/val/img' #### change for train val test change line 110 too
ANNOTATION_DIR = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_new/val/mask'

INFO = {
    "description": "COCO_front_2_class_fewshot",
    "url": "https://github.com/sangtrx",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "sangtrx",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'normal',
        'supercategory': 'chicken',
    },
    {
        'id': 2,
        'name': 'defect',
        'supercategory': 'chicken',
    },
    {
        'id': 3,
        'name': 'feather',
        'supercategory': 'feather',
    }
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def main():
    defect_4_Trim_1 = [1,2,3,5,6,8,11]
    normal_4_Trim_1 = [4,7,9,10]
    defect_4_Trim_2 = [2,3,5]
    normal_4_Trim_2 = [1,4,6,7]
    defect_5_Trim = [1,2,4,5,15,7,16,19,20,22]
    normal_5_Trim = [3,6,8,9,10,11,12,13,14,17,18,21]
    defect_10_Trim = [2,3,4,5,6,8,9,11,15,16,17,19,21]
    normal_10_Trim = [1,7,10,12,13,14,18,20,22]
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    count_1 = 0
    count_2 = 0
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    print(annotation_filename)
                    # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                    string = os.path.basename(os.path.splitext(annotation_filename)[0])
                    stt = int(string.split("_")[-1])
                    if "4_Trim_1" in string:
                        if stt in normal_4_Trim_1:
                            class_id = 1
                        elif stt in defect_4_Trim_1:
                            class_id = 2
                    if "4_Trim_2" in string:
                        if stt in normal_4_Trim_2:
                            class_id = 1
                        elif stt in defect_4_Trim_2:
                            class_id = 2
                    if "5_Trim" in string:
                        if stt in normal_5_Trim:
                            class_id = 1
                        elif stt in defect_5_Trim:
                            class_id = 2
                    if "10_Trim" in string:
                        if stt in normal_10_Trim:
                            class_id = 1
                        elif stt in defect_10_Trim:
                            class_id = 2

                    if class_id ==1:
                        count_1 = count_1 + 1
                    else:
                        count_2 = count_2 + 1

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/instances_val2023.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('count_1: ', count_1)
    print('count_2: ', count_2)


if __name__ == "__main__":
    main()