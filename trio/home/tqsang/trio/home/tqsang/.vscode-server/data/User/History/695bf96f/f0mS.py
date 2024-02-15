from mmdet.apis import init_detector, inference_detector
import mmcv

import cv2
from tqdm import tqdm
import os

# Specify the path to model config and checkpoint file
config_file = '/home/tqsang/RefineMask/work_dirs/front_2_class/front_2_class.py'
checkpoint_file = '/home/tqsang/RefineMask/work_dirs/front_2_class/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/10_Trim_1000_5.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='/home/tqsang/RefineMask/work_dirs/front_2_class/vis/10_Trim_1000_5.jpg')

