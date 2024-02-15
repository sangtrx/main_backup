from os import listdir, makedirs
from os.path import join
from bs4 import BeautifulSoup

import cv2
import os
import glob
import numpy as np


def get_box_from_ann_img(img, ann_img):
    '''
    get box xyxy by ann_img
    '''
    diff_img = np.array(img != ann_img)
    diff_img[:, :100, :] = False
    box_coord = np.where(diff_img==True)
    x1, y1 = box_coord[1][0], box_coord[0][0]
    x2, y2 = box_coord[1][-1], box_coord[0][-1]

    return [x1, y1, x2, y2]


def get_box_from_xml(xml_fpath):
    with open(xml_fpath, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml") 
    b_unique = Bs_data.find_all('unique')
    x1 = int(Bs_data.find('bndbox').find('xmin').string)
    y1 = int(Bs_data.find('bndbox').find('ymin').string)
    x2 = int(Bs_data.find('bndbox').find('xmax').string)
    y2 = int(Bs_data.find('bndbox').find('ymax').string)

    return [x1, y1, x2, y2]


def box_cxcywh_to_xyxy(box):
    x_c, y_c, w, h = box 
    return [int(x_c - 0.5 * w), int(y_c - 0.5 * h),
         int(x_c + 0.5 * w), int(y_c + 0.5 * h)]


def box_xyxy_to_cxcywh(box):
    x0, y0, x1, y1 = box
    return [int((x0 + x1) / 2), int((y0 + y1) / 2),
         int(x1 - x0), int(y1 - y0)]

def box_xyxy_to_xywh(box):
    x0, y0, x1, y1 = box
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]

def frames2vid(frames_path, output_path, ext='png'):
    ''' 
    '''
    img_array = []
    size = None
    for filename in sorted(glob.glob(frames_path + '/*.{}'.format(ext)), 
                                key=os.path.getmtime, reverse=False):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
 
def listdir_fullpath(path):
    return [join(path, name) for name in os.listdir(path)]


def init_ann_coco_json_dict(cats):
    return {
        "images": [],
        "annotations": [],
        "categories": cats # categories
    }


def create_json_img_format(im_h, im_w, iname, img_id):
    return {
      "license": None,
      "file_name": iname,
      "coco_url": None,
      "height": im_h,
      "width": im_w,
      "id": img_id
    }


def create_json_ann_format(im_id, 
                            polygon,
                            cxcywh_box,
                            cat_id, 
                            ann_id, 
                            inmodal_bbox=None, 
                            inmodal_seg=None):
    
    _, _, w, h = cxcywh_box
    return {
      "segmentation": polygon,
      "area": w*h,
      "iscrowd": 0,
      "image_id": im_id,
      "bbox": cxcywh_box,
      "category_id": cat_id,
      "id": ann_id,
      "inmodal_bbox": inmodal_bbox,
      "inmodal_seg": inmodal_seg 
    }


def show_box_xyxy(box, img):
    color = (255,255,255)
    thickness = 1
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    cv2.imwrite('test_box.png', img)
