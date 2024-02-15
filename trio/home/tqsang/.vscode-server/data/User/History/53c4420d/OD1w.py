import os 
import os.path as osp 
import cv2 
from tqdm import tqdm
import numpy as np 
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import json

IMG_DIR = '/home/tqsang/V100/tqsang/crop_obj/front_2_class/val2017'
ANNOT_FILE = '/home/tqsang/V100/tqsang/crop_obj/front_2_class/instances_val2017.json'
SAVE_DIR = '/home/tqsang/V100/tqsang/crop_obj/front_2_class/visval'

# IMG_DIR = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/val2017'
# ANNOT_FILE = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/annotations/instances_val2017.json'
# SAVE_DIR = '/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/visval'
COLOR_MAP = {}
BLEND_RATIO = 0.4
cats_to_vis = [3] 

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
def from_cv_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img) 

def from_pil_to_cv(im):
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

def get_binary_mask(polygons, height, width):
    # print(f'n polygons: {len(polygons)}')
    formatted_polygons = []
    for p in polygons:
        formatted_polygons.append((p[0], p[1]))

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(formatted_polygons, outline=1, fill=1)
    mask = np.array(img)
    return mask

def get_color_given_id(obj_id: int):
    # if COLOR_MAP.get(obj_id) is not None:
    #     return COLOR_MAP[obj_id]

    #TODO: find more simple random
    step = 10
    r = int((0 + obj_id*step + obj_id*step*step/256)%256) #int(obj_id/(256**3))
    g = int((255*(obj_id%2) - obj_id*step + obj_id*step*step/256)%256)
    b = int((255*(obj_id+1)%2 + obj_id*step + obj_id*step*step/256)%256)
    COLOR_MAP[obj_id] = np.array([b, g, r])

    return np.array([b, g, r])


def main():
    coco = COCO(ANNOT_FILE)
    cat_ids = coco.getCatIds()
    # if cats_to_vis is None:
    #     cats_to_vis = cat_ids

    N_PRINT = 1000
    count = 0
    for imgId in coco.imgs:
        if count > N_PRINT:
            break
        img_info = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)

        cv_img = cv2.imread('{}/{}'.format(IMG_DIR,img_info['file_name']))
        H, W, C = cv_img.shape


        draw_img = cv_img.copy() 
        is_draw = True
        for i, ann in enumerate(anns):
            ann_id = ann['id']
            # if ann['category_id'] != 1:
            #     continue
            bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
            centroid={
                'x': bbox_x + bbox_w / 2,
                'y': bbox_y + bbox_h / 2
            }
            box_area = bbox_w * bbox_h
            seg = ann['segmentation'][0]
            polygon = np.array(seg).reshape((int(len(seg)/2), 2))
            bin_mask = get_binary_mask(polygon, H, W)
            polygon_color = get_color_given_id(ann_id)

            draw_img[np.where(bin_mask > 0)] = cv_img[np.where(bin_mask > 0)]*BLEND_RATIO + (1-BLEND_RATIO)*polygon_color  
            is_draw = True
            start_point = (int(bbox_x), int(bbox_y))
            end_point = (int(bbox_x + bbox_w), int(bbox_y + bbox_h))

            cv2.rectangle(draw_img, start_point, end_point, [0, 0, 255], 1)

        if is_draw:
            cv2.imwrite(osp.join(SAVE_DIR, img_info['file_name']), draw_img)
            count += 1


if __name__ == '__main__':
    main()
