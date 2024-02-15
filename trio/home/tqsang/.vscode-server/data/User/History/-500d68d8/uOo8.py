import numpy as np
from copy import deepcopy
import cv2
import os
from os.path import splitext, join
from pathlib import Path
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import imantics
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

import datetime

N_RAND_IMAGES=1

def draw_multiple_boxes(img, bboxes, class_names, image_name):
    for i, bbox in enumerate(bboxes):
        x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
        cv2.putText(img, class_names[i], (int(x1), int(y1)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # cv2.imwrite("../data/outtest/" + image_name + ".png", img)
    cv2.imwrite("../data/outtest/" + image_name + ".jpg", img)


def draw_polygon_on_image(pil_im, polygons):
    img = pil_im

    img2 = img.copy()
    for polygon in polygons:
        formatted_polygon = []
        for p in polygon.tolist():
            formatted_polygon.append((p[0], p[1]))

        draw = ImageDraw.Draw(img2)
        draw.polygon(formatted_polygon, fill = "wheat")

    img3 = Image.blend(img, img2, 0.5)
    # img3.save('../data/outtest/plg_drawing.png')
    img3.save('../data/outtest/plg_drawing.jpg')


def get_bbox_from_binary_mask(binary_mask):
    '''
    binary mask: cv2 grayscale image
    '''
    _, thresholded = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_OTSU)
    bbox = cv2.boundingRect(thresholded)
    return list(map(int, bbox))

def mk_my_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass

def from_cv_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img) 

def from_pil_to_cv(im):
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

def get_edges_mask(polygon, height, width):
    img = Image.new('L', (width, height), 0)
    line_width=10
    for i, p in enumerate(polygon):
        if i > 0:
            line = [(polygon[i-1][0], polygon[i-1][1]), (polygon[i][0], polygon[i][1])]
            ImageDraw.Draw(img).line(line, fill=1, width=line_width)

    #draw the last line here
    n=len(polygon)
    line = [(polygon[n-1][0], polygon[n-1][1]), (polygon[0][0], polygon[0][1])]
    ImageDraw.Draw(img).line(line, fill=1, width=line_width)
    mask=np.array(img) 

    return mask

def get_binary_mask(polygon, height, width):
    formatted_polygon = []
    for p in polygon:
        formatted_polygon.append((p[0], p[1]))
    
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(formatted_polygon, outline=1, fill=1)
    mask = np.array(img)
    return mask

def get_img_anns(imgId, catId, data_dir, data_type, coco):
    img_info = coco.loadImgs(imgId)[0]
    img = cv2.imread('{}/{}2017/{}'.format(data_dir,data_type,img_info['file_name']))

    annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catId, iscrowd=None)
    anns = coco.loadAnns(annIds)

    # assert len(anns) == 1 # because it's single chicken per frame dataset
    # ann = anns[0]
    ret = []
    for ann in anns:
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']

        centroid={
            'x': bbox_x + bbox_w / 2,
            'y': bbox_y + bbox_h / 2
        }
        box_area = bbox_w * bbox_h

        assert len(ann['segmentation']) == 1

        seg = ann['segmentation'][0] 
        polygon = np.array(seg).reshape((int(len(seg)/2), 2))
        ret.append([img, polygon, ann['bbox'], centroid, box_area, img_info['file_name']])
    return ret
    
def get_cropped_img_and_mask(cv_img, polygon, bbox):
    x, y, w, h = bbox
    mask = get_binary_mask(polygon, cv_img.shape[0], cv_img.shape[1])
    polygon_cropped_img = cv2.bitwise_and(cv_img, cv_img, mask=mask)

    # polygon_cropped_img = polygon_cropped_img[int(y): int(y + h), int(x): int(x + w)]
    # polygon_cropped_mask = mask[int(y): int(y + h), int(x): int(x + w)]
    polygon_cropped_img = polygon_cropped_img
    polygon_cropped_mask = mask


    return from_cv_to_pil(cv_img), from_cv_to_pil(polygon_cropped_img), \
        from_cv_to_pil(np.uint8(255*polygon_cropped_mask))

def euler_distance_to_bottom(img, point=(0,0)):
    return np.linalg.norm(
        np.array([img.size[0], img.size[1]]) \
            - np.array([point[0], point[1]])
    )

def resize_pil_img(im, scale_w, scale_h):
    return im.resize((round(im.size[0]*scale_w), round(im.size[1]*scale_h)))

def blur_edge_points(im, edge_mask):
    #cv_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    cv_im = from_pil_to_cv(im)
    blurred_im = cv2.GaussianBlur(cv_im, (3, 3), 1, borderType=cv2.BORDER_ISOLATED)
    edge_mask = np.stack((edge_mask,)*3, axis=-1)

    cv_res_im = np.where(edge_mask==np.array([0, 0, 0]), cv_im, blurred_im)
    #res_im = Image.fromarray(cv2.cvtColor(cv_res_im, cv2.COLOR_BGR2RGB))
    res_im = from_cv_to_pil(cv_res_im)

    return res_im

def morphology_smooth(mask):
    cv_mask = np.array(mask) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    (thresh, binRed) = cv2.threshold(cv_mask, 128, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(cv_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    res_im = Image.fromarray(opening)
    return res_im

def get_fullsize_mask(cropped_mask, position, full_size):
    x, y = position[0], position[1]
    fmask = Image.new("L", full_size, 0)
    fmask.paste(cropped_mask, position)

    return fmask

def average_smooth(res_im, m_im, pasted_fsize_mask, a_im, ori_fsize_mask, a_box, paste_pos):
    # cv_m_im = cv2.cvtColor(np.asarray(m_im), cv2.COLOR_RGB2BGR)
    # cv_a_im = cv2.cvtColor(np.asarray(a_im), cv2.COLOR_RGB2BGR)
    cv_m_im = from_pil_to_cv(m_im)
    cv_a_im = from_pil_to_cv(a_im)
    cv_p_mask = np.array(pasted_fsize_mask)
    cv_o_mask = np.array(ori_fsize_mask)
    cv_p_mask = np.stack((cv_p_mask,)*3, axis=-1)
    cv_o_mask = np.stack((cv_o_mask,)*3, axis=-1)

    m_shrink = cv2.erode(cv_p_mask, np.ones((5,5), np.uint8), iterations=1)
    m_shrink_part = cv2.bitwise_and(cv_m_im, m_shrink)
    m_expand = cv2.dilate(cv_p_mask, np.ones((5,5),np.uint8), iterations=1)
    m_expand_part = cv2.bitwise_and(cv_m_im, m_expand)
    m_narrow_band = m_expand_part - m_shrink_part
    m_narrow_band_mask = m_expand - m_shrink

    m_expand_band = m_expand_part - cv2.bitwise_and(cv_m_im, cv_p_mask)
    m_expand_band_mask = m_expand - cv_p_mask
    m_expand_band = from_cv_to_pil(m_expand_band)
    m_expand_band_mask = Image.fromarray(m_expand_band_mask).convert("L")

    a_shrink = cv2.erode(cv_o_mask, np.ones((5,5), np.uint8), iterations=1)
    a_shrink_part = cv2.bitwise_and(cv_a_im, a_shrink)
    a_mask_part = cv2.bitwise_and(cv_a_im, cv_o_mask)
    a_narrow_band = a_mask_part - a_shrink_part
    a_narrow_band_mask = cv_o_mask - a_shrink
    
    # move added narrow band to the paste position
    x,y,w,h=a_box
    crop_a_narrow_band = a_narrow_band[int(y): int(y + h), int(x): int(x + w)]
    crop_a_narrow_band_mask = a_narrow_band_mask[int(y): int(y + h), int(x): int(x + w)]

    crop_a_narrow_band = from_cv_to_pil(crop_a_narrow_band)
    crop_a_narrow_band_mask = from_cv_to_pil(crop_a_narrow_band_mask)

    a_narrow_band = Image.new("L", (a_narrow_band.shape[1], a_narrow_band.shape[0]), 0)
    a_narrow_band_mask = deepcopy(a_narrow_band)
    a_narrow_band.paste(crop_a_narrow_band, paste_pos)
    a_narrow_band = from_pil_to_cv(a_narrow_band)
    a_narrow_band_mask.paste(crop_a_narrow_band_mask, paste_pos)
    a_narrow_band_mask = from_pil_to_cv(a_narrow_band_mask)

    avg_band = (0.9*m_narrow_band.astype('float') + 0.1*a_narrow_band.astype('float')) / 1
    avg_band = avg_band.astype(np.uint8)
    avg_band = from_cv_to_pil(avg_band)

    avg_band_mask = cv2.bitwise_or(m_narrow_band_mask, a_narrow_band_mask)
    avg_band_mask = from_cv_to_pil(avg_band_mask).convert("L")

    res_im.paste(avg_band, (0, 0), mask=avg_band_mask)
    res_im.paste(m_expand_band, (0, 0), mask=m_expand_band_mask)

    return res_im

def create_json_img_format(im, iname, id):
    return {
      "id": id,
      "file_name": iname,
      "width": im.size[0],
      "height": im.size[1],
      "date_captured": datetime.datetime.utcnow().isoformat(' '),
      "license": 1,
      "coco_url": "",
      "flickr_url": ""
    }

def create_json_ann_format(im_id, 
                            plg,
                            box,
                            cat_id, seg_ii, inmodal_bbox=None, inmodal_seg=None):
    
    x, y, w, h = box
    box = [int(x), int(y),int(w),int(h)]
    # import pdb; pdb.set_trace()
    return {
      "id": seg_ii,
      "image_id": im_id,
      "category_id": cat_id,
      "iscrowd": 0,
      "area": int(w*h),
      "bbox": box,
      "segmentation": [
          plg.reshape((plg.shape[0]*plg.shape[1])).tolist()
      ],
      "width": int(w), 
      "height": int(h)
    #   "inmodal_bbox": inmodal_bbox,  #### cmt 2 dong de tao mask cho refine
    #   "inmodal_seg": inmodal_seg 
    }


if __name__ == "__main__":
    # TODO declare the data dir - CHANGE 2 DIR HERE
    # dataDir = "/data/tqminh/AmodalSeg/chicken_data/single_small_chickens"
    # add_dataDir = "/data/tqminh/AmodalSeg/chicken_data/single_small_chickens"

    dataDir = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO"
    add_dataDir = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO"
    background =  "/home/tqsang/V100/tqsang/crop_obj/background/background_front_noise3_gaussian.jpg"

    # TODO declare dataType
    dataType='val'
    annFile='{}/annotations/instances_{}2017.json'.format(dataDir,dataType)
    a_annFile='{}/annotations/instances_{}2017.json'.format(add_dataDir,dataType)

    # TODO create syn data dir - CHANGE DIR HERE
    # syn_dataDir = "../data/chicken_data/syn_2_chickens"
    syn_dataDir = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg"

    syn_annFile = '{}/annotations/instances_{}2017.json'.format(syn_dataDir,dataType)
    mk_my_dir(syn_dataDir)
    needed_folders = ['train2017', 'val2017', 'annotations']
    for folder in needed_folders:
        mk_my_dir(os.path.join(syn_dataDir, folder))

    # Initialize the COCO api 
    coco=COCO(annFile)
    catIds = coco.getCatIds()
    a_coco = COCO(a_annFile)
    a_catIds = a_coco.getCatIds()

    # define the coco annotation for syn data
    # coco_cats = [
    #     {'supercategory': 'chicken', 'id': 1, 'name': 'chicken_frontfacing'}, 
    #     {'supercategory': 'chicken', 'id': 2, 'name': 'chicken_backfacing'}, 
    #     {'supercategory': 'chicken', 'id': 3, 'name': 'chicken_sideways_left'}, 
    #     {'supercategory': 'chicken', 'id': 4, 'name': 'chicken_sideways_Right'}]

    # TODO CHANGE HERE THE CATEGORIES OF METADATA
    # coco_cats = [
    #     {'supercategory': 'chicken', 'id': 1, 'name': 'chicken'}]
    coco_cats = [
        {'supercategory': 'chicken', 'id': 1, 'name': 'normal'},
        {'supercategory': 'chicken', 'id': 2, 'name': 'defect'}
    ]

    INFO = {
        "description": "front_2_class",
        "url": "https://github.com/sangtrx",
        "version": "0.1.0",
        "year": 2022,
        "contributor": "sangtrx",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
        # "date_created": datetime.datetime.utcnow().isoformat(' ')
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
    }

    ]

    ann_dict = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    # ann_dict = {
    #     "images": [],
    #     "annotations": [],
    #     "categories": coco_cats
    # }
    ii=0  ## img id 
    seg_ii=0 ## annotation ID
    bg_img = from_cv_to_pil(cv2.imread(background)) ## load background sẵn

    for catId in tqdm(catIds):  ## loop từng class
        imgIds = coco.getImgIds(catIds=catId)
        for imgId in tqdm(imgIds):
            try:
                a_cv_img, a_plg, a_box, a_ctr, a_Sbox, a_iname = get_img_anns(imgId, catId, add_dataDir, dataType, a_coco)[0]
                a_im, a_plg_img, a_plg_mask = get_cropped_img_and_mask(a_cv_img, a_plg, a_box)
                # a_plg_mask.save('/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/'+ a_iname +'.jpg')
                # continue
            except:
                continue

            res_im = bg_img.copy()

            a_plg_mask = a_plg_mask.convert('L')

            
            #### smooth hình cho đẹp
            try:
                smooth_a_plgs = imantics.Mask(a_plg_mask).polygons()
                max_len = 0
                max_i = 0
                for i, plg in enumerate(smooth_a_plgs):
                    if max_len < len(plg):
                        max_i = i

                smooth_a_plg = smooth_a_plgs[max_i]
                smooth_a_plg = smooth_a_plg.reshape((int(len(smooth_a_plg)/2), 2))
                if len(smooth_a_plg) < 100:
                    continue
            except:
                continue
            
            ##### dán hình 
            # paste_x = int(random.uniform(970,1270)) #960*2 - 960*2/3-10
            paste_x = 960
            paste_y = 0 
            
            res_im.paste(a_plg_img, 
                            (paste_x, paste_y), 
                            mask=a_plg_mask)
            
            a_box[0] = a_box[0] + paste_x ## trừ tọa độ cũ rồi mới shift

            smooth_a_plg[:, 0] += paste_x 
            smooth_a_plg[:, 1] += paste_y

            # Gausian blur for edge
            a_edge_mask = get_edges_mask(smooth_a_plg, res_im.size[1], res_im.size[0])
            res_im = blur_edge_points(res_im, a_edge_mask)

            ### lưu tên thằng đầu trước
            res_iname = splitext(a_iname)[0] 

            # res_im.save('/home/tqsang/chick_segmentation/my_dt2-main/tools/a/syndata_example_ahp.png')
            ### random gà trái phải
            flag_a1 =  random.choice([1,2,3])
            flag_a2 =  random.choice([1,2,3])
        

            if flag_a1 != 1:
                a1_cat_id = random.choice(a_catIds)
                a1_img_id = random.choice(a_coco.getImgIds(catIds=a1_cat_id))

                try:
                    a1_cv_img, a1_plg, a1_box, a1_ctr, a1_Sbox, a1_iname = get_img_anns(a1_img_id, a1_cat_id, add_dataDir, dataType, a_coco)[0]
                    while a1_box[0] > 950:
                        # import pdb;pdb.set_trace()
                        a_coco = COCO(a_annFile)
                        a1_cv_img, a1_plg, a1_box, a1_ctr, a1_Sbox, a1_iname = get_img_anns(a1_img_id, a1_cat_id, add_dataDir, dataType, a_coco)[0]

                    a1_im, a1_plg_img, a1_plg_mask = get_cropped_img_and_mask(a1_cv_img, a1_plg, a1_box)

                except:
                    continue
                
                ## lưu tên 
                res_iname +=  '__' + splitext(a1_iname)[0] 
                a1_plg_mask = a1_plg_mask.convert('L')

                
                #### smooth hình cho đẹp
                try:
                    smooth_a1_plgs = imantics.Mask(a1_plg_mask).polygons()
                    max_len1 = 0
                    max_i1 = 0
                    for i, plg in enumerate(smooth_a1_plgs):
                        if max_len1 < len(plg):
                            max_i1 = i

                    smooth_a1_plg = smooth_a1_plgs[max_i1]
                    smooth_a1_plg = smooth_a1_plg.reshape((int(len(smooth_a1_plg)/2), 2))
                    if len(smooth_a1_plg) < 100:
                        continue
                except:
                    continue
                
                ##### dán hình 
                # paste_x1 = int(random.uniform(0,310-100)) # 960 - 960*2/3-10
                paste_x1 = 0 # 960 - 960*2/3-10
                paste_y = 0 

                res_im1 = res_im.copy()
                res_im1.paste(a1_plg_img, 
                                (paste_x1, paste_y), 
                                mask=a1_plg_mask)

                a1_box[0] = a1_box[0] + paste_x1 ## trừ tọa độ cũ rồi mới shift

                smooth_a1_plg[:, 0] += paste_x1
                # smooth_a1_plg[:, 0] += int(a1_box[0]) 
                smooth_a1_plg[:, 1] += paste_y

                # Gausian blur for edge
                a1_edge_mask = get_edges_mask(smooth_a1_plg, res_im1.size[1], res_im1.size[0])
                res_im1 = blur_edge_points(res_im1, a1_edge_mask)
                res_im = res_im1.copy()

            if flag_a2 != 1:
                a2_cat_id = random.choice(a_catIds)
                a2_img_id = random.choice(a_coco.getImgIds(catIds=a2_cat_id))
                #a_cat_id = 1
                #a_img_id = 10

                try:
                    a2_cv_img, a2_plg, a2_box, a2_ctr, a2_Sbox, a2_iname = get_img_anns(a2_img_id, a2_cat_id, add_dataDir, dataType, a_coco)[0]
                    a2_im, a2_plg_img, a2_plg_mask = get_cropped_img_and_mask(a2_cv_img, a2_plg, a2_box)
                except:
                    continue


                ## lưu tên 
                res_iname +=  '__' + splitext(a2_iname)[0] 

                a2_plg_mask = a2_plg_mask.convert('L')

                
                #### smooth hình cho đẹp
                try:
                    smooth_a2_plgs = imantics.Mask(a2_plg_mask).polygons()
                    max_len2 = 0
                    max_i2 = 0
                    for i, plg in enumerate(smooth_a2_plgs):
                        if max_len2 < len(plg):
                            max_i2 = i

                    smooth_a2_plg = smooth_a2_plgs[max_i2]
                    smooth_a2_plg = smooth_a2_plg.reshape((int(len(smooth_a2_plg)/2), 2))
                    if len(smooth_a2_plg) < 100:
                        continue
                except:
                    continue
                
                ##### dán hình 
                # paste_x2 = int(random.uniform(1920,2230-100)) # 960*2,960*3 - 960*2/3-10
                paste_x2 = 960*2 # 960*2,960*3 - 960*2/3-10

                paste_y = 0 

                res_im2 = res_im.copy()
                res_im2.paste(a2_plg_img, 
                                (paste_x2, paste_y), 
                                mask=a2_plg_mask)


                a2_box[0] = a2_box[0] + paste_x2 ## trừ tọa độ cũ rồi mới shift

                smooth_a2_plg[:, 0] += paste_x2
                # smooth_a2_plg[:, 0] += int(a2_box[0])
                smooth_a2_plg[:, 1] += paste_y

                # Gausian blur for edge
                a2_edge_mask = get_edges_mask(smooth_a2_plg, res_im2.size[1], res_im2.size[0])
                res_im2 = blur_edge_points(res_im2, a2_edge_mask)
                res_im = res_im2.copy()


            res_iname += '.jpg'

            if Path(join(syn_dataDir, '{}2017/{}'.format(dataType, res_iname))).exists():
                continue
            res_im.save(join(syn_dataDir, '{}2017/{}'.format(dataType, res_iname)))


            ii+=1
            ann_dict["images"].append(create_json_img_format(res_im, res_iname, ii))


            seg_ii+=1
            # sửa box
            # a_box[0] += paste_x
            # a_box[1] = paste_y
            # a_box[2] = res_im.size[0] - a_box[0] - 1
            # a_box[3] = res_im.size[1] - a_box[1] - 1

            ann_dict["annotations"].append(
                create_json_ann_format(
                        ii, smooth_a_plg, a_box, catId, seg_ii)
            )

            if flag_a1 != 1:
                seg_ii+=1
                # sửa box
                # a1_box[0] += paste_x1
                # a1_box[1] = paste_y 


                ann_dict["annotations"].append(
                    create_json_ann_format(
                            ii, smooth_a1_plg, a1_box, a1_cat_id, seg_ii)
                )

            if flag_a2 != 1:
                seg_ii+=1
                # sửa box
                # a2_box[0] += paste_x2
                # a2_box[1] = paste_y 

                ann_dict["annotations"].append(
                    create_json_ann_format(
                            ii, smooth_a2_plg, a2_box, a2_cat_id, seg_ii)
                )

            # res_im.save('/home/tqsang/chick_segmentation/my_dt2-main/tools/a/syndata_example_ahp.jpg')
            # exit(0)
            
                
    dst_path = join(syn_dataDir, 'annotations/instances_{}2017.json'.format(dataType))
    with open(dst_path, 'w') as fp:
        json.dump(ann_dict, fp)

    print('ii', ii)
    print('seg_ii', seg_ii)
