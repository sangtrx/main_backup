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

    polygon_cropped_img = polygon_cropped_img[int(y): int(y + h), int(x): int(x + w)]
    polygon_cropped_mask = mask[int(y): int(y + h), int(x): int(x + w)]


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

    dataDir = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_pad_COCO"
    add_dataDir = "/home/tqsang/V100/tqsang/crop_obj/front_2_class_pad_COCO"

    # TODO declare dataType
    dataType='train'
    annFile='{}/annotations/instances_{}2017.json'.format(dataDir,dataType)
    a_annFile='{}/annotations/instances_{}2017.json'.format(add_dataDir,dataType)

    # TODO create syn data dir - CHANGE DIR HERE
    # syn_dataDir = "../data/chicken_data/syn_2_chickens"
    syn_dataDir = "/home/tqsang/V100/tqsang/crop_obj/back_COCO_pad_synthetic_refine"

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
        {'supercategory': 'shape', 'id': 1, 'name': 'mask'}
    ]

    INFO = {
        "description": "back",
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
            'name': 'mask',
            'supercategory': 'shape',
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
    ii=0
    seg_ii=0
    for catId in tqdm(catIds):
        imgIds = coco.getImgIds(catIds=catId)
        for imgId in tqdm(imgIds):
            # get main ones
            # if len(get_img_anns(imgId, catId, dataDir, dataType, coco)) > 1:
            #     continue
            try:
                m_cv_img, m_plg, m_box, m_ctr, m_Sbox, m_iname = random.choice(get_img_anns(imgId, catId, dataDir, dataType, coco))
            except:
                continue

            m_im, _, m_plg_mask= get_cropped_img_and_mask(m_cv_img, m_plg, m_box)

            # get random n images from the data to create synthetic pairs
            for index_thutu in range(N_RAND_IMAGES):
                # get added ones

                a_cat_id = random.choice(a_catIds)
                a_img_id = random.choice(a_coco.getImgIds(catIds=a_cat_id))
                #a_cat_id = 1
                #a_img_id = 10

                try:
                    a_cv_img, a_plg, a_box, a_ctr, a_Sbox, a_iname = get_img_anns(a_img_id, a_cat_id, add_dataDir, dataType, a_coco)[0]
                    a_im, a_plg_img, a_plg_mask = get_cropped_img_and_mask(a_cv_img, a_plg, a_box)
                except:
                    continue


                # scale = euler_distance_to_bottom(m_im, (m_ctr['x'], m_ctr['y'])) \
                #     / euler_distance_to_bottom(m_im, (a_ctr['x'], a_ctr['y']))

                # scale = m_Sbox / a_Sbox
                # if scale < 0.8 or scale > 1.2:
                #     continue

                # scale = 1

                res_im = m_im.copy()
                # scale_w = (random.uniform(0.5, 0.8) * m_box[2]) / (a_box[2])
                # scale_h = (random.uniform(0.6, 0.8) * m_box[3]) / (a_box[3])
                scale_w = 1
                scale_h = 1

                a_plg_mask = a_plg_mask.convert('L')
                a_plg_mask = resize_pil_img(a_plg_mask, scale_w, scale_h)
                # a_plg_mask = morphology_smooth(a_plg_mask)
                a_plg_img = resize_pil_img(a_plg_img, scale_w, scale_h)
                a_im = resize_pil_img(a_im, scale_w, scale_h)

                # re-get the added polygon after scale and smooth
                # a_plg = imantics.Mask(a_plg_mask).polygons()[0]
                # a_plg = a_plg.reshape((int(len(a_plg)/2), 2))

                # create a good mask for smoothing but not use for annotation, no, we're now using it
                # smooth_a_plg = None

                # import pdb;pdb.set_trace()

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
                

                oke = random.choice([1,2])
                rand_x = 0
                if oke == 1: # right
                    # rand_x = random.uniform(m_ctr['x'] + m_box[2]/6, 
                    #                         m_ctr['x'] + m_box[2]/4)
                    rand_x = random.uniform(m_ctr['x'] + 120, 
                                            m_ctr['x'] + 150)

                elif oke == 2: # left
                    # rand_x = random.uniform(m_ctr['x'] -  m_box[2]*1.2, 
                    #                         m_ctr['x'] - m_box[2]*1.25)
                    rand_x = random.uniform(m_ctr['x'] -  780, 
                                            m_ctr['x'] - 720)

                rand_y = m_ctr['y'] - m_box[3] / 2
                
                paste_x = int(rand_x)
                paste_y = int(rand_y)

                #print(paste_x, paste_y)
                #paste_x = 242
                #paste_y = 227

                res_im.paste(a_plg_img, 
                                (paste_x, paste_y), 
                                mask=a_plg_mask)

                pasted_fsize_mask = get_fullsize_mask(a_plg_mask, (paste_x, paste_y), m_im.size)
                ori_fsize_mask = get_fullsize_mask(a_plg_mask, (int(a_box[0]), int(a_box[1])), m_im.size)
                # res_im = average_smooth(res_im, m_im, pasted_fsize_mask, a_im, ori_fsize_mask, a_box, (paste_x, paste_y))

                '''
                a_plg[:, 0] -= a_box[0]
                a_plg[:, 1] -= a_box[1]
                a_plg[:, 0] += paste_x
                a_plg[:, 1] += paste_y
                '''

                a_box[2] *= scale_w
                a_box[3] *= scale_h
                smooth_a_plg[:, 0] += paste_x
                smooth_a_plg[:, 1] += paste_y

                # Gausian blur for edge
                a_edge_mask = get_edges_mask(smooth_a_plg, res_im.size[1], res_im.size[0])
                res_im = blur_edge_points(res_im, a_edge_mask)

                # res_im.save('../data/outtest/syndata_example_ahp.png')


                # mk_my_dir('../data/outtest/syn_examples/')
                # res_im.save('../data/outtest/syn_examples/out%d.png'%ii)
                # ii+=1
                # if ii==40:
                #     exit(0)
                # else:
                #     continue

                # res_iname = splitext(m_iname)[0] + '__' + splitext(a_iname)[0] + '_{}'.format(str(index_thutu)) + '.png'
                res_iname = splitext(m_iname)[0] + '__' + splitext(a_iname)[0] + '_{}'.format(str(index_thutu)) + '.jpg'


                if Path(join(syn_dataDir, '{}2017/{}'.format(dataType, res_iname))).exists():
                    continue
                res_im.save(join(syn_dataDir, '{}2017/{}'.format(dataType, res_iname)))

                
                ii+=1
                ann_dict["images"].append(create_json_img_format(res_im, res_iname, ii))

                '''
                for ret in get_img_anns(imgId, catId, dataDir, dataType, coco):
                    seg_ii+=1
                    _, m_plg, m_box, _, _, _ = ret
                    
                    ann_dict["annotations"].append(
                        create_json_ann_format(
                                ii, m_plg, m_box, catId, seg_ii)
                    )
                '''
                # get inmodal mask
                m_fsize_mask = get_fullsize_mask(m_plg_mask, (int(m_box[0]), int(m_box[1])), m_im.size)
                cv_pasted_fsize_mask = from_pil_to_cv(pasted_fsize_mask)
                cv_m_fsize_mask = from_pil_to_cv(m_fsize_mask)

                cv_m_inmodal_mask = cv_m_fsize_mask - cv2.bitwise_and(cv_pasted_fsize_mask, cv_m_fsize_mask)
                m_inmodal_mask = from_cv_to_pil(cv_m_inmodal_mask).convert('L')
                m_inmodal_plgs = imantics.Mask(m_inmodal_mask).polygons()
                m_inmodal_plgs_anns = []
                for plg in m_inmodal_plgs:
                    m_inmodal_plgs_anns.append(plg.tolist())

                m_inmodal_bin_mask = deepcopy(np.asarray(m_inmodal_mask))
                m_inmodal_bin_mask[m_inmodal_bin_mask>0] = 1
                m_inmodal_bin_mask_rle = maskUtils.encode(np.asfortranarray(m_inmodal_bin_mask))

                # pasted_fsize_mask.save('../data/outtest/pasted_fsize_mask.png')
                # m_fsize_mask.save('../data/outtest/m_fize_mask.png')
                # m_inmodal_mask.save('../data/outtest/m_inmodal_mask.png')
                # plt.imsave('../data/outtest/m_inmodal_bin_mask.png', m_inmodal_bin_mask)

                m_inmodal_bin_mask[m_inmodal_bin_mask>0] = 255
                m_inmodal_bbox = get_bbox_from_binary_mask(m_inmodal_bin_mask)

                seg_ii+=1
                ann_dict["annotations"].append(
                    create_json_ann_format(
                            ii, m_plg, m_box, catId, seg_ii, inmodal_bbox=m_inmodal_bbox, 
                            inmodal_seg=m_inmodal_plgs_anns)
                )
                # if oke == 0:
                #     continue

                seg_ii+=1
                dst_a_box = deepcopy(a_box)
                dst_a_box[0] = paste_x
                dst_a_box[1] = paste_y

                # if out of images
                if dst_a_box[0] < 0 or dst_a_box[1] > res_im.size[0]:
                    continue
                if dst_a_box[1] < 0 or dst_a_box[1] > res_im.size[1]:
                    continue

                # post process adding box and seg
                if dst_a_box[0] + dst_a_box[2] > res_im.size[0]:
                    dst_a_box[2] = res_im.size[0] - dst_a_box[0] - 1
                if dst_a_box[1] + dst_a_box[3] > res_im.size[1]:
                    dst_a_box[3] = res_im.size[1] - dst_a_box[1] - 1
                post_proc_a_plg = []
                for plg in smooth_a_plg:
                    if plg[0] < res_im.size[0] and plg[1] < res_im.size[1]:
                        post_proc_a_plg.append([plg[0], plg[1]])

                post_proc_a_plg = np.array(post_proc_a_plg)

                if len(post_proc_a_plg) <= 0:
                    continue

                # draw_polygon_on_image(res_im, [post_proc_a_plg])
                # draw_multiple_boxes(from_pil_to_cv(res_im), [m_box, dst_a_box], '01234', 'showboxes')

                a_inmodal_bin_mask = deepcopy(np.asarray(pasted_fsize_mask))
                a_inmodal_bin_mask[a_inmodal_bin_mask>0] = 1
                a_inmodal_bin_mask_rle = maskUtils.encode(np.asfortranarray(a_inmodal_bin_mask))

                # draw_polygon_on_image(res_im, [ m_inmodal_plg])
                # draw_multiple_boxes(from_pil_to_cv(res_im), [dst_a_box, m_inmodal_bbox], '01234', 'showboxes')
                # pasted_fsize_mask.save('../data/outtest/pasted_fsize_mask.png')

                ann_dict["annotations"].append(
                    create_json_ann_format(
                            ii, post_proc_a_plg, dst_a_box, a_cat_id, seg_ii,
                            inmodal_bbox=dst_a_box, 
                            inmodal_seg=[post_proc_a_plg.reshape((post_proc_a_plg.shape[0]*post_proc_a_plg.shape[1])).tolist()])
                )

                
    dst_path = join(syn_dataDir, 'annotations/instances_{}2017.json'.format(dataType))
    with open(dst_path, 'w') as fp:
        json.dump(ann_dict, fp)

    print('ii', ii)
    print('seg_ii', seg_ii)
