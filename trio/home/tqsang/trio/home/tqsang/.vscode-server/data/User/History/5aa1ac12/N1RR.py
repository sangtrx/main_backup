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


N_RAND_IMAGES=1

def draw_multiple_boxes(img, bboxes, class_names, image_name):
    for i, bbox in enumerate(bboxes):
        x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
        cv2.putText(img, class_names[i], (int(x1), int(y1)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imwrite("../data/outtest/" + image_name + ".png", img)

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
    img3.save('../data/outtest/plg_drawing.png')

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

        # assert len(ann['segmentation']) == 1
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
      "license": None,
      "file_name": iname,
      "coco_url": None,
      "height": im.size[1],
      "width": im.size[0],
      "id": id
    }

def create_json_ann_format(im_id, 
                            plg,
                            box,
                            cat_id, seg_ii, inmodal_bbox=None, inmodal_seg=None):
    
    x, y, w, h = box
    return {
      "segmentation": [
          plg.reshape((plg.shape[0]*plg.shape[1])).tolist()
      ],
      "area": w*h,
      "iscrowd": 0,
      "image_id": im_id,
      "bbox": box,
      "category_id": cat_id,
      "id": seg_ii,
    }


if __name__ == "__main__":


# Load the original annotations file
with open('annotations.json', 'r') as f:
    annotations = json.load(f)

# Create a list of all the image IDs
image_ids = [anno['image_id'] for anno in annotations['annotations']]

# Copy and paste feathers to existing samples to create new images
new_annotations = []
num_new_images_per_sample = 10
num_feathers_per_image = 10
for image_id in image_ids:
    source_image_anno = [anno for anno in annotations['annotations'] if anno['image_id'] == image_id][0]
    for i in range(num_new_images_per_sample):
        # Create a new image ID
        new_image_id = max(image_ids) + 1
        image_ids.append(new_image_id)
        
        # Load the original image
        im = Image.open('image_' + str(image_id) + '.jpg')

        # Create a new image with the same dimensions
        new_im = Image.new('RGB', im.size)

        # Randomly select feathers from the source image
        source_feathers = source_image_anno['segmentation']
        selected_feathers = random.sample(source_feathers, num_feathers_per_image)
        
        # Loop through the selected feathers
        for feather in selected_feathers:
            # Randomly select a location to paste the feather
            x_min, y_min, width, height = source_image_anno['bbox']
            x_max = x_min + width
            y_max = y_min + height
            new_x_min = random.uniform(0, 1) * (image_width - width)
            new_y_min = random.uniform(0, 1) * (image_height - height)
            new_x_max = new_x_min + width
            new_y_max = new_y_min + height
            
            # Paste the copied feather onto the new image
            new_im.paste(im, (int(new_x_min), int(new_y_min)))

            # Update the annotation for the new image
            new_annotation = {
                'image_id': new_image_id,
                'category_id': source_image_anno['category_id'],
                'segmentation': feather,
                'bbox': [new_x_min, new_y_min, new_x_max, new_y_max],
                'iscrowd': 0
            }
            new_annotations.append(new_annotation)
        # Save the new image
        new_im.save('image_' + str(new_image_id) + '.jpg')
annotations['annotations'].extend(new_annotations)
annotations['images'] = [{'id': image_id} for image_id in image_ids]

# Save the updated annotations file
with

