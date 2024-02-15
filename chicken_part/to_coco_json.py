import sys
sys.path.append("./")
from utils import *
from tqdm import tqdm
import json
import shutil


def process(img_folders, ann_folders, output_json_file, output_img_folder,
            ann_dict, img_id, ann_id, cam_pos, one_cat_only=True):

    for folder_name in tqdm(sorted(os.listdir(ann_folders))):
        img_folder = join(img_folders, folder_name)
        ann_folder = join(ann_folders, folder_name)
        if len(listdir(img_folder)) != len(listdir(ann_folder)):
            print('a problem')
            continue
        for ann_fname in sorted(listdir(ann_folder)):
            # img
            img_name = os.path.splitext(ann_fname)[0] + '.png'
            img_fpath = join(img_folder, img_name)
            img = cv2.imread(img_fpath)
            img_h, img_w = img.shape[:2]

            new_img_name = '_'.join(img_fpath.split('/')[-3:])
            new_img_path = join(output_img_folder, new_img_name)
            img_id += 1
            shutil.copyfile(img_fpath, new_img_path)
            ann_dict["images"].append(create_json_img_format(img_h, img_w, new_img_name, img_id))

            # ann
            ann_fpath = join(ann_folder, ann_fname)
            if '.xml' in ann_fname:
                box = get_box_from_xml(ann_fpath)
            elif '.png' in ann_fname:
                ann_img = cv2.imread(ann_fpath)
                box = get_box_from_ann_img(img ,ann_img)

            if one_cat_only:
                cat_id = 1
            else:
                if cam_pos == 'back':
                    cat_id = 1
                elif cam_pos == 'side':
                    cat_id = 2

            assert len(box) == 4
            ann_id += 1
            #show_box_xyxy(box, img)
            #import pdb;pdb.set_trace()
            ann_dict["annotations"].append(
                                        create_json_ann_format(
                                            im_id=img_id, 
                                            polygon=None,
                                            cxcywh_box=box_xyxy_to_xywh(box),
                                            cat_id=cat_id, 
                                            ann_id=ann_id
                                        )
                                    )    

    return ann_dict, img_id, ann_id




def full():
    base_ann_folders = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/box_annotated_segments/' ### xml
    base_img_folders = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/significant_segments/'
    output_folder = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/full_chicken_boxdet_fs/'
    output_json_file = join(output_folder, 'annotations/all.json')
    output_img_folder = join(output_folder, 'imgs/')

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(join(output_folder, 'annotations/'), exist_ok=True)
    os.makedirs(output_img_folder, exist_ok=True)
   

    cats = [
        {'supercategory': 'chicken', 'id': 1, 'name': 'back_chicken'},
        {'supercategory': 'chicken', 'id': 2, 'name': 'side_chicken'},
    ]
    ann_dict = init_ann_coco_json_dict(cats)
    img_id, ann_id = 0, 0

    for cam_pos in ['back', 'side']:
        img_folders = join(base_img_folders, cam_pos)
        ann_folders = join(base_ann_folders, cam_pos)
        ann_dict, img_id, ann_id = process(img_folders, ann_folders, output_json_file, output_img_folder, 
                ann_dict=ann_dict, img_id=img_id, ann_id=ann_id, cam_pos=cam_pos, one_cat_only=False)

    with open(output_json_file, 'w') as fp:
        json.dump(ann_dict, fp)

    print(img_id, ann_id)


def side():
    base_ann_folders = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/box_annotated_segments/'
    base_img_folders = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/significant_segments/'
    output_folder = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/side_chicken_boxdet_fs/'
    output_json_file = join(output_folder, 'annotations/all.json')
    output_img_folder = join(output_folder, 'imgs/')

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(join(output_folder, 'annotations/'), exist_ok=True)
    os.makedirs(output_img_folder, exist_ok=True)
   

    cats = [
        {'supercategory': 'chicken', 'id': 1, 'name': 'side_chicken'},
    ]
    ann_dict = init_ann_coco_json_dict(cats)
    img_id, ann_id = 0, 0

    for cam_pos in ['side']:
        img_folders = join(base_img_folders, cam_pos)
        ann_folders = join(base_ann_folders, cam_pos)
        ann_dict, img_id, ann_id = process(img_folders, ann_folders, output_json_file, output_img_folder, 
                ann_dict=ann_dict, img_id=img_id, ann_id=ann_id, cam_pos=cam_pos)

    with open(output_json_file, 'w') as fp:
        json.dump(ann_dict, fp)

    print(img_id, ann_id)


def back():
    base_ann_folders = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/box_annotated_segments/' ## xml
    base_img_folders = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/significant_segments/' ## frames
    output_folder = '/home/tqminh/Downloads/Data/Gait/ChickenGaitData_P2/back_chicken_boxdet_fs/'  ## json
    output_json_file = join(output_folder, 'annotations/all.json')
    output_img_folder = join(output_folder, 'imgs/')

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(join(output_folder, 'annotations/'), exist_ok=True)
    os.makedirs(output_img_folder, exist_ok=True)
   
    cats = [
        {'supercategory': 'chicken', 'id': 1, 'name': 'back_chicken'},
    ]
    ann_dict = init_ann_coco_json_dict(cats)
    img_id, ann_id = 0, 0

    for cam_pos in ['back']:
        img_folders = join(base_img_folders, cam_pos)
        ann_folders = join(base_ann_folders, cam_pos)
        ann_dict, img_id, ann_id = process(img_folders, ann_folders, output_json_file, output_img_folder, 
                ann_dict=ann_dict, img_id=img_id, ann_id=ann_id, cam_pos=cam_pos)

    with open(output_json_file, 'w') as fp:
        json.dump(ann_dict, fp)

    print(img_id, ann_id)


if __name__=='__main__':
    full()
    side()
    back()
