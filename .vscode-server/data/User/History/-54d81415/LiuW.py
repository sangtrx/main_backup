# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# # ==== Predefined datasets and splits for COCO ==========

# _PREDEFINED_SPLITS_COCO = {}
# _PREDEFINED_SPLITS_COCO["coco"] = {
#     "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
#     "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
#     "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
#     "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
#     "coco_2014_valminusminival": (
#         "coco/val2014",
#         "coco/annotations/instances_valminusminival2014.json",
#     ),
#     "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
#     "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
#     "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
#     "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
#     "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
# }

# _PREDEFINED_SPLITS_COCO["coco_person"] = {
#     "keypoints_coco_2014_train": (
#         "coco/train2014",
#         "coco/annotations/person_keypoints_train2014.json",
#     ),
#     "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
#     "keypoints_coco_2014_minival": (
#         "coco/val2014",
#         "coco/annotations/person_keypoints_minival2014.json",
#     ),
#     "keypoints_coco_2014_valminusminival": (
#         "coco/val2014",
#         "coco/annotations/person_keypoints_valminusminival2014.json",
#     ),
#     "keypoints_coco_2014_minival_100": (
#         "coco/val2014",
#         "coco/annotations/person_keypoints_minival2014_100.json",
#     ),
#     "keypoints_coco_2017_train": (
#         "coco/train2017",
#         "coco/annotations/person_keypoints_train2017.json",
#     ),
#     "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
#     "keypoints_coco_2017_val_100": (
#         "coco/val2017",
#         "coco/annotations/person_keypoints_val2017_100.json",
#     ),
# }


# _PREDEFINED_SPLITS_COCO_PANOPTIC = {
#     "coco_2017_train_panoptic": (
#         # This is the original panoptic annotation directory
#         "coco/panoptic_train2017",
#         "coco/annotations/panoptic_train2017.json",
#         # This directory contains semantic annotations that are
#         # converted from panoptic annotations.
#         # It is used by PanopticFPN.
#         # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
#         # to create these directories.
#         "coco/panoptic_stuff_train2017",
#     ),
#     "coco_2017_val_panoptic": (
#         "coco/panoptic_val2017",
#         "coco/annotations/panoptic_val2017.json",
#         "coco/panoptic_stuff_val2017",
#     ),
#     "coco_2017_val_100_panoptic": (
#         "coco/panoptic_val2017_100",
#         "coco/annotations/panoptic_val2017_100.json",
#         "coco/panoptic_stuff_val2017_100",
#     ),
# }


# def register_all_coco(root):
#     for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
#         for key, (image_root, json_file) in splits_per_dataset.items():
#             # Assume pre-defined datasets live in `./datasets`.
#             register_coco_instances(
#                 key,
#                 _get_builtin_metadata(dataset_name),
#                 os.path.join(root, json_file) if "://" not in json_file else json_file,
#                 os.path.join(root, image_root),
#             )

#     for (
#         prefix,
#         (panoptic_root, panoptic_json, semantic_root),
#     ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
#         prefix_instances = prefix[: -len("_panoptic")]
#         instances_meta = MetadataCatalog.get(prefix_instances)
#         image_root, instances_json = instances_meta.image_root, instances_meta.json_file
#         # The "separated" version of COCO panoptic segmentation dataset,
#         # e.g. used by Panoptic FPN
#         register_coco_panoptic_separated(
#             prefix,
#             _get_builtin_metadata("coco_panoptic_separated"),
#             image_root,
#             os.path.join(root, panoptic_root),
#             os.path.join(root, panoptic_json),
#             os.path.join(root, semantic_root),
#             instances_json,
#         )
#         # The "standard" version of COCO panoptic segmentation dataset,
#         # e.g. used by Panoptic-DeepLab
#         register_coco_panoptic(
#             prefix,
#             _get_builtin_metadata("coco_panoptic_standard"),
#             image_root,
#             os.path.join(root, panoptic_root),
#             os.path.join(root, panoptic_json),
#             instances_json,
#         )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

def register_chick():
    try:
        DatasetCatalog.get("chick_dataset_train")
    except:
        register_coco_instances("chick_dataset_train", {}, 
            "/home/tqsang/Mask2Former/datasets/coco_front2class/annotations/instances_train2017.json", 
            "/home/tqsang/Mask2Former/datasets/coco_front2class/train2017"
        )
    try:
        DatasetCatalog.get("chick_dataset_val")
    except:
        register_coco_instances("chick_dataset_val", {}, 
            "/home/tqsang/Mask2Former/datasets/coco_front2class/annotations/instances_val2017.json", 
            "/home/tqsang/Mask2Former/datasets/coco_front2class/val2017"
        )

    try:
        meta = MetadataCatalog.get('chick_dataset_train')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('chick_dataset_val')
        meta.thing_classes =['normal','defect']
    except:
        pass

def register_chick_syn():
    try:
        DatasetCatalog.get("chick_dataset_train_syn")
    except:
        register_coco_instances("chick_dataset_train_syn", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/annotations/instances_train2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/train2017"
        )
    try:
        DatasetCatalog.get("chick_dataset_val_syn")
    except:
        register_coco_instances("chick_dataset_val_syn", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/annotations/instances_val2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/val2017"
        )

    try:
        meta = MetadataCatalog.get('chick_dataset_train_syn')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('chick_dataset_val_syn')
        meta.thing_classes =['normal','defect']
    except:
        pass
##################### new #########################################################################


def register_chick_new():
    try:
        DatasetCatalog.get("chick_dataset_train_new")
    except:
        register_coco_instances("chick_dataset_train_new", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/instances_train2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/train2017"
        )
    try:
        DatasetCatalog.get("chick_dataset_val_new")
    except:
        register_coco_instances("chick_dataset_val_new", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/instances_val2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/val2017"
        )
    try:
        DatasetCatalog.get("chick_dataset_test_new")
    except:
        register_coco_instances("chick_dataset_test_new", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/instances_test2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/test2017"
        )
    try:
        meta = MetadataCatalog.get('chick_dataset_train_new')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('chick_dataset_val_new')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('chick_dataset_test_new')
        meta.thing_classes =['normal','defect']
    except:
        pass

def register_chick_syn_new():
    try:
        DatasetCatalog.get("chick_dataset_train_syn_new")
    except:
        register_coco_instances("chick_dataset_train_syn_new", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/instances_train2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/train2017"
        )
    try:
        DatasetCatalog.get("chick_dataset_val_syn_new")
    except:
        register_coco_instances("chick_dataset_val_syn_new", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/instances_val2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/val2017"
        )
    try:
        DatasetCatalog.get("chick_dataset_test_syn_new")
    except:
        register_coco_instances("chick_dataset_test_syn_new", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/instances_test2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/test2017"
        )
    try:
        meta = MetadataCatalog.get('chick_dataset_train_syn_new')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('chick_dataset_val_syn_new')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('chick_dataset_test_syn_new')
        meta.thing_classes =['normal','defect']
    except:
        pass


# ==== Predefined datasets and splits for COCO ==========




def register_all_front2class(root):
    _PREDEFINED_SPLITS_COCO = {}
    _PREDEFINED_SPLITS_COCO["coco"] = {
        "front2class_2017_train": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/train2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/annotations/instances_train2017.json"),
        "front2class_2017_val": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/val2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/annotations/instances_val2017.json"),
    }


    _PREDEFINED_SPLITS_COCO_PANOPTIC = {
        "front2class_2017_train_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/panoptic_train2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/annotations/panoptic_train2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/panoptic_semseg_train2017",
        ),
        "front2class_2017_val_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/panoptic_val2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/annotations/panoptic_val2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/panoptic_semseg_val2017",
        ),
    }

    try:
        meta = MetadataCatalog.get('front2class_2017_train')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_val')
        meta.thing_classes =['normal','defect']
    except:
        pass
    root = ''

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        # import pdb;pdb.set_trace()
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
    try:
        meta = MetadataCatalog.get('front2class_2017_train_panoptic')
        meta.thing_classes =['normal','defect']
        # meta = MetadataCatalog.get('front2class_2017_val_panoptic_with_sem_seg')
        # meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_val_panoptic')
        meta.thing_classes =['normal','defect']
    except:
        pass

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )

#######################################################################################################



def register_all_front2class_new(root):
    
    _PREDEFINED_SPLITS_COCO = {}
    _PREDEFINED_SPLITS_COCO["coco"] = {
        "front2class_2017_train_new": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/train2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/instances_train2017.json"),
        "front2class_2017_val_new": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/val2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/instances_val2017.json"),
        "front2class_2017_test_new": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/test2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/instances_test2017.json"),
    }


    _PREDEFINED_SPLITS_COCO_PANOPTIC = {
        "front2class_2017_train_new_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/panoptic_train2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/panoptic_train2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/panoptic_semseg_train2017",
        ),
        "front2class_2017_val_new_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/panoptic_val2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/panoptic_val2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/panoptic_semseg_val2017",
        ),
        "front2class_2017_test_new_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/panoptic_val2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/annotations/panoptic_val2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_syn_bg/panoptic_semseg_val2017",
        ),
    }
    try:
        meta = MetadataCatalog.get('front2class_2017_train_new')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_val_new')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_test_new')
        meta.thing_classes =['normal','defect']
    except:
        pass
    root = ''

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        # import pdb;pdb.set_trace()
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
    try:
        meta = MetadataCatalog.get('front2class_2017_train_new_panoptic')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_val_new_panoptic')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_test_new_panoptic')
        meta.thing_classes =['normal','defect']
    except:
        pass

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )

###### panoptic cho single
def register_all_front2class_single_new(root):
    
    _PREDEFINED_SPLITS_COCO = {}
    _PREDEFINED_SPLITS_COCO["coco"] = {
        "front2class_single_2017_train_new": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/train2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/instances_train2017.json"),
        "front2class_single_2017_val_new": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/val2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/instances_val2017.json"),
        "front2class_single_2017_test_new": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/test2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/instances_test2017.json"),
    }


    _PREDEFINED_SPLITS_COCO_PANOPTIC = {
        "front2class_single_2017_train_new_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/panoptic_train2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/panoptic_train2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/panoptic_semseg_train2017",
        ),
        "front2class_single_2017_val_new_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/panoptic_val2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/panoptic_val2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/panoptic_semseg_val2017",
        ),
        "front2class_single_2017_test_new_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/panoptic_val2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/annotations/panoptic_val2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_new_COCO/panoptic_semseg_val2017",
        ),
    }
    try:
        meta = MetadataCatalog.get('front2class_single_2017_train_new')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_single_2017_val_new')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_single_2017_test_new')
        meta.thing_classes =['normal','defect']
    except:
        pass
    root = ''

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        # import pdb;pdb.set_trace()
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
    try:
        meta = MetadataCatalog.get('front2class_single_2017_train_new_panoptic')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_single_2017_val_new_panoptic')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_single_2017_test_new_panoptic')
        meta.thing_classes =['normal','defect']
    except:
        pass

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )

def register_all_front2class_combine(root):
    
    _PREDEFINED_SPLITS_COCO = {}
    _PREDEFINED_SPLITS_COCO["coco"] = {
        "front2class_2017_train_combine": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/train2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/annotations/instances_train2017.json"),
        "front2class_2017_val_combine": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/val2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/annotations/instances_val2017.json"),
        "front2class_2017_test_combine": ("/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/test2017", "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/annotations/instances_test2017.json"),
    }


    _PREDEFINED_SPLITS_COCO_PANOPTIC = {
        "front2class_2017_train_combine_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/panoptic_train2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/annotations/panoptic_train2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/panoptic_semseg_train2017",
        ),
        "front2class_2017_val_combine_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/panoptic_val2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/annotations/panoptic_val2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/panoptic_semseg_val2017",
        ),
        "front2class_2017_test_combine_panoptic": (
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/panoptic_val2017",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/annotations/panoptic_val2017.json",
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_combine/panoptic_semseg_val2017",
        ),
    }
    try:
        meta = MetadataCatalog.get('front2class_2017_train_combine')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_val_combine')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_test_combine')
        meta.thing_classes =['normal','defect']
    except:
        pass
    root = ''

    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        # import pdb;pdb.set_trace()
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
    try:
        meta = MetadataCatalog.get('front2class_2017_train_combine_panoptic')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_val_combine_panoptic')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('front2class_2017_test_combine_panoptic')
        meta.thing_classes =['normal','defect']
    except:
        pass

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )

def register_chick_overlap():
    try:
        DatasetCatalog.get("chick_dataset_train_overlap")
    except:
        register_coco_instances("chick_dataset_train_overlap", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_overlap/annotations/instances_train2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_overlap/train2017"
        )
    try:
        DatasetCatalog.get("chick_dataset_val_overlap")
    except:
        register_coco_instances("chick_dataset_val_overlap", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_overlap/annotations/instances_val2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_overlap/val2017"
        )
    try:
        DatasetCatalog.get("chick_dataset_test_overlap")
    except:
        register_coco_instances("chick_dataset_test_overlap", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_overlap/annotations/instances_test2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_overlap/test2017"
        )
    try:
        meta = MetadataCatalog.get('chick_dataset_train_overlap')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('chick_dataset_val_overlap')
        meta.thing_classes =['normal','defect']
        meta = MetadataCatalog.get('chick_dataset_test_overlap')
        meta.thing_classes =['normal','defect']
    except:
        pass

def register_local_defect():
    try:
        DatasetCatalog.get("local_defect_train")
    except:
        register_coco_instances("local_defect_train", {}, 
            "/home/tqsang/runs/labelme2coco/train.json", 
            "/mnt/tqsang/local_data"
        )
    try:
        DatasetCatalog.get("local_defect_val")
    except:
        register_coco_instances("local_defect_val", {}, 
            "/home/tqsang/runs/labelme2coco/val.json", 
            "/mnt/tqsang/local_data"
        )

    try:
        meta = MetadataCatalog.get('local_defect_train')
        meta.thing_classes =['feather', 'wing', 'skin', 'broken wing']
        meta = MetadataCatalog.get('local_defect_val')
        meta.thing_classes =['feather', 'wing', 'skin', 'broken wing']
    except:
        pass

def register_local_defect_feather():
    try:
        DatasetCatalog.get("feather_train")
    except:
        register_coco_instances("feather_train", {}, 
            "/mnt/tqsang/runs/feather/train.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("feather_val")
    except:
        register_coco_instances("feather_val", {}, 
            "/mnt/tqsang/runs/feather/val.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("feather_test")
    except:
        register_coco_instances("feather_test", {}, 
            "/mnt/tqsang/runs/feather/test.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        meta = MetadataCatalog.get('feather_train')
        meta.thing_classes =['feather']
        meta = MetadataCatalog.get('feather_val')
        meta.thing_classes =['feather']
        meta = MetadataCatalog.get('feather_test')
        meta.thing_classes =['feather']
    except:
        pass


def register_local_defect_back_skin():
    try:
        DatasetCatalog.get("back_skin_train")
    except:
        register_coco_instances("back_skin_train", {}, 
            "/mnt/tqsang/runs/back_skin/train.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("back_skin_val")
    except:
        register_coco_instances("back_skin_val", {}, 
            "/mnt/tqsang/runs/back_skin/val.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("back_skin_test")
    except:
        register_coco_instances("back_skin_test", {}, 
            "/mnt/tqsang/runs/back_skin/test.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        meta = MetadataCatalog.get('back_skin_train')
        meta.thing_classes =['hanging back skin']
        meta = MetadataCatalog.get('back_skin_val')
        meta.thing_classes =['hanging back skin']
        meta = MetadataCatalog.get('back_skin_test')
        meta.thing_classes =['hanging back skin']
    except:
        pass

def register_local_defect_breast_skin():
    try:
        DatasetCatalog.get("breast_skin_train")
    except:
        register_coco_instances("breast_skin_train", {}, 
            "/mnt/tqsang/runs/breast_skin/train.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("breast_skin_val")
    except:
        register_coco_instances("breast_skin_val", {}, 
            "/mnt/tqsang/runs/breast_skin/val.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("breast_skin_test")
    except:
        register_coco_instances("breast_skin_test", {}, 
            "/mnt/tqsang/runs/breast_skin/test.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        meta = MetadataCatalog.get('breast_skin_train')
        meta.thing_classes =['hanging breast skin']
        meta = MetadataCatalog.get('breast_skin_val')
        meta.thing_classes =['hanging breast skin']
        meta = MetadataCatalog.get('breast_skin_test')
        meta.thing_classes =['hanging breast skin']
    except:
        pass

def register_local_defect_broken_wing():
    try:
        DatasetCatalog.get("broken_wing_train")
    except:
        register_coco_instances("broken_wing_train", {}, 
            "/mnt/tqsang/runs/broken_wing/train.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("broken_wing_val")
    except:
        register_coco_instances("broken_wing_val", {}, 
            "/mnt/tqsang/runs/broken_wing/val.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("broken_wing_test")
    except:
        register_coco_instances("broken_wing_test", {}, 
            "/mnt/tqsang/runs/broken_wing/test.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        meta = MetadataCatalog.get('broken_wing_train')
        meta.thing_classes =['wing', 'broken wing']
        meta = MetadataCatalog.get('broken_wing_val')
        meta.thing_classes =['wing', 'broken wing']
        meta = MetadataCatalog.get('broken_wing_test')
        meta.thing_classes =['broken wing', 'wing']
    except:
        pass

def register_local_defect_neck_skin():
    try:
        DatasetCatalog.get("neck_skin_train")
    except:
        register_coco_instances("neck_skin_train", {}, 
            "/mnt/tqsang/runs/neck_skin/train.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("neck_skin_val")
    except:
        register_coco_instances("neck_skin_val", {}, 
            "/mnt/tqsang/runs/neck_skin/val.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        DatasetCatalog.get("neck_skin_test")
    except:
        register_coco_instances("neck_skin_test", {}, 
            "/mnt/tqsang/runs/neck_skin/test.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class/img/"
        )
    try:
        meta = MetadataCatalog.get('neck_skin_train')
        meta.thing_classes =['hanging neck skin']
        meta = MetadataCatalog.get('neck_skin_val')
        meta.thing_classes =['hanging neck skin']
        meta = MetadataCatalog.get('neck_skin_test')
        meta.thing_classes =['hanging neck skin']
    except:
        pass


def register_chicken_part():
    try:
        DatasetCatalog.get("chicken_part_train")
    except:
        register_coco_instances("chicken_part_train", {}, 
            "/mnt/tqsang/chicken_part/train.json", 
            "/mnt/tqsang/chicken_part/frames"
        )
    try:
        DatasetCatalog.get("chicken_part_val")
    except:
        register_coco_instances("chicken_part_val", {}, 
            "/mnt/tqsang/chicken_part/val.json", 
            "/mnt/tqsang/chicken_part/frames"
        )
    try:
        DatasetCatalog.get("chicken_part_test")
    except:
        register_coco_instances("chicken_part_test", {}, 
            "/mnt/tqsang/chicken_part/test.json", 
            "/mnt/tqsang/chicken_part/frames"
        )
    try:
        meta = MetadataCatalog.get('chicken_part_train')
        meta.thing_classes =['1', '2', '3', '4', '5', '6', '7', '8', '9']
        meta = MetadataCatalog.get('chicken_part_val')
        meta.thing_classes =['1', '2', '3', '4', '5', '6', '7', '8', '9']
        meta = MetadataCatalog.get('chicken_part_test')
        meta.thing_classes =['1', '2', '3', '4', '5', '6', '7', '8', '9']
    except:
        pass

def register_chicken_part_model1():
    try:
        DatasetCatalog.get("chicken_part_train")
    except:
        register_coco_instances("chicken_part_train", {}, 
            "/mnt/tqsang/chicken_part/train.json", 
            "/mnt/tqsang/chicken_part/frames"
        )
    try:
        DatasetCatalog.get("chicken_part_val")
    except:
        register_coco_instances("chicken_part_val", {}, 
            "/mnt/tqsang/chicken_part/val.json", 
            "/mnt/tqsang/chicken_part/frames"
        )
    try:
        DatasetCatalog.get("chicken_part_test")
    except:
        register_coco_instances("chicken_part_test", {}, 
            "/mnt/tqsang/chicken_part/test.json", 
            "/mnt/tqsang/chicken_part/frames"
        )
    try:
        meta = MetadataCatalog.get('chicken_part_train')
        meta.thing_classes =['1', '2', '3', '4', '5', '6', '7', '8', '9']
        meta = MetadataCatalog.get('chicken_part_val')
        meta.thing_classes =['1', '2', '3', '4', '5', '6', '7', '8', '9']
        meta = MetadataCatalog.get('chicken_part_test')
        meta.thing_classes =['1', '2', '3', '4', '5', '6', '7', '8', '9']
    except:
        pass

def register_chicken_feather():
    try:
        DatasetCatalog.get("chicken_feather_train")
    except:
        register_coco_instances("chicken_feather_train", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_feather/annotations/instances_train2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_feather/train2017/"
        )
    try:
        DatasetCatalog.get("chicken_feather_val")
    except:
        register_coco_instances("chicken_feather_val", {}, 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_feather/annotations/instances_val2017.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_feather/val2017/"
        )
    try:
        DatasetCatalog.get("chicken_feather_test")
    except:
        register_coco_instances("chicken_feather_test", {}, 
            "/mnt/tqsang/runs/neck_skin/test.json", 
            "/home/tqsang/V100/tqsang/crop_obj/front_2_class_feather/train2017/"
        )
    try:
        meta = MetadataCatalog.get('chicken_feather_train')
        meta.thing_classes =['feather']
        meta = MetadataCatalog.get('chicken_feather_val')
        meta.thing_classes =['feather']
        meta = MetadataCatalog.get('chicken_feather_test')
        meta.thing_classes =['feather']
    except:
        pass
# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
    register_chick()
    register_chick_syn()
    # register_all_front2class(_root)
    register_chick_new()
    register_chick_syn_new()
    # register_all_front2class_new(_root)
    # register_all_front2class_single_new(_root)
    # register_all_front2class_combine(_root)
    register_chick_overlap()
    register_local_defect()
    ####
    register_local_defect_feather()
    register_local_defect_back_skin()
    register_local_defect_breast_skin()
    register_local_defect_broken_wing()
    register_local_defect_neck_skin()

    ###
    register_chicken_part()

    register_chicken_feather()
