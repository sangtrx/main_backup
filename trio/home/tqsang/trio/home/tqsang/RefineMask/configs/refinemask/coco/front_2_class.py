# The new config inherits a base config to highlight the necessary modification
_base_ = './r50-refinemask-1x-faster-better.py'

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('normal','defect')
data = dict(
    train=dict(
        img_prefix='/home/tqsang/V100/tqsang/crop_obj/front_2_class/train/img/',
        classes=classes,
        ann_file='/home/tqsang/V100/tqsang/crop_obj/front_2_class/instances_train2017.json'),
    val=dict(
        img_prefix='/home/tqsang/V100/tqsang/crop_obj/front_2_class/val/img/',
        classes=classes,
        ann_file='/home/tqsang/V100/tqsang/crop_obj/front_2_class/instances_val2017.json'),
    test=dict(
        img_prefix='/home/tqsang/V100/tqsang/crop_obj/front_2_class/val/img/',
        classes=classes,
        ann_file='/home/tqsang/V100/tqsang/crop_obj/front_2_class/instances_val2017.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'