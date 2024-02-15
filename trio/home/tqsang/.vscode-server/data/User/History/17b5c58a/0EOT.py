# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import some common libraries
import numpy as np
import cv2
import torch
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("front2class_2017_val_panoptic")

# import Mask2Former project
from mask2former import add_maskformer2_config

im = cv2.imread("/home/tqsang/V100/tqsang/crop_obj/front_2_class_COCO_synthetic_bg/val2017/4_Trim_1_43_1__4_Trim_2_1222_6__5_Trim_1240_6.jpg")



cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("/home/tqsang/Mask2Former/output_coco_2880to256to256_panoptic_/config.yaml")
cfg.MODEL.WEIGHTS = '/home/tqsang/Mask2Former/output_coco_2880to256to256_panoptic_/model_final.pth'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Show panoptic/instance/semantic predictions: 
v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]
cv2.imwrite(str('/home/tqsang/Mask2Former/output_coco_2880to256to256_panoptic_/out.png'), result)