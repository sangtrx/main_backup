from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg


def add_config(cfg):
    # PVT backbone
    cfg.MODEL.PVT = CN()
    cfg.MODEL.PVT.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]

    cfg.MODEL.PVT.PATCH_SIZE = 4 
    cfg.MODEL.PVT.EMBED_DIMS = [64, 128, 320, 512]
    cfg.MODEL.PVT.NUM_HEADS = [1, 2, 5, 8]
    cfg.MODEL.PVT.MLP_RATIOS = [8, 8, 4, 4]
    cfg.MODEL.PVT.DEPTHS = [3, 4, 6, 3]
    cfg.MODEL.PVT.SR_RATIOS = [8, 4, 2, 1]

    cfg.MODEL.PVT.WSS = [7, 7, 7, 7]

    # not an original pvt config
    cfg.MODEL.ALL_LAYERS_ROI_POOLING = False

    # qm_net config
    cfg.MODEL.QM_NET = CN()
    cfg.MODEL.QM_NET.USE = False
    cfg.MODEL.QM_NET.JUSTIFY_LOSS = False
    cfg.MODEL.QM_NET.POSE_ATN = False
    cfg.MODEL.QM_NET.REC_BY_SHAPE = 0.
    cfg.MODEL.QM_NET.REC_BY_FEATURE = 1.

    # addation
    cfg.SOLVER.OPTIMIZER = "AdamW"

if __name__== '__main__':
    cfg = get_cfg()
    add_config(cfg)
