from ultralytics import YOLO

# Load a model
model = YOLO('/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det/runs/detect/yolov8x_brio_det_v3/weights/best.pt')  # load a pretrained model (recommended for training)

# Validate the model
metrics = model.val(data='/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det/carcass.yaml',split='test)  # no arguments needed, dataset and settings remembered
metrics.confusion_matrix
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category