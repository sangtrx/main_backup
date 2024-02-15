from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='/home/tqsang/JSON2YOLO/carcass_YOLO_v1/datasets/dataset.yaml', 
            epochs=200, 
            imgsz=256, 
            batch=16, 
            name='yolov8x_test',
            # hsv_h = 0,
            # hsv_s = 0,
            # hsv_v = 0,
            # translate = 0,
            # scale = 0,
            # fliplr = 0,
            # mosaic = 0
            )
