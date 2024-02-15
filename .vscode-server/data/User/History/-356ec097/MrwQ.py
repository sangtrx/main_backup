from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='carcass.yaml', 
            epochs=1000, 
            imgsz=256, 
            batch=64, 
            name='yolov8x_brio_det_v2',
            )
