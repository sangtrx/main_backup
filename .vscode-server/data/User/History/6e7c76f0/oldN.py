from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='carcass.yaml', 
            epochs=500, 
            imgsz=256, 
            batch=16, 
            name='yolov8x_v1',
            )
