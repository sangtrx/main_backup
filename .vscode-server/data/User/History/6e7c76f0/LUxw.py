from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='/home/tqsang/JSON2YOLO/carcass_YOLO_v1/datasets/dataset.yaml', 
            epochs=500, 
            imgsz=512, 
            batch=16, 
            name='yolov8x_v1',
            )
