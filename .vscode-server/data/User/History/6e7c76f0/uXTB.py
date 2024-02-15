from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='carcass.yaml', 
            epochs=500, 
            imgsz=640, 
            batch=1, 
            name='yolov8x_seg',
            task = 'segment'
            )
