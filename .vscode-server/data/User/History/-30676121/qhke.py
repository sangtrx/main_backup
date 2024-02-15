from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='chick_part.yaml', 
            epochs=50, 
            imgsz=256, 
            batch=1, 
            name='yolov8x_default_rot',
            )
