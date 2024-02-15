from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='chick_part.yaml', 
            epochs=50, 
            imgsz=256, 
            batch=1, 
            name=yolov8n_custom_rot,
            degrees=(0,90,180,270)
            )
