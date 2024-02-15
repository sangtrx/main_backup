from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='chick_part.yaml', 
            epochs=50, 
            imgsz=256, 
            batch=16, 
            name='yolov8n_no_aug',
            hsv_h = 0,
            hsv_s = 0,
            hsv_v = 0,
            translate = 0,
            scale = 0,
            fliplr = 0,
            mosaic = 0
            )
