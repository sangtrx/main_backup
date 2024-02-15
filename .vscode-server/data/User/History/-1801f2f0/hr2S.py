from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='hand.yaml', 
            epochs=300, 
            imgsz=256, 
            batch=1, 
            name='yolov8n_hand_full_aug',
            # hsv_h = 0,
            # hsv_s = 0,
            # hsv_v = 0,
            # translate = 0,
            # scale = 0,
            # fliplr = 0,
            # mosaic = 0
            )
