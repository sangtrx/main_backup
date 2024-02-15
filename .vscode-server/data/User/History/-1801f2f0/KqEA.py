from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='hand.yaml', 
            epochs=500, 
            imgsz=1024, 
            batch=32, 
            name='yolov8x_hand_full_aug_v3',
            flipud = 0.1, #v2 
            hsv_h = 0.5, #v3
            # hsv_s = 0,
            # hsv_v = 0,
            # translate = 0,
            # scale = 0,
            # fliplr = 0,
            # mosaic = 0
            )
