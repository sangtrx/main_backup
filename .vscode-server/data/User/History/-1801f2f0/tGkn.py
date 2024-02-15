from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='hand.yaml', 
            epochs=400, 
            imgsz=256, 
            batch=1, 
            name='yolov8x_hand_full_aug_v2',
            # hsv_h = 0,
            # hsv_s = 0,
            # hsv_v = 0,
            # translate = 0,
            # scale = 0,
            # fliplr = 0,
            # mosaic = 0
            )
