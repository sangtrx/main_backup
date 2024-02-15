from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='carcass.yaml', 
            epochs=1000, 
            imgsz=256, 
            batch=64, 
            name='yolov8x_brio_det_v2',
            flipud = 0.1, #v2 
            hsv_h = 0.2, # v2 0.2 #v3 0.5
            # hsv_s = 0,
            # hsv_v = 0,
            # translate = 0,
            # scale = 0,
            # fliplr = 0,
            # mosaic = 0            
            )
