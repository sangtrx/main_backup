from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='feather.yaml', 
            epochs=1000, 
            imgsz= (512,512), 
            patience = 100,
            batch=64, 
            name='yolov8x_brio_feather_det_512_NoRotate_NoGray',
            # hsv_h = 0.01, # v2
            # hsv_s = 0.05,
            # hsv_v = 0.05,
            # translate = 0,
            # scale = 0.05,
            # fliplr = 0,
            # mosaic = 0            
            )
