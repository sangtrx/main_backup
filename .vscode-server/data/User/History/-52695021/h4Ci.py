from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='feather.yaml', 
            epochs=1000, 
            imgsz= (256,256), 
            patience = 100,
            batch=64, 
            name='yolov8x_brio_feather_seg',
            copy_paste = 0.5          
            )
