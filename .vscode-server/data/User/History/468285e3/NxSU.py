from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='skin.yaml', 
            epochs=1000, 
            imgsz= (512,512), 
            patience = 100,
            batch=32, 
            name='yolov8x_skin_dot3',
            mosaic = 0,   # turn this off 
            scale = 0  # turn this off               
            )
