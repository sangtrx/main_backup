import cv2
import numpy as np
from ultralytics import YOLO

def process_video(model, video_path, output_path, area):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        output_frame = frame.copy()

        # Define the bounding rectangle area
        x, y, w, h = area
        x2, y2 = x + w, y + h

        # Crop the frame to area
        cropped_frame = frame[y:y2, x:x2]

        # Perform inference on the area
        results = model.predict(cropped_frame, conf=0.8)

        if len(results[0].boxes) > 0:
            # If there are detections, draw them on the frame
            annotated_frame = results[0].plot()
            output_frame[y:y2, x:x2] = annotated_frame

        # Draw the bounding rectangle
        cv2.rectangle(output_frame, (x, y), (x2, y2), (255, 0, 0), 2)

        out.write(output_frame)

    cap.release()
    out.release()


model_path = '/home/tqsang/JSON2YOLO/carcass_YOLO_v1/runs/segment/yolov8x_seg/weights/best.pt'
model = YOLO(model_path)

video_path = '/mnt/tqsang/dot1_1.mp4'
output_path = '/mnt/tqsang/dot1_1_output.mp4'

area = [602, 235, 300+602, 300+235] # [x, y, w, h]

process_video(model, video_path, output_path, area)
