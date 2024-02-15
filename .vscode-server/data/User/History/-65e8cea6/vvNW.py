import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def process_video(model, video_path, output_path, areas):
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

        for idx, area in enumerate(areas):
            x1, y1, x2, y2 = area
            cropped_frame = frame[y1:y2, x1:x2]

            results = model.predict(cropped_frame, conf=0.8)

            if len(results[0].boxes) > 0:
                annotated_frame = results[0].plot()
                output_frame[y1:y2, x1:x2] = annotated_frame

            color = (0, 255, 0) if idx == 0 else (0, 0, 255)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        out.write(output_frame)

    cap.release()
    out.release()

model_path = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/runs/detect/yolov8x_brio_det_only/weights/best.pt'
model = YOLO(model_path)

video_path_part1 = '/mnt/tqsang/dot1_vid/dot1_20_to.mp4'

output_path_part1 = '/mnt/tqsang/test_vid_det_only.mp4'

areas_part1 = [(711, 391, 711+420, 391+420)]

process_video(model, video_path_part1, output_path_part1, areas_part1)
