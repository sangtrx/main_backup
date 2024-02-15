import cv2
import numpy as np
from ultralytics import YOLO

def process_video(model, video_path, output_path, area):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    chick_count = 0
    defect_count = 0
    current_chick = False

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        output_frame = frame.copy()

        # Define the bounding rectangle area
        x, y, x2, y2 = area

        # Crop the frame to area
        cropped_frame = frame[y:y2, x:x2]

        # Perform inference on the area
        results = model.predict(cropped_frame, conf=0.8)

        # Iterate through the detections
        for i in range(len(results[0].boxes)):
            box = results[0].boxes[i]
            score = results[0].probs[i]
            class_ = results[0].names[i]

            box_x1, box_y1, box_x2, box_y2 = box.xyxy

            
            # Check if the box overlaps with the main area
            if (box_x1 >= x and box_x2 <= x2) and (box_y1 >= y and box_y2 <= y2):
                continue

            # If the box does not overlap, increase the count
            if not current_chick:
                chick_count += 1
                current_chick = True

            # Count defects
            if class_ == 'defect':
                defect_count += 1

            # Draw boxes for the non-overlapping detections
            cv2.rectangle(output_frame, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)
            cv2.putText(output_frame, str(class_), (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Reset current_chick flag
        current_chick = False

        # Update count display
        count_text = "Normal: {} Defects: {}".format(chick_count - defect_count, defect_count)
        cv2.putText(output_frame, count_text, (width//2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        out.write(output_frame)

    cap.release()
    out.release()


model_path = '/home/tqsang/JSON2YOLO/carcass_YOLO_det/runs/detect/yolov8x_det/weights/best.pt'
model = YOLO(model_path)

video_path = '/mnt/tqsang/dot1_1.mp4'
output_path = '/mnt/tqsang/dot1_1_output_det_count.mp4'

area = [602, 235, 600+602, 600+235] # [x, y, w, h]

process_video(model, video_path, output_path, area)
