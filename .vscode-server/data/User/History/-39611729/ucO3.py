import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

def process_video(model, input_video, output_video, areas):
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        for idx, area in enumerate(areas):
            x1, y1, x2, y2 = area
            cropped_frame = frame[y1:y2, x1:x2]

            # Convert the cropped frame to PIL image
            cropped_frame_pil = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

            # Run YOLOv8 inference on the cropped frame
            results = model.predict(cropped_frame_pil, conf=0.9)

            # Check if there are detections
            if len(results[0].boxes) > 0:
                # Draw the detections on the cropped frame
                annotated_img = results[0].plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                # Paste the annotated cropped frame back into the original frame
                frame[y1:y2, x1:x2] = annotated_img_rgb

                # Set the color of the bounding box (green for left, red for right)
                color = (0, 255, 0) if idx == 0 else (0, 0, 255)

                # Draw a bounding box around the area in the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Write the frame with detections to the output video
        out.write(frame)

    cap.release()
    out.release()

# Load the YOLOv8 model
model = YOLO('/mnt/tqsang/data_chick_part_YOLO/runs/detect/yolov8x_default_rot/weights/best.pt')

areas_part1 = [(240, 100, 1008, 868), (878, 139, 1646, 907)]
areas_part2 = [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]

input_video_part1 = "/mnt/tqsang/part1-test_chick_x.mp4"
input_video_part2 = "/mnt/tqsang/part2-test_chick_x.mp4"

output_video_part1 = "/mnt/tqsang/part1-test_output.mp4"
output_video_part2 = "/mnt/tqsang/part2-test_output.mp4"

process_video(model, input_video_part1, output_video_part1, areas_part1)
process_video(model, input_video_part2, output_video_part2, areas_part2)
