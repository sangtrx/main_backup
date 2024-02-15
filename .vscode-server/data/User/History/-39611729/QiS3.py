import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def process_video(model, video_path, areas, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video dimensions and set up the output video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    colors = [(0, 255, 0), (0, 0, 255)]  # green, red

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            for idx, (x1, y1, x2, y2) in enumerate(areas):
                # Crop the area
                cropped_img = frame[y1:y2, x1:x2]

                # Run the model on the cropped area
                results = model.predict(cropped_img, conf=0.9)

                # Draw the bounding boxes on the original frame
                for box in results[0].boxes.xywh:
                    x, y, w, h = box
                    x_center = int(x1 + x)
                    y_center = int(y1 + y)
                    x1_box = int(x_center - w / 2)
                    y1_box = int(y_center - h / 2)
                    x2_box = int(x_center + w / 2)
                    y2_box = int(y_center + h / 2)
                    cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), colors[idx], 2)

            # Write the frame with boxes to the output video
            out.write(frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, output video writer and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Load the YOLOv8 model
model = YOLO('/mnt/tqsang/data_chick_part_YOLO/runs/detect/yolov8n_default_rot/weights/best.pt')

# Define the areas for part 1 and part 2
areas_part1 = [(240, 100, 1008, 868), (878, 139, 1646, 907)]
areas_part2 = [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]

# Define the input and output video paths
video_path_part1 = "/mnt/tqsang/part1-test.mp4"
output_path_part1 = "/mnt/tqsang/part1-test-output_n.mp4"

video_path_part2 = "/mnt/tqsang/part2-test.mp4"
output_path_part2 = "/mnt/tqsang/part2-test-output_n.mp4"

# Process the videos
process_video(model, video_path_part1, areas_part1, output_path_part1)
process_video(model, video_path_part2, areas_part2, output_path_part2)
