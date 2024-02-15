import os
import cv2
from ultralytics import YOLO
import numpy as np

def process_video(model_hand, model_chicken, video_path, areas, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create the output video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, 2 * height))

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference for hand detection on the full frame
            results_hand = model_hand(frame, conf=0.5)
            if len(results_hand[0].boxes) == 0:
                # Initialize the output frame
                output_frame = np.zeros((2 * height, width, 3), dtype=np.uint8)
                output_frame[:height, :, :] = frame
            else:
                # if detect hand visualize the hand
                output_frame = np.zeros((2 * height, width, 3), dtype=np.uint8)
                # Visualize the results on the cropped frame
                annotated_frame = results_hand[0].plot()
                annotated_cropped_frame_rgb = cv2.cvtColor(annotated_cropped_frame, cv2.COLOR_BGR2RGB)
                output_frame[:height, :, :] = annotated_cropped_frame_rgb            
            # Check if there are no hand detections
            if len(results_hand[0].boxes) == 0:
                for i, area in enumerate(areas):
                    x1, y1, x2, y2 = area
                    cropped_frame = frame[y1:y2, x1:x2]

                    # Run YOLOv8 inference for chicken part detection on the cropped frame
                    results_chicken = model_chicken(cropped_frame, conf=0.9)

                    # Check if there are detections
                    if len(results_chicken[0].boxes) > 0:
                        # Visualize the results on the cropped frame
                        annotated_cropped_frame = results_chicken[0].plot()
                        annotated_cropped_frame_rgb = cv2.cvtColor(annotated_cropped_frame, cv2.COLOR_BGR2RGB)
                        output_frame[y1:y2, x1:x2] = annotated_cropped_frame_rgb

            # Write the output frame to the output video file
            out.write(output_frame)

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, output video object, and close the display window
    cap.release()
    out.release()

# Load the YOLOv8 models
model_hand = YOLO("/mnt/tqsang/hand_dataset_YOLO/runs/detect/yolov8n_hand_full_aug/weights/best.pt")
model_chicken = YOLO("/mnt/tqsang/data_chick_part_YOLO/runs/detect/yolov8x_default_rot/weights/best.pt")

areas_part1 = [(240, 100, 1008, 868), (878, 139, 1646, 907)]
areas_part2 = [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]

video_path_part1 = "/mnt/tqsang/part1-test.mp4"
video_path_part2 = "/mnt/tqsang/part2-test.mp4"

output_video_path_part1 = "/mnt/tqsang/output_part1.mp4"
output_video_path_part2 = "/mnt/tqsang/output_part2.mp4"

process_video(model_hand, model_chicken, video_path_part1, areas_part1, output_video_path_part1)
process_video(model_hand, model_chicken, video_path_part2, areas_part2, output_video_path_part2)

