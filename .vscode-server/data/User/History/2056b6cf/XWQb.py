import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Load the models
model1 = YOLO('/mnt/tqsang/hand_dataset_YOLO/runs/detect/yolov8n_hand_full_aug/weights/best.pt')
model2 = YOLO('/mnt/tqsang/data_chick_part_YOLO/runs/detect/yolov8x_default_rot/weights/best.pt')

# Define the video paths
video_path1 = '/mnt/tqsang/part1-test.mp4'
video_path2 = '/mnt/tqsang/part2-test.mp4'

areas_part1 = [(240, 100, 1008, 868), (878, 139, 1646, 907)]
areas_part2 = [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]

output_video_path = 'output_video.mp4'

def process_video(video_path, areas, model1, model2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    frames = []

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # Run hand detection model (model1) on the full frame
        results1 = model1.predict(frame, conf=0.9)

        # If no hands detected, process the frame using model2
        if len(results1[0].boxes) == 0:
            frame_parts = []
            for area in areas:
                x1, y1, x2, y2 = area
                cropped_frame = frame[y1:y2, x1:x2]

                # Run chicken part detection model (model2) on the cropped frame
                results2 = model2.predict(cropped_frame, conf=0.9)

                # If there are detections, plot them on the cropped frame
                if len(results2[0].boxes) > 0:
                    cropped_frame = results2[0].plot()

                frame_parts.append(cropped_frame)

            # Concatenate the cropped frames
            frame_bottom = cv2.hconcat(frame_parts)

            # Create black area with the same width as the frame and height equal to the cropped frame
            black_area = np.zeros((frame_bottom.shape[0], frame.shape[1] - frame_bottom.shape[1], 3), dtype=np.uint8)

            # Concatenate the black area with the frame_bottom
            frame_bottom = cv2.hconcat([frame_bottom, black_area])

            # Concatenate the original frame and the frame_bottom
            frame = cv2.vconcat([frame, frame_bottom])

        frames.append(frame)

    cap.release()
    return frames

# Process both videos
frames_part1 = process_video(video_path1, areas_part1, model1, model2)
frames_part2 = process_video(video_path2, areas_part2, model1, model2)

# Combine the frames from both videos
frames = frames_part1 + frames_part2

# Save the output video
height, width, _ = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

for frame in frames:
    out.write(frame)

out.release()
