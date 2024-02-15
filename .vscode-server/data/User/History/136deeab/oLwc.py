import cv2
import os
import glob
import numpy as np
from PIL import Image
from pathlib import Path

# Define the model path
model_path = 'weights/best.pt'
model = YOLO(model_path)

# Define the video paths
video_paths = glob.glob('/mnt/tqsang/dot1_vid/*.mp4')

# Define the output path
output_path = '/mnt/tqsang/dot1_data'

# Define the groups and their corresponding areas
groups = {
    '1': [602, 235, 600+602, 600+235],
    '2-12': [602, 235, 600+602, 600+235],
    '13-29': [602, 235, 600+602, 600+235],
    '30-40': [602, 235, 600+602, 600+235]
}

# Iterate over each video
for video_path in video_paths:
    video_name = os.path.basename(video_path).split('.')[0]
    group = next((g for g in groups if video_name in g), None)
    if group is None:
        continue

    # Define the areas for the current group
    areas = groups[group]

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Iterate over each frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Iterate over each area
        for area in areas:
            x, y, x2, y2 = area

            # Crop the area
            cropped_frame = frame[y:y2, x:x2]

            # Perform inference on the area
            results = model.predict(cropped_frame, conf=0.8)

            # Iterate through the detections
            for i, (box, score, class_) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
                box_x1, box_y1, box_x2, box_y2 = box.tolist()

                # Check if the box is completely inside the area
                if box_x1 >= x+5 and box_y1 >= y+5 and box_x2 <= x2-5 and box_y2 <= y2-5:
                    # Save the original crop area image
                    img_path = os.path.join(output_path, 'original', f'{video_name}_{frame_count}.jpg')
                    Image.fromarray(cropped_frame).save(img_path)

                    # Save the crop area image with the visualization of the box
                    vis_path = os.path.join(output_path, 'visualize', f'{video_name}_{frame_count}.jpg')
                    cv2.rectangle(cropped_frame, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), (255, 0, 0), 2)
                    Image.fromarray(cropped_frame).save(vis_path)

                    # Save the label in YOLO format
                    label_path = os.path.join(output_path, 'label', f'{video_name}_{frame_count}.txt')
                    with open(label_path, 'w') as f:
                        f.write(f'{class_} {score} {box_x1} {box_y1} {box_x2} {box_y2}\n')

        frame_count += 1

    cap.release()
