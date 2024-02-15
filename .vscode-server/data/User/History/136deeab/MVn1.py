import cv2
import numpy as np
import glob
from ultralytics import YOLO
import os 

def process_video(model, video_path, output_path, area):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1

        if not success:
            break

        # Define the bounding rectangle area
        x, y, x2, y2 = area

        # Crop the frame to area
        cropped_frame = frame[y:y2, x:x2]

        # Perform inference on the area
        results = model.predict(cropped_frame, conf=0.8)

        # Iterate through the detections
        for i, (box, score, class_) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
            box_x1, box_y1, box_x2, box_y2 = box.tolist()

            # Check if the box is completely inside the area
            if (box_x1 >= x+5 and box_x2 <= x2-5) and (box_y1 >= y+5 and box_y2 <= y2-5):
                print(ok)
                # Save the original crop area img, the crop area img with the visualization of that one box, .txt file to save the label in YOLO format
                cv2.imwrite(f'/mnt/tqsang/dot1_data/original/{video_path.split("/")[-1]}_{frame_count}.jpg', cropped_frame)
                cv2.rectangle(cropped_frame, (int(box_x1-x), int(box_y1-y)), (int(box_x2-x), int(box_y2-y)), (255, 0, 0), 2)
                cv2.imwrite(f'/mnt/tqsang/dot1_data/visualize/{video_path.split("/")[-1]}_{frame_count}.jpg', cropped_frame)
                with open(f'/mnt/tqsang/dot1_data/label/{video_path.split("/")[-1]}_{frame_count}.txt', 'w') as f:
                    f.write(f'{class_} {score} {box_x1} {box_y1} {box_x2} {box_y2}\n')

    cap.release()

model_path = '/home/tqsang/JSON2YOLO/carcass_YOLO_det/runs/detect/yolov8x_det/weights/best.pt'
model = YOLO(model_path)

# Define the areas for each group
areas = {
    '1': [602, 235, 600+602, 600+235],
    '2-12': [602, 235, 600+602, 600+235], # replace with actual area
    '13-29': [602, 235, 600+602, 600+235], # replace with actual area
    '30-40': [602, 235, 600+602, 600+235] # replace with actual area
}

# Get all video files
video_files = glob.glob('/mnt/tqsang/dot1_vid/dot1_*.mp4')

# Group video files
video_groups = {
    '1': [],
    '2-12': [],
    '13-29': [],
    '30-40': []
}

# Create output directories if they don't exist
output_dirs = ['/mnt/tqsang/dot1_data','/mnt/tqsang/dot1_data/original', '/mnt/tqsang/dot1_data/visualize', '/mnt/tqsang/dot1_data/label']
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

for video_file in video_files:
    
    if video_file != '/mnt/tqsang/dot1_vid/dot1_1.mp4':
        continue
    # print(video_file.split('_')[2].split('.')[0])
    video_number = int(video_file.split('_')[2].split('.')[0])
    if video_number == 1:
        video_groups['1'].append(video_file)
    elif 2 <= video_number <= 12:
        video_groups['2-12'].append(video_file)
    elif 13 <= video_number <= 29:
        video_groups['13-29'].append(video_file)
    elif 30 <= video_number <= 40:
        video_groups['30-40'].append(video_file)

# Process each video
for group, videos in video_groups.items():
    for video in videos:
        print(video)
        process_video(model, video, f'/mnt/tqsang/dot1_data/{video.split("/")[-1]}', areas[group])
