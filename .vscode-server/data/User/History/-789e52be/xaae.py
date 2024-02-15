import cv2
import glob
import os
import numpy as np

from ultralytics import YOLO

# Function to calculate IOU (Intersection over Union)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the coordinates of the intersection rectangle
    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x6 - x5 + 1) * max(0, y6 - y5 + 1)

    # Calculate the area of both bounding boxes
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

    # Calculate IOU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Function to perform inference and draw bounding boxes
def perform_inference_and_draw_boxes(model, img, output_frame, color, part_name, ii, conf_threshold=0.3, show_score=False):
    results = model.predict(img, conf=conf_threshold)
    part_count = 0
    
    # Draw bounding boxes for the detected objects
    for i, (box, score, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        if score >= conf_threshold:
            # cv2.rectangle(output_frame, 
            #               (int(box_x1), int(box_y1)), 
            #               (int(box_x2), int(box_y2)), 
            #               color, 2)
            if show_score:
                text = f'{score:.2f}' # {part_name}: 
                text_position = (int(box_x1), int(box_y1) - 10)
                cv2.putText(output_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            part_count += 1

    # Put text next to the image with the part counts using the same color
    text_position = (10, 30 + 40 * ii)  # Update text position for each part
    cv2.putText(output_frame, f'{part_name}: {part_count}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return results

# Define the paths to the YOLO models and confidence thresholds for each part
part_config = {
    'feather': {'model_path': '/home/tqsang/JSON2YOLO/feather_YOLO_det/runs/detect/yolov8x_dot3_noRot/weights/best.pt', 'conf_threshold': 0.2},
    'feather on skin': {'model_path': '/home/tqsang/JSON2YOLO/feather_on_skin_YOLO_det/runs/detect/yolov8x_feather_on_skin/weights/best.pt', 'conf_threshold': 0.2},
    'wing': {'model_path': '/home/tqsang/JSON2YOLO/wing_YOLO_det/runs/detect/yolov8x_wing_dot3_nohardcase_noMosaic_noScale/weights/best.pt', 'conf_threshold': 0.2},
    'skin': {'model_path': '/home/tqsang/JSON2YOLO/skin_YOLO_det/runs/detect/yolov8x_skin/weights/best.pt', 'conf_threshold': 0.5},
    'flesh': {'model_path': '/home/tqsang/JSON2YOLO/flesh_YOLO_det/runs/detect/yolov8x_flesh_dot3/weights/best.pt', 'conf_threshold': 0.5}
}

# Load the YOLO models
models = {part: YOLO(part_config[part]['model_path']) for part in part_config}

# Define a fixed color list
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (50, 100, 255), (0, 255, 255)]

# Create the output folder if it doesn't exist
output_directory = '/mnt/tqsang/dot2_test/test_output_dot3'
os.makedirs(output_directory, exist_ok=True)

# Toggle for showing scores
show_scores = False  # Change this to False if you want to hide scores

# Loop through images in the specified directory
image_directory = '/mnt/tqsang/dot2_test/test_imgs'

for image_path in glob.glob(os.path.join(image_directory, '*.png')):
    img = cv2.imread(image_path)
    output_frame = img.copy()

    # Perform inference for 'feather' and 'feather on skin' separately
    feather_results = perform_inference_and_draw_boxes(models['feather'], img, output_frame, colors[0], 'feather', 0, part_config['feather']['conf_threshold'], show_scores)
    feather_on_skin_results = perform_inference_and_draw_boxes(models['feather on skin'], img, output_frame, colors[1], 'feather on skin', 1, part_config['feather on skin']['conf_threshold'], show_scores)
    
    # Combine the results of 'feather' and 'feather on skin'
    combined_results = feather_results + feather_on_skin_results

    # Filter results with IOU > 90 and keep the one with a higher score
    final_results = []
    for i in range(len(combined_results)):
        add = True
        for j in range(len(combined_results)):
            if i != j:
                if combined_results[i].boxes and combined_results[j].boxes:  # Check if boxes are not empty
                    iou = calculate_iou(combined_results[i].boxes.xyxy[0], combined_results[j].boxes.xyxy[0])
                    if iou > 0.9 and combined_results[i].boxes.conf < combined_results[j].boxes.conf:
                        add = False
                        break
        if add:
            final_results.append(combined_results[i])

    # Run through the final results and classify as 'feather' or 'feather on skin'
    for i, (box, score, cls) in enumerate(zip(final_results[0].boxes.xyxy, final_results[0].boxes.conf, final_results[0].boxes.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        if score >= part_config['feather']['conf_threshold']:
            # Calculate the number of blackish pixels in the bbox area
            bbox_area = img[int(box_y1):int(box_y2), int(box_x1):int(box_x2)]
            num_blackish_pixels = np.sum(np.all(bbox_area < [30, 30, 30], axis=-1))
            if num_blackish_pixels > 5:
                part_name = 'feather'
            else:
                part_name = 'feather on skin'
            
            color = colors[0] if part_name == 'feather' else colors[1]
            cv2.rectangle(output_frame, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), color, 2)

    # Save the output image to the output directory
    output_image_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(output_image_path, output_frame)

print("Processing complete.")
