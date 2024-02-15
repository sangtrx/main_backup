import cv2
import glob
import os

from ultralytics import YOLO

# Function to calculate IOU (Intersection over Union) between two bounding boxes
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / (area_box1 + area_box2 - intersection_area)
    return iou

# Function to process 'feather' and 'feather on skin' together
def process_feather_combined(model, img, output_frame, color, part_name, ii, conf_threshold=0.3, show_score=False):
    results = model.predict(img, conf=conf_threshold)
    part_count = 0
    boxes = []

    # Draw bounding boxes for the detected objects
    for i, (box, score, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        if score >= conf_threshold:
            cv2.rectangle(output_frame, 
                          (int(box_x1), int(box_y1)), 
                          (int(box_x2), int(box_y2)), 
                          color, 2)
            if show_score:
                text = f'{score:.2f}'  # {part_name}:
                text_position = (int(box_x1), int(box_y1) - 10)
                cv2.putText(output_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            part_count += 1
            boxes.append((box_x1, box_y1, box_x2, box_y2, score))

    # Calculate IOU and choose the one with the higher score if IOU > 0.90
    chosen_boxes = []
    for i in range(len(boxes)):
        iou_threshold = 0.90
        iou_over_threshold = [False] * len(boxes)
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i][:4], boxes[j][:4]) > iou_threshold:
                iou_over_threshold[i] = True
                iou_over_threshold[j] = True
        if not any(iou_over_threshold):
            chosen_boxes.append(boxes[i])

    # Check if there are more than 15 blackish pixels in the bbox area to determine 'feather' or 'feather on skin'
    for box in chosen_boxes:
        box_x1, box_y1, box_x2, box_y2, score = box
        bbox = img[int(box_y1):int(box_y2), int(box_x1):int(box_x2)]
        grayscale = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
        black_pixels = cv2.inRange(grayscale, 0, 50)
        black_pixel_count = cv2.countNonZero(black_pixels)
        part_name = 'feather' if black_pixel_count > 15 else 'feather on skin'

        # Draw bounding box with the determined part name
        cv2.rectangle(output_frame, 
                      (int(box_x1), int(box_y1)), 
                      (int(box_x2), int(box_y2)), 
                      color, 2)
        text = f'{part_name}: {score:.2f}'
        text_position = (int(box_x1), int(box_y1) - 10)
        cv2.putText(output_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Put text next to the image with the part counts using the same color
    text_position = (10, 30 + 40 * ii)  # Update text position for each part
    cv2.putText(output_frame, f'{part_name}: {part_count}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return part_count

# Define the paths to the YOLO models
part_config = {
    'feather_combined': {'model_path': '/path/to/combined_model_weights.pt', 'conf_threshold': 0.2},
    'wing': {'model_path': '/home/tqsang/JSON2YOLO/wing_YOLO_det/runs/detect/yolov8x_wing_dot3_nohardcase_noMosaic_noScale/weights/best.pt', 'conf_threshold': 0.2},
    'skin': {'model_path': '/home/tqsang/JSON2YOLO/skin_YOLO_det/runs/detect/yolov8x_skin/weights/best.pt', 'conf_threshold': 0.5},
    'flesh': {'model_path': '/home/tqsang/JSON2YOLO/flesh_YOLO_det/runs/detect/yolov8x_flesh_dot3/weights/best.pt', 'conf_threshold': 0.5}
}

# Load the YOLO models
models = {part: YOLO(part_config[part]['model_path']) for part in part_config}

# Create the output folder if it doesn't exist
output_directory = '/mnt/tqsang/dot2_test/test_output_dot3'
os.makedirs(output_directory, exist_ok=True)

# Toggle for showing scores
show_scores = True  # Change this to False if you want to hide scores

# Loop through images in the specified directory
image_directory = '/mnt/tqsang/dot2_test/test_imgs'

for image_path in glob.glob(os.path.join(image_directory, '*.png')):
    img = cv2.imread(image_path)
    output_frame = img.copy()

    part_counts = {}

    # Perform inference and draw boxes for the combined 'feather' and 'feather on skin' part
    part_name = 'feather_combined'
    color = (0, 0, 255)  # Choose a color for the combined part
    conf_threshold = part_config[part_name]['conf_threshold']
    part_count = process_feather_combined(models[part_name], img, output_frame, color, part_name, 0, conf_threshold, show_scores)
    part_counts[part_name] = part_count

    # Save the output image to the output directory
    output_image_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(output_image_path, output_frame)

print("Processing complete.")
