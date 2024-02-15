import cv2
import numpy as np
import glob
from ultralytics import YOLO
import os 
from collections import Counter






# Global counter for normal and defect chickens
defect_count = 0
normal_count = 0

# Dictionary to keep track of each chicken
chicken_dict = {}

# Function to perform inference and draw bounding boxes
def perform_inference_and_draw_boxes(model, img, output_frame, color, part_name, ii, conf_threshold=0.3, show_score=False):
    results = model.predict(img, conf=conf_threshold)
    part_count = 0
    
    # Draw bounding boxes for the detected objects
    for i, (box, score, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        if score >= conf_threshold:
            cv2.rectangle(output_frame, 
                          (int(box_x1), int(box_y1)), 
                          (int(box_x2), int(box_y2)), 
                          color, 2)
            if show_score:
                text = f'{score:.2f}' # {part_name}: 
                text_position = (int(box_x1), int(box_y1) - 10)
                cv2.putText(output_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            part_count += 1

    # Put text next to the image with the part counts using the same color
    text_position = (10, 30 + 40 * ii)  # Update text position for each part
    cv2.putText(output_frame, f'{part_name}: {part_count}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return part_count

######################
def process_video(model, video_path, output_path, area):
    global defect_count, normal_count, chicken_dict
    # Define a counter for unique chicken identifiers
    chicken_id_counter = 0
    hit_left = False
    hit_right = False
    chicken_dict = {}
    defect_count = 0
    normal_count = 0
    featheravg_count = 0

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1

        if not success:
            break

        output_frame = frame.copy()

                # Draw counters on the top of area box



        # Define the bounding rectangle area
        x, y, x2, y2 = area

        # Crop the frame to area
        cropped_frame = frame[y:y2, x:x2]

        # Perform inference on the area
        results = model.predict(cropped_frame, conf=0.8)


        cv2.putText(output_frame, f"Avg flesh count: {round(featheravg_count, 2)}", (x, y-100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(output_frame, f"Defect count: {defect_count}", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(output_frame, f"Normal count: {normal_count}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        # Iterate through the detections
        for i, (box, score, class_) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
            box_x1, box_y1, box_x2, box_y2 = box.tolist()
            # Check if the box is completely inside the area
            if (box_x1 >= 5 and box_x2 <= x2-x-5): # Only need to check 2left n right vertical line 

                ## feather detection
                # Calculate the smallest square area that contains the bounding box
                min_side_length = max(box_x2 - box_x1, box_y2 - box_y1)
                center_x = (box_x1 + box_x2) / 2
                center_y = (box_y1 + box_y2) / 2
                square_x1 = int(center_x - min_side_length / 2)
                square_y1 = int(center_y - min_side_length / 2)
                square_x2 = int(center_x + min_side_length / 2)
                square_y2 = int(center_y + min_side_length / 2)

                # Adjust coordinates to stay within the valid range of the cropped_frame
                if square_x1 < 0:
                    square_x2 -= square_x1
                    square_x1 = 0
                if square_y1 < 0:
                    square_y2 -= square_y1
                    square_y1 = 0
                if square_x2 > cropped_frame.shape[1]:
                    square_x1 -= (square_x2 - cropped_frame.shape[1])
                    square_x2 = cropped_frame.shape[1]
                if square_y2 > cropped_frame.shape[0]:
                    square_y1 -= (square_y2 - cropped_frame.shape[0])
                    square_y2 = cropped_frame.shape[0]


                # Crop the smallest square area from the frame
                cropped_square = cropped_frame[square_y1:square_y2, square_x1:square_x2]
                img = cropped_square.copy()
                draw_frame = cropped_square.copy()
                ## inf for each parts here 
                part_counts = {}

                # Perform inference and draw boxes for each part using the fixed colors and respective confidence thresholds
                for i, part in enumerate(part_config):
                    color = colors[i % len(colors)]  # Cycle through the fixed color list
                    conf_threshold = part_config[part]['conf_threshold']
                    part_count = perform_inference_and_draw_boxes(models[part], img, draw_frame, color, part, i, conf_threshold, show_scores)
                    part_counts[part] = part_count

                # paste draw_frame back onto output_frame  
                output_frame[square_y1:square_y2, square_x1:square_x2] = draw_frame   

                
                # # Perform inference on the smallest square area for feather detection
                # results_feather = model_feather.predict(cropped_square, conf=0.8)

                # # Count all the feathers that in the box_x1, box_y1, box_x2, box_y2 area
                # feather_count = sum(1 for i, (feather_box, feather_score, feather_class) in 
                #                         enumerate(zip(results_feather[0].boxes.xyxy, 
                #                                     results_feather[0].boxes.conf, 
                #                                     results_feather[0].boxes.cls)) 
                #                         if feather_box[0] + square_x1  >= box_x1 and feather_box[1] + square_y1 >= box_y1 and 
                #                         feather_box[2] + square_x1 <= box_x2 and feather_box[3] + square_y1 <= box_y2)
                
                # # Draw bounding box for each feather in results_feather inside the area
                # for i, (feather_box, feather_score, feather_class) in enumerate(zip(results_feather[0].boxes.xyxy, 
                #                                                                     results_feather[0].boxes.conf, 
                #                                                                     results_feather[0].boxes.cls)):
                #     if feather_box[0] >= box_x1 and feather_box[1] >= box_y1 and feather_box[2] <= box_x2 and feather_box[3] <= box_y2:
                #         # Draw the bounding box for each feather
                #         feather_box_x1, feather_box_y1, feather_box_x2, feather_box_y2 = feather_box.tolist()
                #         cv2.rectangle(output_frame, 
                #                     (int(feather_box_x1+x + square_x1), int(feather_box_y1+y + square_y1 )), 
                #                     (int(feather_box_x2+x + square_x1), int(feather_box_y2+y + square_y1)), 
                #                     (0, 255, 100), 2)  



                
                # Get class prediction
                if feather_count >= 1:
                    predicted_class = 'defect'
                else:
                    predicted_class = 'normal'
                # predicted_class = class_names[preds]

                # Check if chicken is crossing the area of left line
                if    20 >= box_x1 >= 5:
                    if hit_left == False:
                        hit_left = True  
                        hit_right = False                  
                        # Create a unique chicken id
                        chicken_id = f"{chicken_id_counter}"
                        chicken_id_counter += 1

                        chicken_dict[chicken_id] = {"classifications": [predicted_class], "feather_count":[feather_count]}

                # If chicken already crossed the left line, continue to update its classifications
                if  hit_left == True and hit_right == False:
                        chicken_dict[chicken_id]["classifications"].append(predicted_class)
                        chicken_dict[chicken_id]["feather_count"].append(feather_count)

                # If chicken crossed the area of right line
                if    x2-x-20<= box_x2 <= x2-x-5:
                    if hit_right == False and hit_left == True:
                        hit_right = True
                        hit_left  = False
                        # Compute average classification result
                        avg_result = Counter(chicken_dict[chicken_id]["classifications"]).most_common(1)[0][0]

                        # Calculate average feather count
                        feather_sum = sum(chicken_dict[chicken_id]["feather_count"])
                        feather_total = len(chicken_dict[chicken_id]["feather_count"])
                        featheravg_count = feather_sum / feather_total if feather_total != 0 else 0

                        # Update the counters and remove the chicken from the dictionary
                        if avg_result == 'defect':
                            defect_count += 1
                        else:
                            normal_count += 1

                        del chicken_dict[chicken_id]


                color = (255, 0, 0) if predicted_class == 'defect' else (0, 0, 255)
                cv2.rectangle(output_frame, (int(box_x1+x), int(box_y1+y)), (int(box_x2+x), int(box_y2+y)), color, 2)
                # Display text on top left of the box
                cv2.putText(output_frame, predicted_class, (int(box_x1+x), int(box_y1+y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)




        # Draw the bounding rectangle area
        color = (0, 255, 0) 
        cv2.rectangle(output_frame, (x, y), (x2, y2), color, 2)

        # write frame
        out.write(output_frame)
    cap.release()
    out.release()

# det model
model_path = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/runs/detect/yolov8x_brio_det_only/weights/best.pt'
model = YOLO(model_path)

# Define the paths to the YOLO models and confidence thresholds for each part
part_config = {
    'feather': {'model_path': '/home/tqsang/JSON2YOLO/feather_YOLO_det/runs/detect/yolov8x_brio_feather_det_512_NoRotate_dot2_2/weights/best.pt', 'conf_threshold': 0.4},
    'feather on skin': {'model_path': '/home/tqsang/JSON2YOLO/feather_on_skin_YOLO_det/runs/detect/yolov8x_feather_on_skin/weights/best.pt', 'conf_threshold': 0.2},
    'wing': {'model_path': '/home/tqsang/JSON2YOLO/wing_YOLO_det/runs/detect/yolov8x_wing2/weights/best.pt', 'conf_threshold': 0.2},
    'skin': {'model_path': '/home/tqsang/JSON2YOLO/skin_YOLO_det/runs/detect/yolov8x_skin/weights/best.pt', 'conf_threshold': 0.3},
    'flesh': {'model_path': '/home/tqsang/JSON2YOLO/flesh_YOLO_det/runs/detect/yolov8x_flesh_dot3/weights/best.pt', 'conf_threshold': 0.1}
}

# Load the YOLO models
models = {part: YOLO(part_config[part]['model_path']) for part in part_config}

# Define a fixed color list
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (50, 100, 255), (0, 255, 255)]

# Create the output folder if it doesn't exist
output_directory = '/mnt/tqsang/dot2_test/test_output_dot3'
os.makedirs(output_directory, exist_ok=True)

# Toggle for showing scores
show_scores = True  # Change this to False if you want to hide scores

# Define the areas for each group
areas = {
    '1-13': [1319, 598, 1319+1000, 598+1000],
}

# Get all video files
video_files = glob.glob('/mnt/tqsang/dot2_test/dot2_*.mp4')

# Group video files
video_groups = {
    '1-13': [],
}

for video_file in video_files:
    video_number = int(video_file.split('_')[2].split('.')[0])
    if 1 <= video_number <= 13:
        video_groups['1-13'].append(video_file)

# Create output directories if they don't exist
output_dirs = ['/mnt/tqsang/test_vid_flesh']
for dir in output_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
# Process each video
for group, videos in video_groups.items():
    for video in videos:
            print(video)
            process_video(model, video, f'/mnt/tqsang/test_vid_flesh/{video.split("/")[-1]}', areas[group])
