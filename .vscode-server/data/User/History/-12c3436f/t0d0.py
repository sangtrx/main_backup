import os
import glob
import json
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

# Define paths and maximum frame numbers
part1_frames = "/mnt/tqsang/chicken_part1/frames"
part1_xml = "/mnt/tqsang/chicken_part1/xml_44606"
part1_max_frame = 800 #44606

part2_frames = "/mnt/tqsang/chicken_part2/frames"
part2_xml = "/mnt/tqsang/chicken_part2/xml1_4893"
part2_max_frame = 200 #4893

# Define square areas
part1_areas = [(240, 100, 1008, 868), (878, 139, 1646, 907)]
part2_areas = [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]

# Create output directories
os.makedirs("train", exist_ok=True)
os.makedirs("val", exist_ok=True)

# Initialize COCO format data
coco_data_train = {"images": [], "annotations": [], "categories": []}
coco_data_val = {"images": [], "annotations": [], "categories": []}

annotation_id = 1

# Function to parse Pascal VOC XML and return objects within defined areas
def parse_voc_xml(xml_path, areas):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        for area in areas:
            if is_bbox_in_area((xmin, ymin, xmax, ymax), area):
                obj_class = int(obj.find("name").text)
                objects.append({"class": obj_class, "bbox": (xmin, ymin, xmax, ymax)})
                break

    return objects

# Function to check if a bounding box is within one of the defined areas
def is_bbox_in_area(bbox, area):
    xmin, ymin, xmax, ymax = bbox
    axmin, aymin, axmax, aymax = area

    return (xmin >= axmin and xmax <= axmax) and (ymin >= aymin and ymax <= aymax)

# Function to compute new bounding box coordinates after cropping
def compute_new_bbox_coordinates(bbox, crop_area):
    xmin, ymin, xmax, ymax = bbox
    axmin, aymin, axmax, aymax = crop_area

    return (xmin - axmin, ymin - aymin, xmax - axmin, ymax - aymin)

# Function to split dataset into train and val based on 80/20 split
def get_split(xml_file_count, part_max_frame):
    split_point = int(xml_file_count * 0.8)
    train_frame_limit = int(part_max_frame * 0.8)
    return split_point, train_frame_limit

def process_part(part_name, xml_folder, frame_folder, areas, part_max_frame):
    global annotation_id

    xml_files = sorted(glob.glob(os.path.join(xml_folder, "*.xml")))
    xml_file_count = len(xml_files)
    split_point, train_frame_limit = get_split(xml_file_count, part_max_frame)
    print(train_frame_limit)

    for i, xml_file in enumerate(xml_files):
        frame_number = int(os.path.splitext(os.path.basename(xml_file))[0])

        # Parse
        objects = parse_voc_xml(xml_file, areas)
        if not objects:
            continue

        # Load the corresponding frame as an image
        frame_file = os.path.join(frame_folder, f"{frame_number:08}.png")
        frame_img = Image.open(frame_file)

        for area in areas:
            # Crop the image based on the area
            cropped_img = frame_img.crop(area)
            axmin, aymin, axmax, aymax = area

            for obj in objects:
                if is_bbox_in_area(obj["bbox"], area):
                    # Save the cropped image in the train or val folder
                    new_bbox = compute_new_bbox_coordinates(obj["bbox"], area)
                    output_folder = "train" if i < split_point else "val"
                    output_filename = f"{part_name}_{frame_number}_{obj['class']}.png"
                    cropped_img.save(os.path.join(output_folder, output_filename))

                    # Save the annotation data in COCO format
                    annotation = {
                        "id": annotation_id,
                        "image_id": frame_number,
                        "category_id": obj["class"],
                        "bbox": new_bbox,
                        "area": (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]),
                        "iscrowd": 0,
                    }

                    image_data = {
                        "id": frame_number,
                        "file_name": output_filename,
                        "width": axmax - axmin,
                        "height": aymax - aymin,
                    }

                    if output_folder == "train":
                        coco_data_train["annotations"].append(annotation)
                        coco_data_train["images"].append(image_data)
                    else:
                        coco_data_val["annotations"].append(annotation)
                        coco_data_val["images"].append(image_data)

                    annotation_id += 1

#Process part 1 and part 2
process_part("part1", part1_xml, part1_frames, part1_areas, part1_max_frame)
process_part("part2", part2_xml, part2_frames, part2_areas, part2_max_frame)

#Create categories for COCO format
for i in range(9):
    category = {"id": i, "name": str(i), "supercategory": "object"}
coco_data_train["categories"].append(category)
coco_data_val["categories"].append(category)

#Save COCO format data to JSON files
with open("train.json", "w") as train_json_file:
    json.dump(coco_data_train, train_json_file)

with open("val.json", "w") as val_json_file:
    json.dump(coco_data_val, val_json_file)