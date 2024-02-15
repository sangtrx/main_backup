import os
import cv2
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

def is_inside_area(bbox, areas):
    for area in areas:
        x1, y1, x2, y2 = area
        if x1 <= bbox[0] < x2 and y1 <= bbox[1] < y2:
            return True
    return False

def pascal_voc_to_coco(xml_path, img_path, img_id, ann_id, areas):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    
    coco_image = {
        "id": img_id,
        "license": 1,
        "file_name": os.path.basename(img_path),
        "height": height,
        "width": width,
        "date_captured": None
    }
    
    coco_annotations = []
    for obj in root.findall("object"):
        class_id = int(obj.find("name").text)
        
        bbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = [
            round(float(bbox.find(coord).text)) for coord in ["xmin", "ymin", "xmax", "ymax"]
        ]
        
        if is_inside_area((xmin, ymin, xmax, ymax), areas):
            for area in areas:
                x1, y1, x2, y2 = area
                if x1 <= xmin < x2 and y1 <= ymin < y2:
                    xmin -= x1
                    xmax -= x1
                    ymin -= y1
                    ymax -= y1
                    break
            
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            area = bbox_width * bbox_height
            
            coco_annotation = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [xmin, ymin, bbox_width, bbox_height],
                "area": area,
                "iscrowd": 0
            }
            coco_annotations.append(coco_annotation)
            ann_id += 1
    
    return coco_image, coco_annotations, ann_id

def create_coco_dataset(parts_info, output_dir, train_val_split=0.8):
    coco_info = {
        "year": "2023",
        "version": "1.0",
        "description": "Custom COCO dataset",
        "contributor": "",
        "url": "",
        "date_created": "2023-04-18"
    }
    
    coco_licenses = [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ]
    
    coco_categories = [{"id": i, "name": str(i), "supercategory": "class"} for i in range(9)]

    img_id = 1
    ann_id = 1
    coco_images = []
    coco_annotations = []
    
    for part, part_info in parts_info.items():
        xml_files = [f for f in os.listdir(part_info["xml_dir"]) if f.endswith(".xml")]
        xml_files = sorted(xml_files, key=lambda f: int(f.split(".")[0]))[:part_info["max_frame"]]
        num_train = int(train_val_split * len(xml_files))
        
        for idx, xml_file in enumerate(xml_files):
            frame = xml_file[:-4] + ".            png"
            xml_path = os.path.join(part_info["xml_dir"], xml_file)
            img_path = os.path.join(part_info["frame_dir"], frame)
            
            if idx < num_train:
                output_subdir = os.path.join(output_dir, "train")
            else:
                output_subdir = os.path.join(output_dir, "val")
            
            Path(output_subdir).mkdir(parents=True, exist_ok=True)
            
            coco_image, coco_annotations_part, ann_id = pascal_voc_to_coco(xml_path, img_path, img_id, ann_id, part_info["areas"])
            coco_images.append(coco_image)
            coco_annotations.extend(coco_annotations_part)
            
            for annotation in coco_annotations_part:
                bbox = annotation["bbox"]
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                cropped_img = img[y1:y2, x1:x2]
                output_filename = f"{part}_{frame}_{annotation['category_id']}.png"
                output_path = os.path.join(output_subdir, output_filename)
                cv2.imwrite(output_path, cropped_img)
            
            img_id += 1
    
    coco_dataset = {
        "info": coco_info,
        "licenses": coco_licenses,
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations
    }
    
    return coco_dataset

parts_info = {
    "part1": {
        "xml_dir": "/mnt/tqsang/chicken_part1/xml_44606",
        "frame_dir": "/mnt/tqsang/chicken_part1/frames",
        "max_frame": 800, #44606,
        "areas": [(240, 100, 1008, 868), (878, 139, 1646, 907)]
    },
    "part2": {
        "xml_dir": "/mnt/tqsang/chicken_part2/xml1_4893",
        "frame_dir": "/mnt/tqsang/chicken_part2/frames",
        "max_frame": 200, # 4893,
        "areas": [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]
    }
}

output_dir = "output"

coco_dataset = create_coco_dataset(parts_info, output_dir)

train_dataset = {
    key: coco_dataset[key] if key != "images" and key != "annotations" else [] for key in coco_dataset
}
val_dataset = {
    key: coco_dataset[key] if key != "images" and key != "annotations" else [] for key in coco_dataset
}

for img, anns in zip(coco_dataset["images"], coco_dataset["annotations"]):
    if "train" in img["file_name"]:
        train_dataset["images"].append(img)
        train_dataset["annotations"].extend(anns)
    else:
        val_dataset["images"].append(img)
        val_dataset["annotations"].extend(anns)

with open(os.path.join(output_dir, "train.json"), "w") as f:
    json.dump(train_dataset, f)

with open(os.path.join(output_dir, "val.json"), "w") as f:
    json.dump(val_dataset, f)

