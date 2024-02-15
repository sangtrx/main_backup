import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import math

def is_in_area(bbox, area):
    x1, y1, x2, y2 = bbox
    ax1, ay1, ax2, ay2 = area
    return ax1 <= x1 and ax2 >= x2 and ay1 <= y1 and ay2 >= y2

def main():
    part1_max_frame = 44606
    part2_max_frame = 4893
    part1_xml_folder = '/mnt/tqsang/chicken_part1/xml_44606'
    part2_xml_folder = '/mnt/tqsang/chicken_part2/xml1_4893'
    part1_frame_folder = '/mnt/tqsang/chicken_part1/frames'
    part2_frame_folder = '/mnt/tqsang/chicken_part2/frames'

    part1_area1 = (240, 100, 1008, 868)
    part1_area2 = (878, 139, 1646, 907)
    part2_area1 = (633, 535, 1913, 1815)
    part2_area2 = (1901, 586, 3181, 1866)

    output_train_folder = 'train'
    output_val_folder = 'val'
    train_json_file = 'train.json'
    val_json_file = 'val.json'

    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_val_folder, exist_ok=True)

    part1_xml_count = sum(1 for f in os.listdir(part1_xml_folder) if f.endswith('.xml') and int(f[:-4]) < part1_max_frame)
    part2_xml_count = sum(1 for f in os.listdir(part2_xml_folder) if f.endswith('.xml') and int(f[:-4]) < part2_max_frame)

    part1_train_count = math.floor(0.8 * part1_xml_count)
    part2_train_count = math.floor(0.8 * part2_xml_count)

    categories = [{"id": i, "name": str(i)} for i in range(9)]

    train_json = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    val_json = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    annotation_id = 1

    for part_id, (max_frame, xml_folder, frame_folder, train_count, areas) in enumerate([
        (part1_max_frame, part1_xml_folder, part1_frame_folder, part1_train_count, (part1_area1, part1_area2)),
        (part2_max_frame, part2_xml_folder, part2_frame_folder, part2_train_count, (part2_area1, part2_area2)),
    ], 1):
        for frame_number in range(max_frame):
            xml_file = f'{frame_number:08d}.xml'
            xml_path = os.path.join(xml_folder, xml_file)

            if not os.path.exists(xml_path):
                continue

            image_path = os.path.join(frame_folder, f'{frame_number:08d}.png')
            image = Image.open(image_path)

            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                bbox_elem = obj.find('bndbox')
                bbox = [
                    int(round(float(bbox_elem.find('xmin').text))),
                    int(round(float(bbox_elem.find('ymin').text))),
                    int(round(float(bbox_elem.find('xmax').text))),
                    int(round(float(bbox_elem.find('ymax').text)))
                ]

                obj_class = int(obj.find('name').text)

                for area_idx, area in enumerate(areas):
                    if is_in_area(bbox, area):
                        ax1, ay1, ax2, ay2 = area
                        cropped_image = image.crop(area)
                        new_bbox = [bbox[0] - ax1, bbox[1] - ay1, bbox[2] - ax1, bbox[3] - ay1]
                        output_filename = f'part{part_id}_{frame_number:08d}_class{obj_class}_area{area_idx + 1}.png'

                        if frame_number < train_count:
                            output_folder = output_train_folder
                            output_json = train_json
                        else:
                            output_folder = output_val_folder
                            output_json = val_json

                        cropped_image.save(os.path.join(output_folder, output_filename))

                        image_id = len(output_json['images']) + 1
                        output_json['images'].append({
                            "id": image_id,
                            "license": 1,
                            "file_name": output_filename,
                            "height": ay2 - ay1,
                            "width": ax2 - ax1
                        })

                        output_json['annotations'].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": obj_class,
                            "bbox": new_bbox,
                            "area": (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]),
                            "iscrowd": 0
                        })

                        annotation_id += 1

    with open(train_json_file, 'w') as f:
        json.dump(train_json, f)

    with open(val_json_file, 'w') as f:
        json.dump(val_json, f)

if __name__ == '__main__':
    main()
