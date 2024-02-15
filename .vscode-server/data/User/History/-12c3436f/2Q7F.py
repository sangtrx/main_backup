import os
import json
import xml.etree.ElementTree as ET
from PIL import Image


def get_xml_files(xml_folder, max_frame_number):
    xml_files = []
    for frame_number in range(max_frame_number):
        xml_filename = f"{frame_number:08d}.xml"
        if os.path.exists(os.path.join(xml_folder, xml_filename)):
            xml_files.append(xml_filename)
    return xml_files


def process_square_areas(obj, areas):
    for area_idx, area in enumerate(areas):
        xmin, ymin, xmax, ymax = map(int, obj["bbox"])
        if xmin >= area[0] and ymin >= area[1] and xmax <= area[2] and ymax <= area[3]:
            obj["bbox"] = [xmin - area[0], ymin - area[1], xmax - area[0], ymax - area[1]]
            return area_idx
    return -1


def save_coco_dataset(name, xml_folder, frames_folder, xml_files, areas, train_xml_files, output_train_folder, output_val_folder, image_id=1, annotation_id=1):
    images = []
    annotations = []

    for xml_filename in xml_files:
        xml_filepath = os.path.join(xml_folder, xml_filename)

        tree = ET.parse(xml_filepath)
        root = tree.getroot()

        frame_number = int(os.path.splitext(xml_filename)[0])
        img_filename = os.path.join(frames_folder, f"{frame_number:08d}.png")
        img = Image.open(img_filename)


        for obj in root.findall("object"):
            category_id = int(obj.find("name").text)
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            obj_dict = {
                "category_id": category_id,
                "bbox": [xmin, ymin, xmax, ymax],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0,
                "id": annotation_id,
            }

            area_idx = process_square_areas(obj_dict, areas)
            if area_idx != -1:
                cropped_img = img.crop(areas[area_idx])
                output_img_filename = f"{name}_{frame_number:08d}_{category_id}.png"
                output_img_filepath = os.path.join(output_train_folder if xml_filename in train_xml_files else output_val_folder, output_img_filename)
                cropped_img.save(output_img_filepath)

                images.append({
                    "id": image_id,
                    "file_name": output_img_filename,
                    "width": cropped_img.width,
                    "height": cropped_img.height,
                })

                obj_dict["image_id"] = image_id
                annotations.append(obj_dict)

                image_id += 1
                annotation_id += 1

    return images, annotations


def main():
    part1_frames_folder = "/mnt/tqsang/chicken_part1/frames"
    part1_xml_folder = "/mnt/tqsang/chicken_part1/xml_44606"
    part1_max_frame_number = 1000 #44606
    part1_areas = [(240, 100, 1008, 868), (878, 139, 1646, 907)]

    part2_frames_folder = "/mnt/tqsang/chicken_part2/frames"
    part2_xml_folder = "/mnt/tqsang/chicken_part2/xml1_4893"
    part2_max_frame_number = 800 #4893
    part2_areas = [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]

    output_train_folder = "train"
    output_val_folder = "val"
    train_ratio = 0.8

    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_val_folder, exist_ok=True)

    categories = [{"id": i, "name": str(i)} for i in range(9)]

    train_annotations = []
    val_annotations = []
    all_images = []
    image_id = 1
    annotation_id = 1

    for name, frames_folder, xml_folder, max_frame_number, areas in [
        ("part1", part1_frames_folder, part1_xml_folder, part1_max_frame_number, part1_areas),
        ("part2", part2_frames_folder, part2_xml_folder, part2_max_frame_number, part2_areas),
    ]:
        xml_files = get_xml_files(xml_folder, max_frame_number)
        train_xml_files = xml_files[:int(len(xml_files) * train_ratio)]
        val_xml_files = xml_files[int(len(xml_files) * train_ratio):]

        images, annotations = save_coco_dataset(name, xml_folder, frames_folder, xml_files, areas, train_xml_files, output_train_folder, output_val_folder, image_id=image_id, annotation_id=annotation_id)
   

        train_annotations.extend([ann for ann in annotations if ann["image_id"] in [img["id"] for img in images if img["file_name"] in train_xml_files]])
        val_annotations.extend([ann for ann in annotations if ann["image_id"] in [img["id"] for img in images if img["file_name"] in val_xml_files]])

        all_images.extend(images)
        image_id += len(images)
        annotation_id += len(annotations)

    train_images = [img for img in all_images if img["file_name"] in train_xml_files]
    val_images = [img for img in all_images if img["file_name"] in val_xml_files]

    train_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }

    val_json = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
    }

    with open("train.json", "w") as f:
        json.dump(train_json, f)

    with open("val.json", "w") as f:
        json.dump(val_json, f)

if __name__ == "__main__":
    main()

