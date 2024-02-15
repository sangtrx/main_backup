import os
import shutil
from pathlib import Path

def create_yolo_test_dataset(input_image_folder, input_txt_folder, filelist_path, output_dataset_folder):
    # Create the output folders if they don't exist
    Path(output_dataset_folder / "images").mkdir(parents=True, exist_ok=True)
    Path(output_dataset_folder / "labels").mkdir(parents=True, exist_ok=True)

    # Read the filelist.txt with the correct encoding
    with open(filelist_path, "r", encoding="ISO-8859-1") as file:
        filelist = file.readlines()

    for file_name in filelist:
        file_name = file_name.strip()

        # Copy the corresponding image to the dataset
        shutil.copy(input_image_folder / file_name, output_dataset_folder / "images")

        # Read the corresponding .txt file, normalize the values, and save it to the dataset
        txt_file_path = input_txt_folder / file_name.replace(".png", ".txt")
        with open(txt_file_path, "r") as txt_file:
            lines = txt_file.readlines()

        img = Image.open(input_image_folder / file_name)
        width, height = img.size

        with open(output_dataset_folder / "labels" / file_name.replace(".png", ".txt"), "w") as output_txt:
            for line in lines:
                values = line.strip().split(" ")
                cls, x, y, w, h = int(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4])

                x /= width
                y /= height
                w /= width
                h /= height

                output_txt.write(f"{cls} {x} {y} {w} {h}\n")

# Define the paths
part1_image_folder = Path("/mnt/tqsang/chicken_part1/part1_cropped_frames")
part1_txt_folder = Path("/mnt/tqsang/part1_filename/txt")
part1_filelist_path = Path("/mnt/tqsang/part1_filename/filelist.txt")

part2_image_folder = Path("/mnt/tqsang/chicken_part2/part2_cropped_frames")
part2_txt_folder = Path("/mnt/tqsang/part2_filename/txt")
part2_filelist_path = Path("/mnt/tqsang/part2_filename/filelist.txt")

output_dataset_folder = Path("/mnt/tqsang/data_chick_part_YOLO/datasets/test")

# Create the YOLO test dataset
create_yolo_test_dataset(part1_image_folder, part1_txt_folder, part1_filelist_path, output_dataset_folder)
create_yolo_test_dataset(part2_image_folder, part2_txt_folder, part2_filelist_path, output_dataset_folder)
