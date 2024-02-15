from pathlib import Path
import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
import math

def crop_remaining_frames(img_folder, output_folder, areas):
    Path(output_folder).mkdir(exist_ok=True)

    for img_file in sorted(os.listdir(img_folder)):
        frame_number = int(img_file[:-4])

        img = Image.open(os.path.join(img_folder, img_file))

        for i, area in enumerate(areas):
            cropped = img.crop(area)
            cropped_name = f"{frame_number:08d}_{'L' if i == 0 else 'R'}.png"
            cropped.save(os.path.join(output_folder, cropped_name))

root_folder = "/mnt/tqsang"
part1_folder = os.path.join(root_folder, "chicken_part1")
part2_folder = os.path.join(root_folder, "chicken_part2")

part1_img_folder = os.path.join(part1_folder, "frames")
part2_img_folder = os.path.join(part2_folder, "frames")

areas_part1 = [(240, 100, 1008, 868), (878, 139, 1646, 907)]
areas_part2 = [(633, 535, 1913, 1815), (1901, 586, 3181, 1866)]

output_folder_part1 = "/mnt/tqsang/chicken_part1/part1_cropped_frames"
output_folder_part2 = "/mnt/tqsang/chicken_part2/part2_cropped_frames"

crop_remaining_frames(part1_img_folder, output_folder_part1, areas_part1)
crop_remaining_frames(part2_img_folder, output_folder_part2, areas_part2)
