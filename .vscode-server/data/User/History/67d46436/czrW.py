import os
import xml.etree.ElementTree as ET

def get_class_stats(xml_folder):
    class_stats = {str(i): 0 for i in range(9)}

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()

            for obj in root.findall('object'):
                class_label = obj.find('name').text
                if class_label in class_stats:
                    class_stats[class_label] += 1

    return class_stats

video_folders = [
    ("/mnt/tqsang/chicken_part1/part1.mp4", "/mnt/tqsang/chicken_part1/xml_44606"),
    ("/mnt/tqsang/chicken_part2/part2.mp4", "/mnt/tqsang/chicken_part2/xml1_4893")
]

total_class_stats = {str(i): 0 for i in range(9)}

for video, xml_folder in video_folders:
    class_stats = get_class_stats(xml_folder)
    for class_label, count in class_stats.items():
        total_class_stats[class_label] += count

print("Class statistics:")
for class_label, count in total_class_stats.items():
    print(f"Class {class_label}: {count} instances")
