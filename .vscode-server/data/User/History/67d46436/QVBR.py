import os
import glob
from collections import defaultdict
from lxml import etree

def process_xml_files(xml_folder, max_frame):
    class_stats = defaultdict(int)
    
    for frame_num in range(max_frame):
        xml_filename = f"{frame_num:08}.xml"
        xml_filepath = os.path.join(xml_folder, xml_filename)

        if os.path.exists(xml_filepath):
            with open(xml_filepath, "r") as xml_file:
                tree = etree.parse(xml_file)
                root = tree.getroot()

                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    class_stats[class_name] += 1
        else:
            break
            
    return class_stats

def main():
    xml_folder1 = "/mnt/tqsang/chicken_part1/xml_44606"
    xml_folder2 = "/mnt/tqsang/chicken_part2/xml1_4893"
    max_frame1 = 44606
    max_frame2 = 4893

    class_stats1 = process_xml_files(xml_folder1, max_frame1)
    class_stats2 = process_xml_files(xml_folder2, max_frame2)

    # Merge the statistics from both parts
    total_stats = defaultdict(int)
    for k, v in class_stats1.items():
        total_stats[k] += v
    for k, v in class_stats2.items():
        total_stats[k] += v

    print("Class statistics:")
    for class_name, count in total_stats.items():
        print(f"Class {class_name}: {count}")

if __name__ == "__main__":
    main()
