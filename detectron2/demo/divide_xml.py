import os
import shutil

# Set the path to the folder containing the XML files
source_folder = '/mnt/tqsang/chicken_part1/xml'
destination_folder_base = '/mnt/tqsang/chicken_part1/xml'

# Get the list of XML files and sort them
xml_files = [file_name for file_name in os.listdir(source_folder) if file_name.endswith('.xml')]
xml_files.sort()

# Calculate the number of files to be placed in each subfolder
total_files = len(xml_files)
files_per_folder = total_files // 10 + (1 if total_files % 10 > 0 else 0)

# Create 10 subfolders and distribute the files
for i in range(10):
    # Create the subfolder
    subfolder_path = os.path.join(destination_folder_base, f'xml{i + 1}')
    os.makedirs(subfolder_path, exist_ok=True)

    # Calculate the range of files to be moved to the current subfolder
    start_index = i * files_per_folder
    end_index = start_index + files_per_folder

    # Move the files to the subfolder
    for file_name in xml_files[start_index:end_index]:
        old_file_path = os.path.join(source_folder, file_name)
        new_file_path = os.path.join(subfolder_path, file_name)
        shutil.move(old_file_path, new_file_path)

print("XML files have been divided into 10 smaller folders.")
