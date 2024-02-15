import os
import shutil

# Source directories
source_dirs = [
    '/mnt/tqsang/chicken_part1/frames',
    '/mnt/tqsang/chicken_part2/frames'
]

# Destination directory
destination_dir = '/mnt/tqsang/scale_sample'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Copy the first 100 files from each source directory
for source_dir in source_dirs:
    file_list = os.listdir(source_dir)
    for i, file_name in enumerate(file_list):
        if i >= 100:
            break
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.copy2(source_path, destination_path)  # Use shutil.copy() for older Python versions

print("Files copied successfully.")
