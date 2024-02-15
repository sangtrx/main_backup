import os

# Set the path to the folder containing the files
folder_path = '/mnt/tqsang/chicken_part2/xml/'

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is an XML file
    if file_name.endswith('.xml'):
        # Find the first '0' in the file name and remove it
        new_file_name = file_name.replace('0', '', 1)
        
        # Create the full paths for the old and new file names
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)

print("File renaming completed.")
