import os
import time
import csv
from instabot import Bot

# Instagram credentials
username = 'xsquared.lab'
password = 'Sangdeptraivl123456789*'

# Read captions from the CSV file
def read_captions(input_csv):
    captions = {}
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            index = int(row[0]) - 1  # Adjust index to match CSV format
            caption = row[1]
            captions[index] = caption
    return captions

def main(input_csv, image_dir):
    # Initialize bot and login
    bot = Bot()
    bot.login(username=username, password=password)

    captions = read_captions(input_csv)

    # Loop through the image files
    for filename in sorted(os.listdir(image_dir)):
        # Check if the file is an image
        if filename.endswith(".jpg"):
            index = int(filename.split("-")[0])
            caption = captions.get(index)

            if caption:
                # Upload the image and add caption
                image_path = os.path.join(image_dir, filename)
                bot.upload_photo(image_path, caption=caption)

                # Wait 2 hours before posting the next image
                # time.sleep(2 * 60 * 60)

                # Mark image as posted by renaming it
                # os.rename(image_path, f"{image_path}.DONE")

if __name__ == "__main__":
    input_csv1 = "/home/tqsang/x2_lab/image.csv"
    image_dir1 = "/mnt/tqsang/x2_lab/insta/2023-03-28"
    input_csv2 = "/home/tqsang/x2_lab/abnormal.csv"
    image_dir2 = "/mnt/tqsang/x2_lab/insta/ab/2023-03-28"

    while True:
        main(input_csv1, image_dir1)
        time.sleep(1 * 60 * 60)  # Wait 1 hour before posting the next set of images
        main(input_csv2, image_dir2)
        time.sleep(1 * 60 * 60)  # Wait 1 hour before posting the next set of images


