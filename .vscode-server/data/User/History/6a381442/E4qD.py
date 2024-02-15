import os
import time
import csv
from instabot import Bot

# Instagram credentials
username = 'xsquared.lab'
password = 'Sangdeptraivl123456789*'

# Input files and directory
input_csv = "/home/tqsang/x2_lab/image.csv"
image_dir = "/mnt/tqsang/x2_lab/insta/2023-03-28"

# Read captions from the CSV file
def read_captions(input_csv):
    captions = {}
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            index = row[0]
            caption = row[1]
            captions[index] = caption
    return captions

def main():
    # Initialize bot and login
    bot = Bot()
    bot.login(username=username, password=password)

    captions = read_captions(input_csv)

    # Loop through the image files
    for filename in sorted(os.listdir(image_dir)):
        # Check if the file is an image
        if filename.endswith(".jpg"):
            index = filename.split("-")[0]
            caption = captions.get(index)

            # Add hashtags to the caption
            hashtags = "#AI #artificialintelligence #machinelearning #deeplearning #AIimage #AIart #AIphotography #AIportrait #AIchallenge #AImeme #AIquotes #AIinfluencer #explorepage #explore #meme"
            final_caption = f"{caption} \n {hashtags}"

            # Upload the image and add caption
            image_path = os.path.join(image_dir, filename)
            bot.upload_photo(image_path, caption=final_caption)

            # Wait 2 hours before posting the next image
            time.sleep(2 * 60 * 60)

            # Mark image as posted by renaming it
            os.rename(image_path, f"{image_path}.DONE")

if __name__ == "__main__":
    main()
