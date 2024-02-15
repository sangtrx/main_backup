import os
import time
import csv
from instabot import Bot

# Instagram credentials
username = 'xsquared.lab'
password = 'Sangdeptraivl123456789*'

# Input files and directories
input_csv1 = "/home/tqsang/x2_lab/image.csv"
image_dir1 = "/mnt/tqsang/x2_lab/insta/2023-03-28"
input_csv2 = "/home/tqsang/x2_lab/abnormal.csv"
image_dir2 = "/mnt/tqsang/x2_lab/insta/ab/2023-03-28"

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

def post_images(input_csv, image_dir, captions, bot):
    for filename in sorted(os.listdir(image_dir)):
        # Check if the file is an image
        if filename.endswith(".jpg"):
            index = int(filename.split("-")[0])
            caption = captions.get(index)

            if caption:
                # Add hashtags to the caption with a line break
                if "abnormal" in input_csv:
                hashtags = "#weird #creative #fyp #trend #AI #artificialintelligence #machinelearning #deeplearning #AIimage #AIart #AIphotography #AIportrait #AIchallenge #AImeme #AIquotes #AIinfluencer #explorepage #explore #meme #trending"
                whatdoyouwantnext = "What do you want me to generate next? Comment below!"
                final_caption = f"{caption}\n \n{whatdoyouwantnext}\n \n \n \n \n \n \n \n{hashtags}"
                else:
                hashtags = "#AI #artificialintelligence #machinelearning #deeplearning #AIimage #AIart #AIphotography #AIportrait #AIchallenge #AImeme #AIquotes #AIinfluencer #explorepage #explore #meme #trending"
                final_caption = f"{caption}\n \n \n \n \n \n {hashtags}"
            
                # Upload the image and add caption
                image_path = os.path.join(image_dir, filename)
                bot.upload_photo(image_path, caption=final_caption)

                # Wait 1 hour before posting the next image
                time.sleep(60 * 60)

                # Mark image as posted by renaming it

