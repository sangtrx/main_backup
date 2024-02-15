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

def read_captions(input_csv):
    captions = {}
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            index = int(row[0]) - 1  # Adjust index to match CSV format
            caption = row[1]
            captions[index] = caption
    return captions

def post_image(bot, image_dir, input_csv, hashtags):
    captions = read_captions(input_csv)

    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".jpg"):
            index = int(filename.split("-")[0])
            caption = captions.get(index)

            if caption:
                final_caption = f"{caption}\n\n\n\n\n\n{hashtags}"
                image_path = os.path.join(image_dir, filename)
                bot.upload_photo(image_path, caption=final_caption)
                return # Post only one image and return

def main():
# Initialize bot and login
    bot = Bot()
    bot.login(username=username, password=password)
    while True:
        hashtags1 = "#AI #artificialintelligence #machinelearning #deeplearning #AIimage #AIart #AIphotography #AIportrait #AIchallenge #AImeme #AIquotes #AIinfluencer #explorepage #explore #meme #trending"
        post_image(bot, image_dir1, input_csv1, hashtags1)

        # Wait 1 hour before posting the next image
        time.sleep(1 * 60 * 60)

        hashtags2 = "#weird #creative #fyp #trend #AI #artificialintelligence #machinelearning #deeplearning #AIimage #AIart #AIphotography #AIportrait #AIchallenge #AImeme #AIquotes #AIinfluencer #explorepage #explore #meme #trending"
        whatdoyouwantnext = "What do you want me to generate next? Comment below!"
        hashtags2 = f"{hashtags2}\n\n{whatdoyouwantnext}"
        post_image(bot, image_dir2, input_csv2, hashtags2)

        # Wait 1 hour before posting the next image
        time.sleep(1 * 60 * 60)

if __name__ == "__main__":
    main()
